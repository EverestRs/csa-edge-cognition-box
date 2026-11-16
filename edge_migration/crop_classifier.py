from __future__ import annotations

"""Crop 多分类仲裁器（edge 档视觉判定的确定性核心，01 §2.3）。

- 输入：整帧 + bbox 列表（[x,y,w,h]）；letterbox 到 224 后逐 crop 推理；
- 后端：edge_migration.inference_backend.create_backend——OM 优先、.onnx 兜底、
  全失败则 NullBackend（verdict 返回 Unknown，调用方按"未复核"处理）；
- 模型输出约定：logits (1, K) 或与 classes 等长且首维为 batch 的张量（自动 softmax）；
  概率已归一的 sigmoid 输出（(1, K) 且 max<=1）按原值使用。
"""

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .inference_backend import create_backend

DEFAULT_CLASSES = ["Normal", "M01", "M02", "H03", "Wave", "Throw", "Handshake"]


@dataclass
class CropVerdict:
    cls: str
    conf: float

    def to_dict(self) -> dict[str, Any]:
        return {"cls": self.cls, "conf": round(float(self.conf), 3)}


class CropClassifier:
    def __init__(self, config: dict[str, Any]):
        cfg = dict(config or {})
        self.model_path = str(cfg.get("model_path", "models/crop/crop_v1.onnx"))
        self.fallback_onnx = cfg.get("fallback_onnx") or None
        self.input_size = int(cfg.get("input_size", 224))
        self.classes = [str(c) for c in cfg.get("classes", DEFAULT_CLASSES)]
        # 类专属接受阈值（缺省 accept_conf）；低于 reject_conf 视为"未确认"
        self.accept_conf = float(cfg.get("accept_conf", 0.60))
        self.reject_conf = float(cfg.get("reject_conf", 0.45))
        # Normal 翻案门槛：crop 判 Normal 需达到此置信才允许推翻 PC 的异常判定
        # （弱证据不翻案——parity 实测 H03 被低置信 Normal 误翻 24 次后加入）
        self.normal_rebate_conf = float(cfg.get("normal_rebate_conf", 0.75))
        # 仲裁类集：仅这些类可确认/翻案 PC 结果；其余（演示类 Wave/Throw/Handshake）
        # 只服务 NL 布控目标判定，不参与灰区仲裁（域差大，parity 实测 Throw 误触发 28 次）
        self.arbiter_classes = {str(c) for c in cfg.get("arbiter_classes", self.classes)}
        # Normal 不可翻案类（安防语义：漏报代价>误报；H03 被教师偏差拖累，crop 判 Normal
        # 不可信——维持 PC 结果交后续复核/规则验证）
        self.no_rebate_classes = {str(c) for c in cfg.get("no_rebate_classes", [])}
        self.class_thresholds = {str(k): float(v) for k, v in (cfg.get("class_thresholds") or {}).items()}
        self.throttle_frames = int(max(1, cfg.get("throttle_frames", 12)))
        self.runtime = dict(cfg.get("runtime", {}))
        self.pad_value = int(cfg.get("pad_value", 114))

        self._backend = create_backend(self.model_path, fallback_onnx=self.fallback_onnx,
                                       runtime=self.runtime)
        self.session_init_error = getattr(self._backend, "last_error", "")
        self._cache: dict[str, tuple[int, CropVerdict]] = {}
        self.last_debug: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    @property
    def is_real_model_loaded(self) -> bool:
        return bool(getattr(self._backend, "is_real", False))

    def _preprocess(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x, y, bw, bh = (int(v) for v in bbox[:4])
        x, y = max(0, x), max(0, y)
        crop = frame[y:min(h, y + max(bh, 1)), x:min(w, x + max(bw, 1))]
        if crop.size == 0:
            crop = np.full((self.input_size, self.input_size, 3), self.pad_value, dtype=np.uint8)
        ch, cw = crop.shape[:2]
        scale = min(self.input_size / cw, self.input_size / ch)
        nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), self.pad_value, dtype=np.uint8)
        ox, oy = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        return np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]  # (1,3,S,S)

    def _postprocess(self, outputs: list[np.ndarray]) -> CropVerdict:
        if not outputs:
            return CropVerdict("Unknown", 0.0)
        # 选与类数匹配的输出张量
        vec: np.ndarray | None = None
        for o in outputs:
            arr = np.asarray(o).squeeze()
            if arr.ndim == 1 and arr.size == len(self.classes):
                vec = arr
                break
            if arr.ndim >= 2 and arr.shape[-1] == len(self.classes):
                vec = arr.reshape((-1, arr.shape[-1]))[0]  # batch=1 → 取首行
                break
        if vec is None:
            return CropVerdict("Unknown", 0.0)
        vec = vec.astype(np.float32)
        if vec.max() > 1.0 or vec.min() < 0.0:          # 视为 logits → softmax
            vec = np.exp(vec - vec.max())
            vec = vec / vec.sum()
        elif abs(float(vec.sum()) - 1.0) > 0.05:        # 概率未归一（sigmoid 类头）→ 归一
            vec = vec / max(float(vec.sum()), 1e-6)
        idx = int(np.argmax(vec))
        return CropVerdict(self.classes[idx], float(vec[idx]))

    def threshold_for(self, cls: str) -> float:
        return self.class_thresholds.get(cls, self.accept_conf)

    def classify(self, frame: np.ndarray, bboxes: list[list[float]],
                 track_ids: list[str] | None = None, frame_id: int = -1) -> list[CropVerdict]:
        """逐 bbox 推理；未加载真实模型时返回 Unknown（不误伤主路径结果）。"""
        t0 = time.perf_counter()
        if not self.is_real_model_loaded:
            self.last_debug = {"error": self.session_init_error or "model_not_loaded",
                               "count": len(bboxes)}
            return [CropVerdict("Unknown", 0.0) for _ in bboxes]
        verdicts: list[CropVerdict] = []
        for i, bbox in enumerate(bboxes):
            try:
                outs = self._backend.run(self._preprocess(frame, bbox))
                verdicts.append(self._postprocess(outs))
            except Exception as exc:
                verdicts.append(CropVerdict("Unknown", 0.0))
                self.last_debug["run_error"] = str(exc)
        self.last_debug.update({
            "count": len(verdicts),
            "batch_ms": round((time.perf_counter() - t0) * 1000, 2),
            "backend": getattr(self._backend, "kind", "?"),
        })
        if frame_id >= 0 and track_ids:
            for tid, v in zip(track_ids, verdicts):
                self._cache[tid] = (frame_id, v)
        return verdicts

    def cached(self, track_id: str, frame_id: int) -> CropVerdict | None:
        hit = self._cache.get(str(track_id))
        if hit and 0 <= frame_id - hit[0] <= self.throttle_frames:
            return hit[1]
        return None

    def confirm(self, verdict: CropVerdict) -> bool:
        """是否达到该类的接受阈值（Normal 不参与"确认"语义）。"""
        if verdict.cls in ("", "Unknown", "Normal"):
            return False
        return verdict.conf >= self.threshold_for(verdict.cls)
