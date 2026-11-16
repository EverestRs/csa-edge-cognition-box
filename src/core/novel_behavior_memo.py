from __future__ import annotations

"""新行为备忘 (NovelBehaviorMemo) —— ROADMAP Stage 1。

目的：在闭集 11 类之外"察觉并描述"未知行为，为 Stage 2 原型沉淀供料。
事件级触发（区别于 SceneNarrator 的 20s 场景采样），证据锚定落盘（描述+证据对，
不存裸文本）。

触发器:
  A 叙述器驱动：SceneNarrator 的 notable_activities 文本不含任何闭集行为关键词
    → 疑似体系外活动，按文本指纹冷却后提交场景级备忘；
  B 分类器驱动：track 的主路径输出为 Normal 且置信度落在不确定带
    [gray_low, gray_high)（分类器"说不清"），按 track 冷却提交目标级备忘。

数据流（旁路，不阻塞推理）:
  engine.on_frame → 更新 track 历史（bbox/关键点）→ 触发判定 → 入队
  → 单工作线程取 ±window_seconds 环形缓冲帧 + 原帧 person crop + 姿态统计
  → vLLM 结构化填空 prompt（参与/部位/模式/速度/物体/与正常区别）
  → results/novel/{task}/novel_{frame_id}[_{track}].json + evidence/*.jpg

Stage 3 依赖：目标级备忘持久化归一化姿态序列（pose_seq 字段），供伪标注直接组装训练数据。
"""

import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from src.core.vlm_client import parse_json_loose, vlm_chat

# 闭集行为关键词（叙述文本命中任一即视为"已归类"；与 tools/eval_narration_consistency.py 同口径）
_CLOSED_SET_KEYWORDS = {
    "打架", "斗殴", "互殴", "挥拳", "打斗", "霸凌", "欺凌", "欺负", "殴打",
    "跌倒", "摔倒", "倒地", "晕倒", "昏厥", "躺地", "轻生", "跳楼", "翻越", "爬高",
    "自缢", "破坏", "砸", "踹", "踢门", "吸烟", "抽烟", "香烟", "打火机",
    "手机", "打电话", "看手机", "玩手机", "遮挡", "挡住", "徘徊", "踱步",
    "来回走", "逗留", "围栏", "闯入",
}

# COCO-17 身体组（133 点格式下追加手部组）
_BODY_GROUPS = {
    "头部": [0, 1, 2, 3, 4],
    "手臂": [5, 6, 7, 8, 9, 10],
    "下肢": [11, 12, 13, 14, 15, 16],
}
_HAND_GROUP = list(range(91, 133))  # 133 点格式的手部关键点
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_WRIST, _R_WRIST = 9, 10

_MEMO_PROMPT = (
    "你是校园安防分析员。以下是按时间顺序排列的监控画面，"
    "疑似包含一段尚未归类的行为，请观察并填空输出JSON（不要输出其他文字）：\n"
    '{"summary": "<一句话描述具体动作过程，手臂/身体如何运动>", "participants": "<单人/多人(数量)>", '
    '"body_parts": "<主要涉及身体部位>", "motion_pattern": "<一次性/重复/持续>", '
    '"speed": "<慢/中/快>", "object_interaction": "<无/接触了什么>", '
    '"vs_normal": "<该动作与普通站立走动的区别>", '
    '"is_novel": <true或false>, "confidence": <0到1>}\n'
    "is_novel 的判定门槛是**是否存在一个具体可描述的动作**（任何明确动作都算，"
    "挥手、敬礼、投掷、两人之间的互动（握手/递物/推拉）等不限异常与否）；"
    "仅当画面只是普通站立/走动、没有任何具体动作时才填 false。这是为行为库记录新动作，不是异常检测。"
    "只输出一个JSON对象（不是数组）。"
    "若提供了运动统计参考，请与画面互相印证后作答。"
)


def _pose_stats(pose_frames: list[np.ndarray], fps: float) -> dict[str, Any]:
    """姿态序列确定性统计（坐标为 x/W, y/H 归一化口径）。"""
    if len(pose_frames) < 3:
        return {}
    arrs = [np.asarray(f, dtype=np.float32) for f in pose_frames if f is not None and len(f)]
    arrs = [a for a in arrs if a.ndim == 2 and a.shape[0] >= 17 and a.shape[1] >= 2]
    if len(arrs) < 3:
        return {}
    n_dim = arrs[0].shape[0]
    disp_total, disp_groups = [], {g: [] for g in _BODY_GROUPS}
    disp_hands = []
    for prev, cur in zip(arrs, arrs[1:]):
        d = np.linalg.norm(cur[:, :2] - prev[:, :2], axis=1)
        disp_total.append(float(d[:17].mean()))
        for g, idx in _BODY_GROUPS.items():
            if n_dim > max(idx):
                disp_groups[g].append(float(d[idx].mean()))
        if n_dim >= 133:
            disp_hands.append(float(d[_HAND_GROUP].mean()))
    parts = sorted(((g, float(np.mean(v))) for g, v in disp_groups.items() if v),
                   key=lambda kv: -kv[1])
    if disp_hands:
        hand_score = float(np.mean(disp_hands))
        parts.append(("手部", hand_score))
        parts.sort(key=lambda kv: -kv[1])
    # 手-肩距离序列 → 重复性提示（局部峰次/秒）
    wrist_series: list[float] = []
    for a in arrs:
        candidates = []
        for w_idx, s_idx in ((_L_WRIST, _L_SHOULDER), (_R_WRIST, _R_SHOULDER)):
            if n_dim > max(w_idx, s_idx):
                candidates.append(float(np.linalg.norm(a[w_idx, :2] - a[s_idx, :2])))
        if candidates:
            wrist_series.append(min(candidates))
    rep_hint = None
    if len(wrist_series) >= 6 and fps > 0:
        arr = np.asarray(wrist_series)
        thr = arr.mean() + 0.5 * arr.std()
        peaks = sum(1 for i in range(1, len(arr) - 1) if arr[i] > thr and arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1])
        rep_hint = round(peaks * fps / len(arr), 2)
    stats = {
        "n_pose_frames": len(arrs),
        "motion_energy": round(float(np.mean(disp_total)), 5),
        "body_parts_activity": [(g, round(v, 5)) for g, v in parts[:3]],
    }
    if rep_hint is not None:
        stats["repetition_hint_peaks_per_sec"] = rep_hint
    return stats


class NovelBehaviorMemo:
    def __init__(self, vlm_config: dict, output_dir: str = "results"):
        cfg = vlm_config.get("novel_behavior", {}) if isinstance(vlm_config, dict) else {}
        self.enabled = bool(cfg.get("enabled", True))
        self.window_seconds = float(cfg.get("window_seconds", 5))
        self.scene_frames = int(max(2, cfg.get("scene_frames", 4)))
        self.crop_frames = int(max(1, cfg.get("crop_frames", 3)))
        self.max_image_px = int(cfg.get("max_image_px", 768))
        self.gray_low = float(cfg.get("gray_low", 0.40))
        self.gray_high = float(cfg.get("gray_high", 0.55))
        self.track_cooldown_frames = int(max(1, cfg.get("track_cooldown_frames", 200)))
        self.activity_cooldown_seconds = float(cfg.get("activity_cooldown_seconds", 60))
        self.max_queue = int(max(1, cfg.get("max_queue", 3)))
        self.max_latency_ms = int(cfg.get("max_latency_ms", 20000))
        self.backoff_seconds = float(cfg.get("failure_backoff_seconds", 30))

        self.use_vllm = bool(vlm_config.get("use_vllm", False))
        self.vllm_url = str(vlm_config.get("vllm_url", "http://localhost:8001/v1/chat/completions"))
        self.vllm_api_key = str(vlm_config.get("vllm_api_key", "EMPTY"))
        self.model_name = str(vlm_config.get("model_name", "qwen2.5-vl-7b-instruct"))
        if not self.use_vllm:
            self.enabled = False
            self._last_error = "novel_memo_requires_vllm_mode"

        self._source_fps = 25.0
        self._frame_buffer = None
        self._frame_buffer_orig = None
        self._vlm_router = None
        self._pose_buffers: dict | None = None
        self._prototypes: Any | None = None

        self._track_histories: dict[str, deque] = {}
        self._track_last_memo: dict[str, int] = {}
        self._activity_last: dict[str, float] = {}
        self._queue: deque[dict[str, Any]] = deque()
        self._inflight = False
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="novel-memo")

        self.current_task_id = ""
        self._out_root = Path(output_dir) / "novel"
        self._out_dir = self._out_root
        self._dropped_total = 0
        self._memo_count = 0
        self._last_error = ""
        self._backoff_until = 0.0

    # ── 引擎接线 ─────────────────────────────────────────────

    def attach_buffer(self, frame_buffer) -> None:
        self._frame_buffer = frame_buffer

    def attach_buffer_orig(self, frame_buffer_orig) -> None:
        self._frame_buffer_orig = frame_buffer_orig

    def attach_vlm_router(self, vlm_router) -> None:
        self._vlm_router = vlm_router

    def attach_pose_buffers(self, pose_buffers: dict) -> None:
        """ActionRecognizer.track_pose_buffers（归一化姿态序列，供统计与 Stage 3 落盘）。"""
        self._pose_buffers = pose_buffers

    def attach_prototypes(self, prototypes) -> None:
        """BehaviorPrototypes（Stage 2 再识别：新备忘自动归入既有原型）。"""
        self._prototypes = prototypes

    def set_source_fps(self, fps: float) -> None:
        self._source_fps = float(fps if fps and fps > 1.0 else 25.0)

    def begin_task(self, task_id: str) -> None:
        with self._lock:
            self.current_task_id = str(task_id)
            self._out_dir = self._out_root / str(task_id)
            self._out_dir.mkdir(parents=True, exist_ok=True)
            (self._out_dir / "evidence").mkdir(exist_ok=True)
            self._track_histories.clear()
            self._track_last_memo.clear()
            self._activity_last.clear()
            self._queue.clear()
            self._dropped_total = 0

    # ── 主入口（推理线程调用，非阻塞） ────────────────────────

    def on_frame(self, frame_id: int, tracks: list[dict]) -> None:
        """每处理帧调用：更新目标历史 + 触发器 B（分类器不确定带）。"""
        if not self.enabled:
            return
        with self._lock:
            for t in tracks:
                tid = str(t.get("track_id", ""))
                if not tid:
                    continue
                hist = self._track_histories.setdefault(tid, deque(maxlen=96))
                hist.append((
                    int(frame_id),
                    [float(v) for v in t.get("bbox", [0, 0, 0, 0])],
                    t.get("keypoints"),
                ))
            # 触发器 B：Normal 且低置信 → 分类器"说不清"
            for t in tracks:
                tid = str(t.get("track_id", ""))
                label = str(t.get("label", "Normal"))
                conf = float(t.get("confidence", 0.0))
                if label != "Normal" or not (self.gray_low <= conf < self.gray_high):
                    continue
                if frame_id - self._track_last_memo.get(tid, -10 ** 9) < self.track_cooldown_frames:
                    continue
                self._track_last_memo[tid] = int(frame_id)
                self._enqueue({
                    "kind": "track",
                    "track_id": tid,
                    "trigger": "gray_normal",
                    "frame_id": int(frame_id),
                    "label": label,
                    "confidence": conf,
                })
            self._try_dispatch()

    def note_narrator_activities(self, activities: list, frame_id: int) -> None:
        """触发器 A：叙述器 notable_activities 中疑似体系外的活动文本。"""
        if not self.enabled:
            return
        video_t = frame_id / self._source_fps
        for text in activities or []:
            text = str(text).strip()[:200]
            if not text or any(kw in text for kw in _CLOSED_SET_KEYWORDS):
                continue
            fp = text[:16]
            with self._lock:
                if video_t - self._activity_last.get(fp, -10 ** 9) < self.activity_cooldown_seconds:
                    continue
                self._activity_last[fp] = video_t
                self._enqueue({
                    "kind": "scene",
                    "trigger": "narrator_activity",
                    "frame_id": int(frame_id),
                    "activity_text": text,
                })
        with self._lock:
            self._try_dispatch()

    def _enqueue(self, item: dict[str, Any]) -> None:
        """调用方已持有 _lock。"""
        if len(self._queue) >= self.max_queue:
            self._queue.popleft()
            self._dropped_total += 1
        self._queue.append(item)

    def _try_dispatch(self) -> None:
        """调用方需持有 _lock（或无并发风险时）。取槽成功即投递工作线程。"""
        if self._inflight or not self._queue:
            return
        if time.time() < self._backoff_until:
            return
        if self._vlm_router is not None and not self._vlm_router.try_acquire(timeout=0.0):
            return
        item = self._queue.popleft()
        self._inflight = True
        self._executor.submit(self._memo_job, item)

    def _requeue(self, item: dict[str, Any]) -> None:
        """失败重排队（有限次）：备忘一旦丢失该目标/活动就要等下一轮冷却。"""
        attempts = int(item.get("attempts", 0)) + 1
        if attempts > 2:
            return
        item["attempts"] = attempts
        if len(self._queue) < self.max_queue:
            self._queue.append(item)

    def poke(self) -> None:
        """外部驱动重派发（drain/评估用）：退避到期后重新尝试队列。"""
        with self._lock:
            self._try_dispatch()

    # ── 工作线程 ─────────────────────────────────────────────

    def _collect_frames(self, item: dict[str, Any]) -> tuple[list, list]:
        """取 ±window 场景帧；目标级再从原帧缓冲裁 person crop。"""
        frame_id = int(item["frame_id"])
        span = int(self.window_seconds * self._source_fps)
        window = []
        if self._frame_buffer is not None:
            window = self._frame_buffer.window_between(max(1, frame_id - span), frame_id)
            if len(window) < 2 and self._frame_buffer is not None:
                window = self._frame_buffer.recent_entries(self.scene_frames)
        k = min(self.scene_frames, len(window))
        if window:
            # 均匀采样而非取尾部：短片段内动作可能只在窗口中段出现
            stride = max(1, len(window) // k)
            picked = window[::stride][:k]
            if len(picked) < k:
                picked += window[-(k - len(picked)):]
            scene = [f for _, f in picked]
        else:
            scene = []
        crops = []
        if item.get("kind") == "track" and self._frame_buffer_orig is not None:
            # 原帧缓冲按帧存全图，crop 依据历史 bbox 逐帧裁剪
            hist = self._track_histories.get(str(item.get("track_id", "")), deque())
            bboxes = [(fid, bb) for fid, bb, _ in hist if abs(fid - frame_id) <= span]
            if bboxes:
                orig_entries = self._frame_buffer_orig.recent_entries(len(bboxes) * 2)
                orig_by_frame = {fid: f for fid, f in orig_entries}
                for fid, bb in bboxes[-self.crop_frames:]:
                    full = orig_by_frame.get(fid)
                    if full is None:
                        continue
                    x, y, w, h = [int(round(v)) for v in bb]
                    h_f, w_f = full.shape[:2]
                    crop = full[max(0, y):min(h_f, y + h), max(0, x):min(w_f, x + w)]
                    if crop.size:
                        crops.append(crop)
        return scene, crops

    def _build_stats(self, item: dict[str, Any]) -> dict[str, Any]:
        if item.get("kind") != "track" or self._pose_buffers is None:
            return {}
        pose_frames = list(self._pose_buffers.get(str(item.get("track_id", "")), []))
        return _pose_stats(pose_frames, self._source_fps)

    def _build_prompt(self, item: dict[str, Any], stats: dict[str, Any]) -> str:
        lines = [f"[触发来源] {'目标级备忘（行为分类器置信度不足）' if item['kind'] == 'track' else '场景叙述报告的未归类活动'}"]
        if item["kind"] == "track":
            lines.append(f"- 检测模块当前输出: {item.get('label')}（置信度 {item.get('confidence'):.2f}，处于不确定区间）")
        else:
            lines.append(f"- 场景叙述报告: {item.get('activity_text', '')}")
        if stats:
            lines.append(f"- 姿态统计（归一化口径）: 运动能量={stats.get('motion_energy')}, "
                         f"部位活动度={stats.get('body_parts_activity')}, "
                         f"重复性提示(峰次/秒)={stats.get('repetition_hint_peaks_per_sec')}")
        return "\n".join(lines) + "\n\n" + _MEMO_PROMPT

    def _memo_job(self, item: dict[str, Any]) -> None:
        start = time.perf_counter()
        try:
            scene, crops = self._collect_frames(item)
            if not scene and not crops:
                return
            stats = self._build_stats(item)
            frames = [*scene[-self.scene_frames:], *crops[-self.crop_frames:]]
            text = vlm_chat(
                base_url=self.vllm_url,
                api_key=self.vllm_api_key,
                model=self.model_name,
                prompt=self._build_prompt(item, stats),
                frames=frames,
                max_tokens=384,
                timeout=self.max_latency_ms / 1000.0,
                max_image_px=self.max_image_px,
            )
            parsed = parse_json_loose(text) or {}
            is_novel = bool(parsed.get("is_novel", False))
            record = {
                "type": "novel_behavior_memo",
                "task_id": self.current_task_id,
                "kind": item["kind"],
                "trigger": item.get("trigger", ""),
                "frame_id": int(item["frame_id"]),
                "video_time_sec": round(item["frame_id"] / self._source_fps, 2),
                "track_id": item.get("track_id", ""),
                "detection_hint": {"label": item.get("label"), "confidence": item.get("confidence")},
                "activity_text": item.get("activity_text", ""),
                "is_novel": is_novel,
                "memo": {k: str(parsed.get(k, ""))[:200] for k in
                         ("summary", "participants", "body_parts", "motion_pattern",
                          "speed", "object_interaction", "vs_normal")},
                "confidence": float(parsed.get("confidence", 0.0)),
                "pose_stats": stats,
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "model": self.model_name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if is_novel:
                evidence = self._save_evidence(item, scene, crops)
                record["evidence"] = evidence
                if item.get("kind") == "track" and self._pose_buffers is not None:
                    buf = list(self._pose_buffers.get(str(item.get("track_id", "")), []))
                    if buf:
                        npz = Path(self._out_dir) / f"novel_{item['frame_id']}_{item.get('track_id', 'x')}_pose.npz"
                        np.savez_compressed(npz, pose_seq=np.stack(buf).astype(np.float32))
                        record["pose_seq_file"] = str(npz)
                # ROADMAP Stage 2：再识别——新备忘相似度命中既有原型则归簇
                if self._prototypes is not None:
                    try:
                        proto_id = self._prototypes.ingest_memo_record(record)
                        if proto_id:
                            record["prototype_id"] = proto_id
                    except Exception:
                        pass
            self._persist(item, record)
            with self._lock:
                self._memo_count += 1
            self._last_error = ""
        except Exception as exc:
            self._last_error = f"memo_failed: {exc}"
            with self._lock:
                self._backoff_until = time.time() + self.backoff_seconds
                self._requeue(item)
        finally:
            if self._vlm_router is not None:
                self._vlm_router.release()
            with self._lock:
                self._inflight = False
                self._try_dispatch()

    def _save_evidence(self, item: dict[str, Any], scene: list, crops: list) -> list[str]:
        import cv2
        saved = []
        prefix = f"novel_{item['frame_id']}" + (f"_{item.get('track_id', '')}" if item.get("kind") == "track" else "")
        for i, f in enumerate(scene[-self.scene_frames:]):
            path = self._out_dir / "evidence" / f"{prefix}_scene{i}.jpg"
            cv2.imwrite(str(path), f)
            saved.append(str(path))
        for i, c in enumerate(crops[-self.crop_frames:]):
            path = self._out_dir / "evidence" / f"{prefix}_crop{i}.jpg"
            cv2.imwrite(str(path), c)
            saved.append(str(path))
        return saved

    def _persist(self, item: dict[str, Any], record: dict[str, Any]) -> None:
        suffix = f"_{item.get('track_id', '')}" if item.get("kind") == "track" else ""
        path = Path(self._out_dir) / f"novel_{item['frame_id']}{suffix}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 读取接口 ─────────────────────────────────────────────

    def iter_memos(self, task_id: str | None = None) -> list[dict[str, Any]]:
        """读取备忘记录（Stage 2 聚类输入；novel=true 的记录含 evidence/pose_seq 引用）。"""
        root = self._out_root / str(task_id) if task_id else self._out_dir
        if not root.exists():
            return []
        out = []
        for path in sorted(root.glob("novel_*.json")):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return out

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "use_vllm": self.use_vllm,
                "healthy": bool(self.enabled and not self._last_error),
                "queue_length": len(self._queue),
                "inflight": self._inflight,
                "dropped_total": self._dropped_total,
                "memo_count": self._memo_count,
                "tracked_targets": len(self._track_histories),
                "current_task_id": self.current_task_id,
                "last_error": self._last_error,
            }

    def stop(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
