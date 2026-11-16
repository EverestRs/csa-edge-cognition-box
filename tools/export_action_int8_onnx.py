from __future__ import annotations

import argparse
from collections import deque
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)


class SlowFastVideoCalibrationDataReader(CalibrationDataReader):
    def __init__(self, video_path: Path, input_specs: list[tuple[str, tuple[int, ...]]], max_clips: int = 0, clip_step: int = 4):
        self.video_path = Path(video_path)
        self.input_specs = input_specs
        self.max_clips = int(max_clips)
        self.clip_step = max(1, int(clip_step))
        self._cap: cv2.VideoCapture | None = None
        self._clip_count = 0
        self._frame_id = 0
        self._buffer = deque()
        self._open_capture()

    def _open_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open calibration video: {self.video_path}")

    @staticmethod
    def _preprocess_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        target_w, target_h = size
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.225, 0.225, 0.225], dtype=np.float32).reshape(3, 1, 1)
        return (chw - mean) / np.maximum(std, 1e-6)

    @staticmethod
    def _fit_temporal(clip: np.ndarray, target_t: int) -> np.ndarray:
        if clip.shape[0] == target_t:
            return clip
        if clip.shape[0] > target_t:
            idx = np.linspace(0, clip.shape[0] - 1, num=target_t, dtype=np.int32)
            return clip[idx]
        pad = np.repeat(clip[-1:], target_t - clip.shape[0], axis=0)
        return np.concatenate([clip, pad], axis=0)

    def _build_feed(self) -> dict[str, np.ndarray] | None:
        if len(self._buffer) < 32:
            return None
        fast_clip = np.stack(list(self._buffer)[-32:], axis=0).astype(np.float32)
        slow_clip = fast_clip[::4] if fast_clip.shape[0] > 1 else fast_clip

        feed: dict[str, np.ndarray] = {}
        for name, shape in self.input_specs:
            if len(shape) != 5:
                continue
            c = int(shape[1]) if isinstance(shape[1], int) and shape[1] > 0 else 3
            t = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else fast_clip.shape[0]
            h = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else fast_clip.shape[2]
            w = int(shape[4]) if isinstance(shape[4], int) and shape[4] > 0 else fast_clip.shape[3]
            source = slow_clip if t <= 8 or "slow" in name.lower() else fast_clip
            source = self._fit_temporal(source, t)
            if source.shape[1:] != (3, h, w):
                source = np.stack([cv2.resize(np.transpose(frame, (1, 2, 0)), (w, h), interpolation=cv2.INTER_LINEAR) for frame in source], axis=0)
                source = np.transpose(source.astype(np.float32), (0, 3, 1, 2))
            if c != 3:
                source = source[:, :c, :, :]
            # ONNX SlowFast expects [N, C, T, H, W], not [N, T, C, H, W].
            feed[name] = np.expand_dims(np.transpose(source, (1, 0, 2, 3)), axis=0).astype(np.float32)
        return feed or None

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._cap is None:
            self._open_capture()

        while True:
            ok, frame = self._cap.read()
            if not ok:
                return None
            self._frame_id += 1
            if self._frame_id % self.clip_step != 0:
                continue
            self._buffer.append(self._preprocess_frame(frame, (224, 224)))
            if len(self._buffer) > 32:
                self._buffer.popleft()
            feed = self._build_feed()
            if feed is None:
                continue
            self._clip_count += 1
            if self.max_clips > 0 and self._clip_count > self.max_clips:
                return None
            return feed

    def rewind(self) -> None:
        self._clip_count = 0
        self._frame_id = 0
        self._buffer.clear()
        self._open_capture()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def _detect_inputs(model_path: Path) -> list[tuple[str, tuple[int, ...]]]:
    model = onnx.load(str(model_path))
    specs: list[tuple[str, tuple[int, ...]]] = []
    for item in model.graph.input:
        dims = tuple(dim.dim_value if dim.dim_value > 0 else 0 for dim in item.type.tensor_type.shape.dim)
        specs.append((item.name, dims))
    if not specs:
        raise RuntimeError(f"onnx inputs not found: {model_path}")
    return specs


def _summarize_quantization(model_path: Path) -> dict[str, Any]:
    model = onnx.load(str(model_path))
    nodes: dict[str, int] = {}
    inits: dict[str, int] = {}
    for node in model.graph.node:
        nodes[node.op_type] = nodes.get(node.op_type, 0) + 1
    for init in model.graph.initializer:
        dtype = onnx.TensorProto.DataType.Name(init.data_type)
        inits[dtype] = inits.get(dtype, 0) + 1
    return {
        "nodes": {
            "QuantizeLinear": nodes.get("QuantizeLinear", 0),
            "DequantizeLinear": nodes.get("DequantizeLinear", 0),
            "ConvInteger": nodes.get("ConvInteger", 0),
            "MatMulInteger": nodes.get("MatMulInteger", 0),
        },
        "initializer_types": inits,
    }


def _quantize_dynamic(fp32_onnx_path: Path, int8_onnx_path: Path) -> dict[str, Any]:
    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        extra_options={"WeightSymmetric": True},
    )
    return _summarize_quantization(int8_onnx_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SlowFast ONNX to INT8 with static quantization")
    parser.add_argument("--onnx", default="models/action/slowfast.onnx")
    parser.add_argument("--output", default="models/action/slowfast_int8.onnx")
    parser.add_argument("--video", default=str(BASE_DIR / ".." / "data" / "laying_backward_1-1.mp4"))
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--clip-step", type=int, default=4)
    parser.add_argument("--minimum-real-range", type=float, default=1e-4)
    parser.add_argument("--allow-dynamic-fallback", action="store_true")
    parser.add_argument("--quant-mode", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    fp32_onnx = _resolve_path(args.onnx)
    int8_onnx = _resolve_path(args.output)
    video_path = _resolve_path(args.video)
    if not fp32_onnx.exists():
        raise FileNotFoundError(f"missing fp32 onnx: {fp32_onnx}")

    if int8_onnx.exists() and not args.overwrite:
        print({"status": "skipped_exists", "output": str(int8_onnx), "summary": _summarize_quantization(int8_onnx)})
        return

    if args.quant_mode == "dynamic":
        int8_onnx.parent.mkdir(parents=True, exist_ok=True)
        summary = _quantize_dynamic(fp32_onnx, int8_onnx)
        print({"status": "exported_dynamic", "input": str(fp32_onnx), "output": str(int8_onnx), "summary": summary})
        return

    input_specs = _detect_inputs(fp32_onnx)
    reader = SlowFastVideoCalibrationDataReader(video_path, input_specs, max_clips=max(0, args.max_clips), clip_step=max(1, args.clip_step))
    int8_onnx.parent.mkdir(parents=True, exist_ok=True)
    try:
        quantize_static(
            model_input=str(fp32_onnx),
            model_output=str(int8_onnx),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QOperator,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
            calibrate_method=CalibrationMethod.MinMax,
            op_types_to_quantize=["Conv"],
            extra_options={
                "ForceQuantizeNoInputCheck": True,
                "MinimumRealRange": float(max(1e-8, args.minimum_real_range)),
            },
        )
        status = "exported_static"
    except Exception as exc:
        if not args.allow_dynamic_fallback:
            if int8_onnx.exists():
                try:
                    int8_onnx.unlink()
                except Exception:
                    pass
            raise
        if int8_onnx.exists():
            try:
                int8_onnx.unlink()
            except Exception:
                pass
        summary = _quantize_dynamic(fp32_onnx, int8_onnx)
        print({"status": "exported_dynamic_fallback", "reason": str(exc), "input": str(fp32_onnx), "output": str(int8_onnx), "summary": summary})
        return

    print({"status": status, "input": str(fp32_onnx), "output": str(int8_onnx), "summary": _summarize_quantization(int8_onnx)})


if __name__ == "__main__":
    main()