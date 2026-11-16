from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)


class PoseVideoCalibrationDataReader(CalibrationDataReader):
    def __init__(self, video_path: Path, input_name: str, input_size: tuple[int, int], max_frames: int = 0, sample_step: int = 1):
        self.video_path = Path(video_path)
        self.input_name = input_name
        self.input_size = input_size
        self.max_frames = int(max_frames)
        self.sample_step = max(1, int(sample_step))
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._emitted = 0
        self._open_capture()

    def _open_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open calibration video: {self.video_path}")

    @staticmethod
    def _letterbox(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        target_w, target_h = size
        height, width = frame.shape[:2]
        scale = min(target_w / max(1, width), target_h / max(1, height))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y: pad_y + new_h, pad_x: pad_x + new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        return np.transpose(tensor, (2, 0, 1))[None, ...]

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._cap is None:
            self._open_capture()

        while True:
            ok, frame = self._cap.read()
            if not ok:
                return None
            self._frame_id += 1
            if self._frame_id % self.sample_step != 0:
                continue
            self._emitted += 1
            if self.max_frames > 0 and self._emitted > self.max_frames:
                return None
            return {self.input_name: self._letterbox(frame, self.input_size)}

    def rewind(self) -> None:
        self._frame_id = 0
        self._emitted = 0
        self._open_capture()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def _detect_input(model_path: Path) -> tuple[str, tuple[int, int]]:
    model = onnx.load(str(model_path))
    if not model.graph.input:
        raise RuntimeError(f"onnx input not found: {model_path}")
    tensor = model.graph.input[0]
    dims = [dim.dim_value if dim.dim_value > 0 else 0 for dim in tensor.type.tensor_type.shape.dim]
    if len(dims) != 4:
        raise RuntimeError(f"unexpected pose input rank: {dims}")
    h = int(dims[2]) if dims[2] > 0 else 256
    w = int(dims[3]) if dims[3] > 0 else 192
    return tensor.name, (w, h)


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
        },
        "initializer_types": inits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pose ONNX to INT8 with static quantization")
    parser.add_argument("--onnx", default="models/pose/vitpose_wholebody_b.onnx")
    parser.add_argument("--output", default="models/pose/vitpose_wholebody_b_int8.onnx")
    parser.add_argument("--video", default=str(BASE_DIR / ".." / "data" / "laying_backward_1-1.mp4"))
    parser.add_argument("--calibration-max-frames", type=int, default=0)
    parser.add_argument("--calibration-sample-step", type=int, default=1)
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

    input_name, input_size = _detect_input(fp32_onnx)
    reader = PoseVideoCalibrationDataReader(video_path, input_name, input_size, max_frames=max(0, args.calibration_max_frames), sample_step=max(1, args.calibration_sample_step))
    int8_onnx.parent.mkdir(parents=True, exist_ok=True)
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
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        extra_options={"ForceQuantizeNoInputCheck": True},
    )
    print({"status": "exported", "input": str(fp32_onnx), "output": str(int8_onnx), "summary": _summarize_quantization(int8_onnx)})


if __name__ == "__main__":
    main()