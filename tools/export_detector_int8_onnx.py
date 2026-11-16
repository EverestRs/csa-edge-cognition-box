from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
from ultralytics import YOLO

from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)


@dataclass(frozen=True)
class ExportSpec:
    name: str
    pt_path: Path
    fp32_onnx_path: Path
    int8_onnx_path: Path


class VideoCalibrationDataReader(CalibrationDataReader):
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
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        return np.transpose(tensor, (2, 0, 1))[None, ...]

    def _open_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open calibration video: {self.video_path}")

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
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def _export_fp32_onnx(pt_path: Path, fp32_onnx_path: Path, imgsz: int, opset: int) -> Path:
    fp32_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(pt_path))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exported_path = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=True,
            dynamic=False,
            half=False,
            verbose=False,
        )
    exported = Path(exported_path)
    if not exported.exists():
        raise RuntimeError(f"fp32 export failed: {exported}")
    if exported.resolve() != fp32_onnx_path.resolve():
        fp32_onnx_path.write_bytes(exported.read_bytes())
    return fp32_onnx_path


def _detect_input_shape(model_path: Path) -> tuple[str, tuple[int, int]]:
    model = onnx.load(str(model_path))
    if not model.graph.input:
        raise RuntimeError(f"onnx input not found: {model_path}")
    input_tensor = model.graph.input[0]
    input_name = input_tensor.name
    dims = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.dim_value > 0 else 0)
    if len(dims) != 4:
        raise RuntimeError(f"unexpected detector input rank: {dims}")
    height = int(dims[2]) if dims[2] > 0 else 640
    width = int(dims[3]) if dims[3] > 0 else 640
    return input_name, (width, height)


def _summarize_quantization(model_path: Path) -> dict[str, Any]:
    model = onnx.load(str(model_path))
    op_counts: dict[str, int] = {}
    init_types: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    for initializer in model.graph.initializer:
        dtype = onnx.TensorProto.DataType.Name(initializer.data_type)
        init_types[dtype] = init_types.get(dtype, 0) + 1
    return {
        "nodes": {
            "QuantizeLinear": op_counts.get("QuantizeLinear", 0),
            "DequantizeLinear": op_counts.get("DequantizeLinear", 0),
            "ConvInteger": op_counts.get("ConvInteger", 0),
            "MatMulInteger": op_counts.get("MatMulInteger", 0),
        },
        "initializer_types": init_types,
    }


def _quantize_static(fp32_onnx_path: Path, int8_onnx_path: Path, video_path: Path, max_frames: int, sample_step: int) -> dict[str, Any]:
    input_name, input_size = _detect_input_shape(fp32_onnx_path)
    reader = VideoCalibrationDataReader(video_path, input_name, input_size, max_frames=max_frames, sample_step=sample_step)
    int8_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        extra_options={
            "ForceQuantizeNoInputCheck": True,
            "MatMulConstBOnly": True,
        },
    )
    return _summarize_quantization(int8_onnx_path)


def _run_one(spec: ExportSpec, video_path: Path, imgsz: int, opset: int, max_frames: int, sample_step: int, overwrite: bool) -> dict[str, Any]:
    if spec.int8_onnx_path.exists() and not overwrite:
        return {
            "name": spec.name,
            "status": "skipped_exists",
            "int8_onnx": str(spec.int8_onnx_path),
            "summary": _summarize_quantization(spec.int8_onnx_path),
        }

    with tempfile.TemporaryDirectory(prefix=f"detector_export_{spec.name}_", dir=str(BASE_DIR / "results")) as tmp_dir:
        tmp_path = Path(tmp_dir)
        fp32_tmp = tmp_path / f"{spec.name}_fp32.onnx"
        _export_fp32_onnx(spec.pt_path, fp32_tmp, imgsz=imgsz, opset=opset)
        summary = _quantize_static(fp32_tmp, spec.int8_onnx_path, video_path, max_frames=max_frames, sample_step=sample_step)
        if spec.fp32_onnx_path.exists() and overwrite:
            try:
                spec.fp32_onnx_path.unlink()
            except Exception:
                pass
        if not spec.fp32_onnx_path.exists():
            spec.fp32_onnx_path.write_bytes(fp32_tmp.read_bytes())
        return {
            "name": spec.name,
            "status": "exported",
            "pt": str(spec.pt_path),
            "fp32_onnx": str(spec.fp32_onnx_path),
            "int8_onnx": str(spec.int8_onnx_path),
            "summary": summary,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO detector .pt files to INT8 ONNX with ORT static quantization")
    parser.add_argument("--workspace", default=str(BASE_DIR), help="project workspace root")
    parser.add_argument("--video", default=str(BASE_DIR / ".." / "data" / "laying_backward_1-1.mp4"), help="calibration video path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--calibration-max-frames", type=int, default=0, help="0 means use all sampled frames")
    parser.add_argument("--calibration-sample-step", type=int, default=1)
    parser.add_argument("--model-names", nargs="*", default=None, help="detector variants to export, e.g. yolov8n yolov8x")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    workspace = _resolve_path(args.workspace)
    video_path = _resolve_path(args.video)
    model_dir = workspace / "models" / "detector"
    specs = [
        ExportSpec(
            name="yolov8n",
            pt_path=model_dir / "yolov8n.pt",
            fp32_onnx_path=model_dir / "yolov8n.onnx",
            int8_onnx_path=model_dir / "yolov8n_int8.onnx",
        ),
        ExportSpec(
            name="yolov8x",
            pt_path=model_dir / "yolov8x.pt",
            fp32_onnx_path=model_dir / "yolov8x.onnx",
            int8_onnx_path=model_dir / "yolov8x_int8.onnx",
        ),
    ]
    if args.model_names:
        wanted = {name.strip() for name in args.model_names if name.strip()}
        specs = [spec for spec in specs if spec.name in wanted]

    results = []
    for spec in specs:
        if not spec.pt_path.exists():
            raise FileNotFoundError(f"missing pt: {spec.pt_path}")
        results.append(
            _run_one(
                spec=spec,
                video_path=video_path,
                imgsz=args.imgsz,
                opset=args.opset,
                max_frames=max(0, args.calibration_max_frames),
                sample_step=max(1, args.calibration_sample_step),
                overwrite=bool(args.overwrite),
            )
        )

    print({
        "video": str(video_path),
        "workspace": str(workspace),
        "results": results,
    })


if __name__ == "__main__":
    main()
