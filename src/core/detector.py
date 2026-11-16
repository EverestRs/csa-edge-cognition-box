from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


class Detector:
    def __init__(self, config: dict[str, Any]):
        self.name = config.get("name", "yolov8x")
        self.confidence_threshold = float(config.get("confidence_threshold", 0.35))
        self.iou_threshold = float(config.get("iou_threshold", 0.5))
        self.onnx_path = str(config.get("active_onnx_path") or config.get("onnx_path") or "")
        self.onnx_path_resolved = self._resolve_model_path(self.onnx_path)
        self.input_tensor_name = str(config.get("input_tensor_name", "images"))
        self.output_tensor_names = list(config.get("output_tensor_names", ["output0"]))
        self.input_size = tuple(config.get("input_size", [640, 640]))
        self.runtime_cfg = dict(config.get("runtime", {}))
        self.cv_threads = int(max(1, self.runtime_cfg.get("opencv_num_threads", 1)))
        self.preprocess_optimized = bool(config.get("preprocess_optimized", True))
        self.class_mapping = dict(config.get("class_mapping", {"person": "person"}))
        self.target_class_name = self.class_mapping.get("person", "person")
        self._canvas_buffer: np.ndarray | None = None
        self._rgb_buffer: np.ndarray | None = None
        self._tensor_buffer: np.ndarray | None = None

        # Keep OpenCV threading deterministic so detector-side benchmarks are reproducible.
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(self.cv_threads)

        self.runtime_input_tensor_name: str = self.input_tensor_name
        self.runtime_output_tensor_names: list[str] = list(self.output_tensor_names)
        self.session_init_error = ""
        self.om_degraded_to_onnx = False  # OM 不可用回退 ONNX 时置 True（debug 可见）
        self.session = self._build_session(self.onnx_path_resolved)
        self.is_real_model_loaded = self.session is not None
        if self.session is not None:
            try:
                input_meta = self.session.get_inputs()
                input_names = [item.name for item in input_meta]
                output_meta = self.session.get_outputs()
                output_names = [item.name for item in output_meta]
                if input_names:
                    self.runtime_input_tensor_name = self.input_tensor_name if self.input_tensor_name in input_names else input_names[0]
                if output_names:
                    valid = [name for name in self.output_tensor_names if name in output_names]
                    self.runtime_output_tensor_names = valid if valid else output_names
            except Exception:
                pass
        self.last_debug: dict[str, Any] = {
            "is_real_model_loaded": self.is_real_model_loaded,
            "used_fallback": not self.is_real_model_loaded,
            "error": "" if self.is_real_model_loaded else (self.session_init_error or "onnx_session_not_ready"),
            "input_tensor_name": self.input_tensor_name,
            "output_tensor_names": self.output_tensor_names,
            "runtime_input_tensor_name": self.runtime_input_tensor_name,
            "runtime_output_tensor_names": list(self.runtime_output_tensor_names),
            "raw_output_shape": None,
            "detections_before_threshold": 0,
            "detections_after_threshold": 0,
            "detections_after_nms": 0,
            "effective_person_boxes": 0,
        }

    def _resolve_model_path(self, model_path: str) -> str:
        if not model_path:
            return ""

        path = Path(model_path)
        if path.is_absolute():
            return str(path)

        project_root = Path(__file__).resolve().parents[2]
        return str((project_root / path).resolve())

    def _build_session(self, onnx_path: str):
        if ort is None:
            self.session_init_error = "onnxruntime_import_failed"
            return None

        if not onnx_path:
            self.session_init_error = "onnx_path_empty"
            return None

        # 昇腾 OM 后端（CANN 路径；x86 缺 pyACL 时自动回退同源 ONNX，静态检查级）
        if onnx_path.endswith(".om"):
            return self._build_om_session(onnx_path)

        path = Path(onnx_path)
        if not path.exists():
            self.session_init_error = "onnx_file_not_found"
            return None

        graph_opt = str(self.runtime_cfg.get("graph_optimization_level", "ORT_ENABLE_ALL")).upper()
        graph_opt_map = {
            "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        graph_opt_candidates = [
            graph_opt_map.get(graph_opt, ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ]
        seen_levels: set[int] = set()

        available = set(ort.get_available_providers())
        provider = str(self.runtime_cfg.get("execution_provider", "CPUExecutionProvider"))
        device = str(self.runtime_cfg.get("device", "cpu")).lower()
        providers: list[str] = []
        if provider in available:
            providers.append(provider)
        elif device in {"cuda", "gpu"} and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        if not providers:
            providers = ["CPUExecutionProvider"]

        intra_threads = int(max(0, self.runtime_cfg.get("intra_op_num_threads", 0)))
        inter_threads = int(max(0, self.runtime_cfg.get("inter_op_num_threads", 0)))
        last_error = "onnx_session_create_failed"

        for level in graph_opt_candidates:
            level_id = int(level)
            if level_id in seen_levels:
                continue
            seen_levels.add(level_id)
            try:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = level
                if intra_threads > 0:
                    session_options.intra_op_num_threads = intra_threads
                if inter_threads > 0:
                    session_options.inter_op_num_threads = inter_threads
                session_options.enable_mem_pattern = bool(self.runtime_cfg.get("enable_mem_pattern", True))
                session_options.enable_cpu_mem_arena = bool(self.runtime_cfg.get("enable_cpu_mem_arena", True))
                return ort.InferenceSession(str(path), sess_options=session_options, providers=providers)
            except Exception as exc:
                last_error = f"onnx_session_create_failed: {exc}"

        try:
            return ort.InferenceSession(str(path), providers=providers)
        except Exception as exc:
            last_error = f"onnx_session_create_failed: {exc}"

        self.session_init_error = last_error
        return None

    def _om_fallback_onnx(self, om_path: str) -> str:
        """OM 不可用时的同源 ONNX 回退：yolov8n_310B.om → yolov8n_310B.onnx → yolov8n.onnx。"""
        p = Path(om_path)
        candidates = [p.with_suffix(".onnx")]
        if p.stem.endswith("_310B"):
            candidates.append(p.with_name(p.stem[:-5] + ".onnx"))
        for c in candidates:
            if c.exists():
                return str(c)
        return ""

    def _build_om_session(self, om_path: str):
        """昇腾 OM 会话（pyACL，仅盒上可用；CANN 路径，x86 静态检查级）。

        OM 加载失败 → 同源 ONNX 回退（记 session_init_error 与 om_degraded_to_onnx），
        保持"模型不可用不阻塞启动"的项目约定。
        """
        try:
            from edge_migration.inference_backend import ort_compatible_om_session
        except Exception as exc:
            self.session_init_error = f"edge_backend_import_failed: {exc}"
            return None
        adapter, err = ort_compatible_om_session(
            om_path, device_id=int(self.runtime_cfg.get("npu_device_id", 0)))
        if adapter is not None:
            return adapter
        self.session_init_error = f"om_unavailable: {err}"
        fallback = self._om_fallback_onnx(om_path)
        if fallback:
            session = self._build_session(fallback)
            if session is not None:
                self.onnx_path_resolved = fallback  # debug 展示实际生效模型
                self.om_degraded_to_onnx = True
                return session
        return None

    def close(self) -> None:
        """释放推理会话；OM 后端需显式 unload 以释放 NPU 设备（供档切换重建）。"""
        session = getattr(self, "session", None)
        close = getattr(session, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        self.session = None
        self.is_real_model_loaded = False

    def _ensure_buffers(self) -> None:
        input_width, input_height = int(self.input_size[0]), int(self.input_size[1])
        if self._canvas_buffer is None or self._canvas_buffer.shape != (input_height, input_width, 3):
            self._canvas_buffer = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        if self._rgb_buffer is None or self._rgb_buffer.shape != (input_height, input_width, 3):
            self._rgb_buffer = np.empty((input_height, input_width, 3), dtype=np.uint8)
        if self._tensor_buffer is None or self._tensor_buffer.shape != (1, 3, input_height, input_width):
            self._tensor_buffer = np.empty((1, 3, input_height, input_width), dtype=np.float32)

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        input_width, input_height = int(self.input_size[0]), int(self.input_size[1])
        height, width = frame.shape[:2]
        scale = min(input_width / width, input_height / height)

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        if self.preprocess_optimized:
            self._ensure_buffers()
            canvas = self._canvas_buffer
            canvas.fill(114)
        else:
            canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        pad_x = (input_width - new_width) // 2
        pad_y = (input_height - new_height) // 2
        canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
        return canvas, scale, pad_x, pad_y

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        image, scale, pad_x, pad_y = self._letterbox(frame)
        if not self.preprocess_optimized:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = image.astype(np.float32) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))
            tensor = np.expand_dims(tensor, axis=0)
            return tensor, scale, pad_x, pad_y

        self._ensure_buffers()
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=self._rgb_buffer)
        np.multiply(self._rgb_buffer[:, :, 0], 1.0 / 255.0, out=self._tensor_buffer[0, 0], casting="unsafe")
        np.multiply(self._rgb_buffer[:, :, 1], 1.0 / 255.0, out=self._tensor_buffer[0, 1], casting="unsafe")
        np.multiply(self._rgb_buffer[:, :, 2], 1.0 / 255.0, out=self._tensor_buffer[0, 2], casting="unsafe")
        return self._tensor_buffer, scale, pad_x, pad_y

    def _nms(self, boxes_xyxy: np.ndarray, scores: np.ndarray) -> list[int]:
        if len(boxes_xyxy) == 0:
            return []

        order = np.argsort(scores)[::-1].astype(np.int32)
        keep: list[int] = []
        x1 = boxes_xyxy[:, 0]
        y1 = boxes_xyxy[:, 1]
        x2 = boxes_xyxy[:, 2]
        y2 = boxes_xyxy[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

        while len(order) > 0:
            index = int(order[0])
            keep.append(index)
            if len(order) == 1:
                break

            remaining = order[1:]
            xx1 = np.maximum(x1[index], x1[remaining])
            yy1 = np.maximum(y1[index], y1[remaining])
            xx2 = np.minimum(x2[index], x2[remaining])
            yy2 = np.minimum(y2[index], y2[remaining])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            union = areas[index] + areas[remaining] - inter
            ious = np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)
            order = remaining[ious < self.iou_threshold]

        return keep

    def _decode_predictions(self, output: np.ndarray, frame_shape: tuple[int, int], scale: float, pad_x: int, pad_y: int) -> list[dict[str, Any]]:
        detections_before_threshold = 0
        detections_after_threshold = 0
        detections_after_nms = 0

        if output.ndim == 3:
            output = output[0]

        if output.ndim != 2:
            self.last_debug.update(
                {
                    "raw_output_shape": list(output.shape),
                    "detections_before_threshold": detections_before_threshold,
                    "detections_after_threshold": detections_after_threshold,
                    "detections_after_nms": detections_after_nms,
                    "effective_person_boxes": 0,
                }
            )
            return []

        # Typical YOLO ONNX layouts:
        # - [N, 84]   (already candidate-major)
        # - [84, N]   (feature-major, needs transpose)
        # Heuristic: when first dim is feature-like (<=128) and second dim is large, transpose.
        if output.shape[0] <= 128 and output.shape[1] > 128:
            output = output.T

        if output.shape[1] < 5:
            self.last_debug.update(
                {
                    "raw_output_shape": list(output.shape),
                    "detections_before_threshold": detections_before_threshold,
                    "detections_after_threshold": detections_after_threshold,
                    "detections_after_nms": detections_after_nms,
                    "effective_person_boxes": 0,
                }
            )
            return []

        boxes_xywh = output[:, :4]
        class_scores = output[:, 4:]
        detections_before_threshold = int(boxes_xywh.shape[0])

        if class_scores.shape[1] == 0:
            confidences = np.ones((boxes_xywh.shape[0],), dtype=np.float32)
            class_ids = np.zeros((boxes_xywh.shape[0],), dtype=np.int32)
        else:
            confidences = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)

        mask = confidences >= self.confidence_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        detections_after_threshold = int(boxes_xywh.shape[0])

        # Keep only person-class candidates before NMS to cut postprocess cost.
        person_mask = class_ids == 0
        boxes_xywh = boxes_xywh[person_mask]
        confidences = confidences[person_mask]
        class_ids = class_ids[person_mask]

        if boxes_xywh.shape[0] == 0:
            self.last_debug.update(
                {
                    "raw_output_shape": list(output.shape),
                    "detections_before_threshold": detections_before_threshold,
                    "detections_after_threshold": detections_after_threshold,
                    "detections_after_nms": detections_after_nms,
                    "effective_person_boxes": 0,
                }
            )
            return []

        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        bw = boxes_xywh[:, 2]
        bh = boxes_xywh[:, 3]

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = self._nms(boxes_xyxy, confidences)
        detections_after_nms = int(len(keep))

        frame_height, frame_width = frame_shape
        detections: list[dict[str, Any]] = []
        for index in keep:
            left = (boxes_xyxy[index, 0] - pad_x) / scale
            top = (boxes_xyxy[index, 1] - pad_y) / scale
            right = (boxes_xyxy[index, 2] - pad_x) / scale
            bottom = (boxes_xyxy[index, 3] - pad_y) / scale

            left = int(np.clip(left, 0, frame_width - 1))
            top = int(np.clip(top, 0, frame_height - 1))
            right = int(np.clip(right, 0, frame_width - 1))
            bottom = int(np.clip(bottom, 0, frame_height - 1))

            width = max(1, right - left)
            height = max(1, bottom - top)

            detections.append(
                {
                    "bbox": [left, top, width, height],
                    "score": float(confidences[index]),
                    "class": self.target_class_name,
                }
            )

        self.last_debug.update(
            {
                "raw_output_shape": list(output.shape),
                "detections_before_threshold": detections_before_threshold,
                "detections_after_threshold": detections_after_threshold,
                "detections_after_nms": detections_after_nms,
                "effective_person_boxes": int(len(detections)),
            }
        )

        return detections

    def infer(self, frame: Any) -> list[dict[str, Any]]:
        if self.session is not None:
            try:
                tensor, scale, pad_x, pad_y = self._preprocess(frame)
                outputs = self.session.run(self.runtime_output_tensor_names, {self.runtime_input_tensor_name: tensor})
                self.last_debug.update(
                    {
                        "is_real_model_loaded": True,
                        "used_fallback": False,
                        "error": "",
                        "input_tensor_name": self.input_tensor_name,
                        "output_tensor_names": self.output_tensor_names,
                        "runtime_input_tensor_name": self.runtime_input_tensor_name,
                        "runtime_output_tensor_names": list(self.runtime_output_tensor_names),
                    }
                )
                detections = self._decode_predictions(outputs[0], frame.shape[:2], scale, pad_x, pad_y)
                if detections:
                    return detections
                self.last_debug.update(
                    {
                        "is_real_model_loaded": True,
                        "used_fallback": True,
                        "error": "no_person_after_decode",
                    }
                )
            except Exception:
                self.last_debug.update(
                    {
                        "is_real_model_loaded": self.is_real_model_loaded,
                        "used_fallback": True,
                        "error": "onnx_infer_failed",
                    }
                )

        self.last_debug.update(
            {
                "is_real_model_loaded": self.is_real_model_loaded,
                "used_fallback": True,
                "error": self.last_debug.get("error") or "fallback_static_detections",
                "raw_output_shape": self.last_debug.get("raw_output_shape"),
                "effective_person_boxes": 0,
            }
        )
        return []

    def get_debug_info(self) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "onnx_path": self.onnx_path,
            "onnx_path_resolved": self.onnx_path_resolved,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            **self.last_debug,
        }
