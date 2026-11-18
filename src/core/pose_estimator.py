from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


class PoseEstimator:
    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.runtime_cfg = dict(config.get("runtime", {}))
        self.onnx_path = str(config.get("active_onnx_path") or config.get("onnx_path") or "")
        self.onnx_path_resolved = self._resolve_model_path(self.onnx_path)
        self.input_tensor_name = str(config.get("input_tensor_name", "input"))
        self.output_tensor_names = list(config.get("output_tensor_names", ["keypoints"]))
        self.runtime_input_tensor_name = self.input_tensor_name
        self.runtime_output_tensor_names = list(self.output_tensor_names)
        self.keypoint_keep_indices = [int(idx) for idx in config.get("keypoint_keep_indices", [])]
        self.input_size = tuple(config.get("input_size", [256, 192]))
        self.runtime_input_height = int(self.input_size[0]) if len(self.input_size) > 0 else 256
        self.runtime_input_width = int(self.input_size[1]) if len(self.input_size) > 1 else 192
        self.preprocess_optimized = bool(config.get("preprocess_optimized", True))
        self._rgb_buffer: np.ndarray | None = None
        self._tensor_buffer: np.ndarray | None = None
        self.session_init_error = ""
        self.om_degraded_to_onnx = False  # OM 不可用回退 ONNX 时置 True（debug 可见）
        self._is_om = False  # 当前 session 是否为固定 batch=1 的 Ascend OM
        self.session = self._build_session(self.onnx_path_resolved)
        self.is_real_model_loaded = self.session is not None
        # ImageNet mean/std（ViTPose/mmpose 预训练标准归一化，0-255 口径）
        self._imagenet_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
        self._imagenet_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)
        self._crop_expand_ratio = 1.25

        if self.session is not None:
            try:
                input_meta = self.session.get_inputs()
                input_names = [item.name for item in input_meta]
                output_names = [item.name for item in self.session.get_outputs()]
                if input_names:
                    if self.input_tensor_name in input_names:
                        self.runtime_input_tensor_name = self.input_tensor_name
                    else:
                        self.runtime_input_tensor_name = input_names[0]
                if input_meta:
                    shape = input_meta[0].shape
                    if isinstance(shape, list) and len(shape) >= 4:
                        h = shape[-2]
                        w = shape[-1]
                        if isinstance(h, int) and h > 0:
                            self.runtime_input_height = h
                        if isinstance(w, int) and w > 0:
                            self.runtime_input_width = w
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
            "runtime_output_tensor_names": self.runtime_output_tensor_names,
            "runtime_input_hw": [self.runtime_input_height, self.runtime_input_width],
            "raw_output_shape": None,
            "num_keypoints": 0,
            "num_keypoints_raw": 0,
            "keypoint_keep_indices": self.keypoint_keep_indices,
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
                # ORT 线程池默认自旋等待；姿态是低频调用（每 stride 帧一次），
                # 自旋会在空档期白吃 CPU。关掉 intra/inter 自旋，让线程睡眠。
                try:
                    session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                    session_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
                except Exception:
                    pass
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
        """OM 不可用时的同源 ONNX 回退（与 detector 同约定）。"""
        p = Path(om_path)
        candidates = [p.with_suffix(".onnx")]
        if p.stem.endswith("_310B"):
            candidates.append(p.with_name(p.stem[:-5] + ".onnx"))
        for c in candidates:
            if c.exists():
                return str(c)
        return ""

    def _build_om_session(self, om_path: str):
        """昇腾 OM 会话（pyACL，仅盒上可用；CANN 路径，x86 静态检查级）。"""
        try:
            from edge_migration.inference_backend import ort_compatible_om_session
        except Exception as exc:
            self.session_init_error = f"edge_backend_import_failed: {exc}"
            return None
        adapter, err = ort_compatible_om_session(
            om_path, device_id=int(self.runtime_cfg.get("npu_device_id", 0)))
        if adapter is not None:
            self._is_om = True
            return adapter
        self.session_init_error = f"om_unavailable: {err}"
        fallback = self._om_fallback_onnx(om_path)
        if fallback:
            session = self._build_session(fallback)
            if session is not None:
                self.onnx_path_resolved = fallback
                self.om_degraded_to_onnx = True
                return session
        return None

    def _fallback_keypoints(self, track: dict) -> list[list[float]]:
        x, y, w, h = [float(value) for value in track["bbox"]]
        left = x
        top = y
        right = x + w
        bottom = y + h
        mid_x = x + w * 0.5
        head_y = y + h * 0.12
        shoulder_y = y + h * 0.28
        elbow_y = y + h * 0.45
        wrist_y = y + h * 0.62
        hip_y = y + h * 0.62
        knee_y = y + h * 0.80
        ankle_y = y + h * 0.96

        keypoints = [
            [mid_x, head_y, 0.95],
            [mid_x - w * 0.08, head_y - h * 0.03, 0.90],
            [mid_x + w * 0.08, head_y - h * 0.03, 0.90],
            [mid_x - w * 0.14, head_y + h * 0.02, 0.86],
            [mid_x + w * 0.14, head_y + h * 0.02, 0.86],
            [mid_x - w * 0.18, shoulder_y, 0.93],
            [mid_x + w * 0.18, shoulder_y, 0.93],
            [mid_x - w * 0.26, elbow_y, 0.85],
            [mid_x + w * 0.26, elbow_y, 0.85],
            [mid_x - w * 0.32, wrist_y, 0.82],
            [mid_x + w * 0.32, wrist_y, 0.82],
            [mid_x - w * 0.12, hip_y, 0.90],
            [mid_x + w * 0.12, hip_y, 0.90],
            [mid_x - w * 0.12, knee_y, 0.84],
            [mid_x + w * 0.12, knee_y, 0.84],
            [mid_x - w * 0.12, ankle_y, 0.80],
            [mid_x + w * 0.12, ankle_y, 0.80],
        ]

        for point in keypoints:
            point[0] = float(np.clip(point[0], left, right))
            point[1] = float(np.clip(point[1], top, bottom))
        # D3: 补齐到 133 点（COCO WholeBody 格式），防止与正常 133 点混入缓冲导致形状不匹配
        while len(keypoints) < 133:
            keypoints.append([mid_x, (top + bottom) / 2, 0.0])
        return keypoints[:133]

    def _ensure_preprocess_buffers(self) -> None:
        input_h = int(self.runtime_input_height)
        input_w = int(self.runtime_input_width)
        if self._rgb_buffer is None or self._rgb_buffer.shape != (input_h, input_w, 3):
            self._rgb_buffer = np.empty((input_h, input_w, 3), dtype=np.uint8)
        if self._tensor_buffer is None or self._tensor_buffer.shape != (1, 3, input_h, input_w):
            self._tensor_buffer = np.empty((1, 3, input_h, input_w), dtype=np.float32)

    def _preprocess(self, frame: np.ndarray, track: dict) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = [int(value) for value in track.get("bbox", [0, 0, 1, 1])]
        left = int(np.clip(x, 0, frame_w - 1))
        top = int(np.clip(y, 0, frame_h - 1))
        right = int(np.clip(x + w, 1, frame_w))
        bottom = int(np.clip(y + h, 1, frame_h))

        if right <= left:
            right = min(frame_w, left + 1)
        if bottom <= top:
            bottom = min(frame_h, top + 1)

        input_h = int(self.runtime_input_height)
        input_w = int(self.runtime_input_width)

        # ── D2: 1.25 外扩 + 保宽高比（标准 top-down 流程） ──
        exp_w = int(max(1, (right - left) * self._crop_expand_ratio))
        exp_h = int(max(1, (bottom - top) * self._crop_expand_ratio))
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        # 保持 3:4 宽高比（ViTPose 模板 192×256）
        if exp_w * 4 < exp_h * 3:
            exp_w = int(exp_h * 3 / 4)
        else:
            exp_h = int(exp_w * 4 / 3)
        exp_left = max(0, cx - exp_w // 2)
        exp_top = max(0, cy - exp_h // 2)
        exp_right = min(frame.shape[1], exp_left + exp_w)
        exp_bottom = min(frame.shape[0], exp_top + exp_h)
        exp_left = max(0, exp_right - exp_w)
        exp_top = max(0, exp_bottom - exp_h)

        padded = frame[exp_top:exp_bottom, exp_left:exp_right]
        resized = cv2.resize(padded, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

        if not self.preprocess_optimized:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # 先转 CHW 再减 (3,1,1) mean——HWC 上广播 (256,192,3)vs(3,1,1) 必然失败
            tensor = rgb.astype(np.float32).transpose(2, 0, 1)
            tensor = (tensor / 255.0 - self._imagenet_mean / 255.0) / (self._imagenet_std / 255.0)
            tensor = np.expand_dims(tensor, axis=0)
            return tensor, (left, top, right, bottom)

        self._ensure_preprocess_buffers()
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB, dst=self._rgb_buffer)
        for c in range(3):
            self._rgb_buffer[:, :, c] = (self._rgb_buffer[:, :, c] - self._imagenet_mean[c, 0, 0]) / self._imagenet_std[c, 0, 0]
        np.copyto(self._tensor_buffer[0, 0], self._rgb_buffer[:, :, 0])
        np.copyto(self._tensor_buffer[0, 1], self._rgb_buffer[:, :, 1])
        np.copyto(self._tensor_buffer[0, 2], self._rgb_buffer[:, :, 2])
        return self._tensor_buffer, (left, top, right, bottom)

    def _preprocess_batch(self, frame: np.ndarray, tracks: list[dict]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        def _preprocess_local(track: dict) -> tuple[np.ndarray, tuple[int, int, int, int]]:
            frame_h, frame_w = frame.shape[:2]
            x, y, w, h = [int(value) for value in track.get("bbox", [0, 0, 1, 1])]
            left = int(np.clip(x, 0, frame_w - 1))
            top = int(np.clip(y, 0, frame_h - 1))
            right = int(np.clip(x + w, 1, frame_w))
            bottom = int(np.clip(y + h, 1, frame_h))

            if right <= left:
                right = min(frame_w, left + 1)
            if bottom <= top:
                bottom = min(frame_h, top + 1)

            input_h = int(self.runtime_input_height)
            input_w = int(self.runtime_input_width)

            exp_w = int(max(1, (right - left) * self._crop_expand_ratio))
            exp_h = int(max(1, (bottom - top) * self._crop_expand_ratio))
            cx = (left + right) // 2
            cy = (top + bottom) // 2
            if exp_w * 4 < exp_h * 3:
                exp_w = int(exp_h * 3 / 4)
            else:
                exp_h = int(exp_w * 4 / 3)
            exp_left = max(0, cx - exp_w // 2)
            exp_top = max(0, cy - exp_h // 2)
            exp_right = min(frame.shape[1], exp_left + exp_w)
            exp_bottom = min(frame.shape[0], exp_top + exp_h)
            exp_left = max(0, exp_right - exp_w)
            exp_top = max(0, exp_bottom - exp_h)

            padded = frame[exp_top:exp_bottom, exp_left:exp_right]
            resized = cv2.resize(padded, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # 同上：CHW 后再归一化（infer_batch 此前漏改，引擎姿态 100% fallback 的根因）
            tensor = rgb.astype(np.float32).transpose(2, 0, 1)
            tensor = (tensor / 255.0 - self._imagenet_mean / 255.0) / (self._imagenet_std / 255.0)
            tensor = np.expand_dims(tensor, axis=0)
            return tensor, (left, top, right, bottom)

        if not tracks:
            return np.empty((0, 3, self.runtime_input_height, self.runtime_input_width), dtype=np.float32), []

        if len(tracks) == 1:
            tensor, roi = _preprocess_local(tracks[0])
            return np.asarray(tensor, dtype=np.float32), [roi]

        worker_count = min(len(tracks), 4)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(_preprocess_local, tracks))

        tensors = [np.asarray(item[0], dtype=np.float32) for item in results]
        rois = [item[1] for item in results]
        return np.concatenate(tensors, axis=0), rois

    def _decode_pose_output(self, output: np.ndarray, roi: tuple[int, int, int, int]) -> list[list[float]]:
        left, top, right, bottom = roi
        roi_w = max(1, right - left)
        roi_h = max(1, bottom - top)

        keypoints: list[list[float]] = []

        if output.ndim == 4:
            # heatmap: [1, K, H, W]
            heatmap = output[0]
            num_k, map_h, map_w = heatmap.shape
            for idx in range(num_k):
                channel = heatmap[idx]
                flat_idx = int(np.argmax(channel))
                y = flat_idx // map_w
                x = flat_idx % map_w
                conf = float(np.max(channel))
                px = left + (x / max(1, map_w)) * roi_w
                py = top + (y / max(1, map_h)) * roi_h
                keypoints.append([float(px), float(py), conf])
            return keypoints

        if output.ndim == 3:
            output = output[0]

        if output.ndim == 2 and output.shape[1] >= 2:
            # direct keypoints: [K,2] or [K,3]
            for row in output:
                x_norm = float(row[0])
                y_norm = float(row[1])
                conf = float(row[2]) if row.shape[0] > 2 else 0.9
                # if model outputs normalized [0,1], map to ROI; else clip as pixel coords
                if 0.0 <= x_norm <= 1.5 and 0.0 <= y_norm <= 1.5:
                    px = left + x_norm * roi_w
                    py = top + y_norm * roi_h
                else:
                    px = np.clip(x_norm + left, left, right)
                    py = np.clip(y_norm + top, top, bottom)
                keypoints.append([float(px), float(py), conf])
            return keypoints

        return keypoints

    def _filter_keypoints(self, keypoints: list[list[float]]) -> list[list[float]]:
        if not self.keypoint_keep_indices:
            return keypoints
        if not keypoints:
            return keypoints

        keep_set = {idx for idx in self.keypoint_keep_indices if idx >= 0}
        filtered: list[list[float]] = []
        for idx, keypoint in enumerate(keypoints):
            if idx in keep_set:
                filtered.append([float(value) for value in keypoint[:3]])
            else:
                filtered.append([0.0, 0.0, 0.0])
        return filtered

    def infer(self, frame_or_track: Any, track: dict | None = None) -> list[list[float]]:
        # backward compatibility: infer(track)
        if track is None:
            track = frame_or_track
            frame = None
        else:
            frame = frame_or_track

        if self.session is not None and frame is not None:
            try:
                tensor, roi = self._preprocess(frame, track)
                outputs = self.session.run(self.runtime_output_tensor_names, {self.runtime_input_tensor_name: tensor})
                pose_output = outputs[0]
                keypoints_raw = self._decode_pose_output(pose_output, roi)
                keypoints = self._filter_keypoints(keypoints_raw)
                self.last_debug.update(
                    {
                        "is_real_model_loaded": True,
                        "used_fallback": False,
                        "error": "" if keypoints else "no_keypoints_after_decode",
                        "input_tensor_name": self.input_tensor_name,
                        "output_tensor_names": self.output_tensor_names,
                        "runtime_input_tensor_name": self.runtime_input_tensor_name,
                        "runtime_output_tensor_names": self.runtime_output_tensor_names,
                        "runtime_input_hw": [self.runtime_input_height, self.runtime_input_width],
                        "raw_output_shape": list(pose_output.shape) if hasattr(pose_output, "shape") else None,
                        "num_keypoints": len(keypoints),
                        "num_keypoints_raw": len(keypoints_raw),
                        "keypoint_keep_indices": self.keypoint_keep_indices,
                    }
                )
                if keypoints:
                    return keypoints
            except Exception as exc:
                self.last_debug.update(
                    {
                        "is_real_model_loaded": self.is_real_model_loaded,
                        "used_fallback": True,
                        "error": f"onnx_infer_failed: {exc}",
                        "input_tensor_name": self.input_tensor_name,
                        "output_tensor_names": self.output_tensor_names,
                        "runtime_input_tensor_name": self.runtime_input_tensor_name,
                        "runtime_output_tensor_names": self.runtime_output_tensor_names,
                        "runtime_input_hw": [self.runtime_input_height, self.runtime_input_width],
                    }
                )

        fallback = self._fallback_keypoints(track)
        fallback = self._filter_keypoints(fallback)
        self.last_debug.update(
            {
                "is_real_model_loaded": self.is_real_model_loaded,
                "used_fallback": True,
                "error": self.last_debug.get("error") or "fallback_static_pose",
                "num_keypoints": len(fallback),
                "num_keypoints_raw": len(fallback),
                "keypoint_keep_indices": self.keypoint_keep_indices,
            }
        )
        return fallback

    def infer_batch(self, frame: np.ndarray, tracks: list[dict]) -> list[list[list[float]]]:
        if not tracks:
            return []

        # OM 输入是固定 batch=1；不能把 batch=2 塞进 batch=1 的 device buffer。
        # 逐 track 复用单帧 infer()，ONNX 路径仍走批量推理。
        if self._is_om:
            return [self.infer(frame, track) for track in tracks]

        if self.session is None:
            return [self._filter_keypoints(self._fallback_keypoints(track)) for track in tracks]

        try:
            batch_tensor, rois = self._preprocess_batch(frame, tracks)
            outputs = self.session.run(self.runtime_output_tensor_names, {self.runtime_input_tensor_name: batch_tensor})
            pose_output = outputs[0]
            keypoints_list: list[list[list[float]]] = []
            if hasattr(pose_output, "ndim") and pose_output.ndim >= 4 and pose_output.shape[0] >= len(tracks):
                raw_slices = [pose_output[index : index + 1] for index in range(len(tracks))]
            else:
                return [self.infer(frame, track) for track in tracks]

            for index, track in enumerate(tracks):
                keypoints_raw = self._decode_pose_output(np.array(raw_slices[index]), rois[index])
                keypoints = self._filter_keypoints(keypoints_raw)
                if not keypoints:
                    keypoints = self._filter_keypoints(self._fallback_keypoints(track))
                keypoints_list.append(keypoints)

            self.last_debug.update(
                {
                    "is_real_model_loaded": True,
                    "used_fallback": False,
                    "error": "",
                    "input_tensor_name": self.input_tensor_name,
                    "output_tensor_names": self.output_tensor_names,
                    "runtime_input_tensor_name": self.runtime_input_tensor_name,
                    "runtime_output_tensor_names": self.runtime_output_tensor_names,
                    "runtime_input_hw": [self.runtime_input_height, self.runtime_input_width],
                    "raw_output_shape": list(pose_output.shape) if hasattr(pose_output, "shape") else None,
                    "num_keypoints": sum(len(item) for item in keypoints_list),
                    "num_keypoints_raw": sum(len(item) for item in keypoints_list),
                    "keypoint_keep_indices": self.keypoint_keep_indices,
                }
            )
            return keypoints_list
        except Exception as exc:
            self.last_debug.update(
                {
                    "is_real_model_loaded": self.is_real_model_loaded,
                    "used_fallback": True,
                    "error": f"onnx_infer_failed: {exc}",
                    "input_tensor_name": self.input_tensor_name,
                    "output_tensor_names": self.output_tensor_names,
                    "runtime_input_tensor_name": self.runtime_input_tensor_name,
                    "runtime_output_tensor_names": self.runtime_output_tensor_names,
                    "runtime_input_hw": [self.runtime_input_height, self.runtime_input_width],
                }
            )
            return [self._filter_keypoints(self._fallback_keypoints(track)) for track in tracks]

    def get_debug_info(self) -> dict[str, Any]:
        return {
            "onnx_path": self.onnx_path,
            "onnx_path_resolved": self.onnx_path_resolved,
            "input_size": list(self.input_size),
            "runtime_input_hw": [self.runtime_input_height, self.runtime_input_width],
            "keypoint_keep_indices": self.keypoint_keep_indices,
            **self.last_debug,
        }
