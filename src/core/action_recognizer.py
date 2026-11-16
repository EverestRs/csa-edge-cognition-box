from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


class ActionRecognizer:
    CANDIDATES = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"]
    PROJECT_LABELS = {"Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"}
    DIRECT_CLASSES = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"]
    # 原型扩展类（N01...）经 config action.direct_classes 注入——硬编码 11 类会把新类索引吞成 Normal
    DEFAULT_KEYWORD_RULES = {
        "H01": ["fighting", "wrestling", "boxing", "punch", "kick", "martial", "karate", "taekwondo"],
        "H02": ["beating", "slapping", "assault", "attacking", "hitting person"],
        "H03": ["fall", "slip", "stumble", "tripping", "collapse", "faint"],
        "H04": ["jumping", "diving", "cliff", "ledge", "hanging", "suicide", "danger", "abseiling", "rope descent", "descending rope", "parkour"],
        "H05": ["breaking", "smashing", "destroy", "throwing", "kicking object", "vandal"],
        "M01": ["smoking", "cigarette", "lighting", "lighter", "fire"],
        "M02": ["phone", "calling", "texting", "smartphone", "cellphone"],
        "M03": ["covering camera", "blocking lens", "tampering camera", "spray camera"],
        "S01": ["wandering", "pacing", "loiter", "roaming", "patrolling"],
        "S02": ["climb fence", "crossing barrier", "intrusion", "restricted area"],
    }

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.runtime_cfg = dict(config.get("runtime", {}))
        self.onnx_path = str(config.get("active_onnx_path") or config.get("onnx_path") or "")
        self.onnx_path_resolved = self._resolve_model_path(self.onnx_path)
        self.input_tensor_name = str(config.get("input_tensor_name", "video"))
        self.output_tensor_names = list(config.get("output_tensor_names", ["logits"]))
        self.class_mapping = dict(config.get("class_mapping", {}))
        # ROADMAP Stage 3：扩展类标签注入（N01...），主路径才能输出新类名
        extra = [str(c) for c in (config.get("direct_classes") or []) if str(c)]
        if extra:
            self.DIRECT_CLASSES = list(self.DIRECT_CLASSES) + [c for c in extra if c not in self.DIRECT_CLASSES]
        self.source_labels_path = str(config.get("source_labels_path", ""))
        self.enable_keyword_mapping = bool(config.get("enable_keyword_mapping", True))
        self.keyword_mapping_threshold = float(config.get("keyword_mapping_threshold", 0.18))
        self.keyword_rules = self._build_keyword_rules(config.get("keyword_mapping", {}))
        self.source_labels = self._load_source_labels(self.source_labels_path)
        self.input_clip_len = int(config.get("input_clip_len", 32))
        input_size = list(config.get("input_size", [224, 224]))
        self.input_height = int(input_size[0]) if len(input_size) > 0 else 224
        self.input_width = int(input_size[1]) if len(input_size) > 1 else 224
        self.slowfast_alpha = int(max(2, config.get("slowfast_alpha", 4)))
        mean = list(config.get("normalize_mean", [0.45, 0.45, 0.45]))
        std = list(config.get("normalize_std", [0.225, 0.225, 0.225]))
        self.normalize_mean = np.array(mean[:3], dtype=np.float32).reshape(3, 1, 1)
        self.normalize_std = np.maximum(np.array(std[:3], dtype=np.float32).reshape(3, 1, 1), 1e-6)
        self.runtime_input_tensor_name = self.input_tensor_name
        self.runtime_output_tensor_names = list(self.output_tensor_names)
        self.runtime_input_names: list[str] = []
        self.runtime_input_shapes: dict[str, list[int | None]] = {}
        self.runtime_output_names: list[str] = []
        self.session_init_error = ""
        self.session = self._build_session(self.onnx_path_resolved)
        self.is_real_model_loaded = self.session is not None
        self.is_pose_model = False
        if self.session is not None:
            try:
                input_names = [item.name for item in self.session.get_inputs()]
                if len(input_names) == 1 and input_names[0] == "pose_input":
                    self.is_pose_model = True
            except Exception:
                pass

        # ── PoseClassifier ONNX 路径（主路径） ──────────────────
        self.pose_classifier_onnx_path = str(config.get("pose_classifier_onnx_path", ""))
        self.pose_classifier_onnx_resolved = self._resolve_model_path(self.pose_classifier_onnx_path)
        self.pose_classifier_session = None
        self.pose_classifier_loaded = False
        self.pose_classifier_error = ""
        if self.pose_classifier_onnx_resolved:
            self.pose_classifier_session = self._build_session(self.pose_classifier_onnx_resolved)
            self.pose_classifier_loaded = self.pose_classifier_session is not None
            if not self.pose_classifier_loaded:
                self.pose_classifier_error = self.session_init_error or "pose_classifier_load_failed"

        # 三级决策阈值
        self.pc_conf_high = float(config.get("pose_classifier_conf_high", 0.75))
        self.pc_conf_low = float(config.get("pose_classifier_conf_low", 0.40))

        self.track_pose_buffers: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.input_clip_len))
        self.track_frame_buffers: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.input_clip_len))
        self.last_debug: dict[str, Any] = {
            "is_real_model_loaded": self.is_real_model_loaded,
            "pose_classifier_loaded": self.pose_classifier_loaded,
            "used_fallback": not self.is_real_model_loaded,
            "error": "" if self.is_real_model_loaded else (self.session_init_error or "onnx_session_not_ready"),
            "input_tensor_name": self.input_tensor_name,
            "output_tensor_names": self.output_tensor_names,
            "runtime_input_tensor_name": self.runtime_input_tensor_name,
            "runtime_output_tensor_names": self.runtime_output_tensor_names,
            "runtime_input_names": self.runtime_input_names,
            "runtime_input_shapes": self.runtime_input_shapes,
            "raw_output_shape": None,
            "input_mode": "pose",
            "mapping_mode": "direct",
            "external_top1_label": "",
            "source_label_count": len(self.source_labels),
            "buffer_len": 0,
            "top1_label": "Normal",
            "top1_score": 0.0,
        }

        if self.session is not None:
            try:
                input_meta = self.session.get_inputs()
                input_names = [item.name for item in input_meta]
                output_meta = self.session.get_outputs()
                output_names = [item.name for item in output_meta]
                if input_names:
                    self.runtime_input_tensor_name = self.input_tensor_name if self.input_tensor_name in input_names else input_names[0]
                    self.runtime_input_names = input_names
                for item in input_meta:
                    shape = item.shape if hasattr(item, "shape") else []
                    dims = [dim if isinstance(dim, int) and dim > 0 else None for dim in shape]
                    self.runtime_input_shapes[item.name] = dims
                    inferred_hw = self._infer_hw_from_shape(dims)
                    if inferred_hw is not None:
                        self.input_height, self.input_width = inferred_hw
                if output_names:
                    valid = [name for name in self.output_tensor_names if name in output_names]
                    self.runtime_output_tensor_names = valid if valid else output_names
                    self.runtime_output_names = output_names
            except Exception:
                pass

    # ── PoseClassifier 推理方法 ──────────────────────────────────

    def infer_pose_classifier(self, track_id: str) -> tuple[str, float]:
        """PoseClassifier 主路径推理：基于姿态序列直接输出 11 类。

        Returns:
            (label: str, confidence: float)
        """
        if not self.pose_classifier_loaded or self.pose_classifier_session is None:
            return "Normal", 0.0

        frames = list(self.track_pose_buffers[track_id])
        if len(frames) < self.input_clip_len:
            return "Normal", 0.0

        try:
            clip = np.stack(frames[-self.input_clip_len:], axis=0).astype(np.float32)  # (T, V, 3)
            tensor = np.expand_dims(clip, axis=0)  # (1, T, V, 3)

            input_name = self.pose_classifier_session.get_inputs()[0].name
            output_names = [o.name for o in self.pose_classifier_session.get_outputs()]
            outputs = self.pose_classifier_session.run(output_names, {input_name: tensor})
            logits = np.array(outputs[0], dtype=np.float32).squeeze()  # (11,)

            probs = self._softmax(logits)
            top_index = int(np.argmax(probs))
            confidence = float(probs[top_index])
            label = self.DIRECT_CLASSES[top_index] if top_index < len(self.DIRECT_CLASSES) else "Normal"

            self.last_debug.update({
                "used_fallback": False,
                "error": "",
                "input_mode": "pose_classifier",
                "mapping_mode": "direct_11class",
                "top1_label": label,
                "top1_score": round(confidence, 4),
            })
            return label, round(confidence, 2)
        except Exception as exc:
            self.last_debug.update({
                "used_fallback": True,
                "error": f"pose_classifier_failed: {exc}",
            })
            return "Normal", 0.0

    def infer_pose_classifier_batch(self, track_ids: list[str]) -> list[tuple[str, float]]:
        """PoseClassifier 批次推理。"""
        if not self.pose_classifier_loaded or self.pose_classifier_session is None:
            return [("Normal", 0.0) for _ in track_ids]

        tensors: list[np.ndarray] = []
        valid_indices: list[int] = []
        results: list[tuple[str, float]] = [("Normal", 0.0) for _ in track_ids]

        for idx, track_id in enumerate(track_ids):
            frames = list(self.track_pose_buffers[track_id])
            if len(frames) >= self.input_clip_len:
                clip = np.stack(frames[-self.input_clip_len:], axis=0).astype(np.float32)
                tensors.append(np.expand_dims(clip, axis=0))
                valid_indices.append(idx)

        if not tensors:
            return results

        try:
            batch_tensor = np.concatenate(tensors, axis=0).astype(np.float32)
            input_name = self.pose_classifier_session.get_inputs()[0].name
            output_names = [o.name for o in self.pose_classifier_session.get_outputs()]
            outputs = self.pose_classifier_session.run(output_names, {input_name: batch_tensor})
            logits = np.array(outputs[0], dtype=np.float32)  # (B, 11)

            for i, idx in enumerate(valid_indices):
                if logits.ndim == 2 and i < logits.shape[0]:
                    probs = self._softmax(logits[i])
                    top_index = int(np.argmax(probs))
                    confidence = float(probs[top_index])
                    label = self.DIRECT_CLASSES[top_index] if top_index < len(self.DIRECT_CLASSES) else "Normal"
                    results[idx] = (label, round(confidence, 2))
        except Exception:
            pass

        return results

    def get_pose_classifier_confidence(self, track_id: str) -> float:
        """获取 PoseClassifier 对当前 track 的置信度（用于三级决策）。"""
        _, confidence = self.infer_pose_classifier(track_id)
        return confidence

    # ── 原有方法保持不变 ──────────────────────────────────────────

    @classmethod
    def _build_keyword_rules(cls, user_rules: Any) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {key: list(values) for key, values in cls.DEFAULT_KEYWORD_RULES.items()}
        if isinstance(user_rules, dict):
            for key, values in user_rules.items():
                if key not in cls.PROJECT_LABELS:
                    continue
                if isinstance(values, list):
                    merged[key] = [str(item).strip().lower() for item in values if str(item).strip()]
        return merged

    @staticmethod
    def _read_labels_file(labels_path: str) -> list[str]:
        if not labels_path:
            return []
        path = Path(labels_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            path = (project_root / path).resolve()
        if not path.exists():
            return []
        labels: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    labels.append(text)
        return labels

    @staticmethod
    def _load_builtin_kinetics_labels() -> list[str]:
        try:
            from torchvision.models.video import R3D_18_Weights

            categories = R3D_18_Weights.KINETICS400_V1.meta.get("categories", [])
            if isinstance(categories, list) and len(categories) == 400:
                return [str(item) for item in categories]
        except Exception:
            return []
        return []

    def _load_source_labels(self, source_labels_path: str) -> list[str]:
        labels = self._read_labels_file(source_labels_path)
        if labels:
            return labels
        return self._load_builtin_kinetics_labels()

    def _resolve_external_label(self, top_index: int, num_classes: int) -> str:
        if len(self.source_labels) == num_classes and top_index < len(self.source_labels):
            return self.source_labels[top_index]

        mapping_keys = list(self.class_mapping.keys())
        if len(mapping_keys) == num_classes and top_index < len(mapping_keys):
            return mapping_keys[top_index]

        if top_index < len(self.CANDIDATES):
            return self.CANDIDATES[top_index]
        return f"class_{top_index}"

    def _keyword_map_label(self, probs: np.ndarray) -> tuple[str | None, float, str]:
        if not self.enable_keyword_mapping:
            return None, 0.0, ""
        if len(self.source_labels) != int(probs.shape[0]):
            return None, 0.0, ""

        best_label = ""
        best_score = 0.0
        best_source = ""

        for index, source_label in enumerate(self.source_labels):
            score = float(probs[index])
            source_lower = source_label.lower()
            for target_label, keywords in self.keyword_rules.items():
                if not keywords:
                    continue
                if any(keyword in source_lower for keyword in keywords):
                    if score > best_score:
                        best_score = score
                        best_label = target_label
                        best_source = source_label

        if best_label and best_score >= self.keyword_mapping_threshold:
            return best_label, best_score, best_source
        if best_label == "H04" and best_source and "abseiling" in best_source.lower() and best_score >= 0.02:
            return best_label, max(best_score, self.keyword_mapping_threshold), best_source
        return None, 0.0, ""

    @staticmethod
    def _infer_hw_from_shape(shape: list[int | None]) -> tuple[int, int] | None:
        if len(shape) != 5:
            return None
        if shape[1] == 3 or shape[1] is None:
            h, w = shape[3], shape[4]
        elif shape[2] == 3 or shape[2] is None:
            h, w = shape[3], shape[4]
        else:
            h, w = shape[2], shape[3]
        if isinstance(h, int) and h > 0 and isinstance(w, int) and w > 0:
            return h, w
        return None

    @staticmethod
    def _is_video_shape(shape: list[int | None]) -> bool:
        if len(shape) != 5:
            return False
        return shape[1] in (3, None) or shape[2] in (3, None) or shape[4] in (3, None)

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

    def _fallback(self, track_id: str, frame_id: int, reason: str) -> tuple[str, float]:
        label = "Normal"
        confidence = 0.0
        self.last_debug.update(
            {
                "used_fallback": True,
                "error": reason,
                "top1_label": label,
                "top1_score": confidence,
            }
        )
        return label, confidence

    def _infer_pose_direct(self, track_id: str, frame_id: int) -> tuple[str, float]:
        """姿态输入模型的直接推理（11 类输出，无需关键词映射）。"""
        frames = list(self.track_pose_buffers[track_id])
        if len(frames) < self.input_clip_len:
            return self._fallback(track_id, frame_id, "insufficient_clip_buffer")

        # 构建姿态张量: (1, T, V, 3)
        clip = np.stack(frames[-self.input_clip_len :], axis=0).astype(np.float32)
        tensor = np.expand_dims(clip, axis=0)  # (1, 32, 17, 3)

        try:
            input_feed = {self.runtime_input_tensor_name: tensor}
            outputs = self.session.run(self.runtime_output_tensor_names, input_feed)
            logits = np.array(outputs[0], dtype=np.float32).squeeze()  # (11,)

            probs = self._softmax(logits)
            top_index = int(np.argmax(probs))
            confidence = float(probs[top_index])
            mapped_label = self.DIRECT_CLASSES[top_index] if top_index < len(self.DIRECT_CLASSES) else "Normal"

            self.last_debug.update(
                {
                    "used_fallback": False,
                    "error": "",
                    "input_mode": "pose_direct",
                    "mapping_mode": "direct_11class",
                    "external_top1_label": mapped_label,
                    "source_label_count": 11,
                    "top1_label": mapped_label,
                    "top1_score": round(confidence, 4),
                }
            )
            return mapped_label, round(confidence, 2)
        except Exception as exc:
            return self._fallback(track_id, frame_id, f"pose_infer_failed: {exc}")

    def _infer_pose_direct_batch(self, track_ids: list[str], frame_id: int) -> list[tuple[str, float]]:
        """姿态输入模型的批次推理。"""
        tensors: list[np.ndarray] = []
        valid_indices: list[int] = []
        results: list[tuple[str, float]] = []

        for idx, track_id in enumerate(track_ids):
            frames = list(self.track_pose_buffers[track_id])
            if len(frames) >= self.input_clip_len:
                clip = np.stack(frames[-self.input_clip_len :], axis=0).astype(np.float32)
                tensors.append(np.expand_dims(clip, axis=0))
                valid_indices.append(idx)
            else:
                results.append(("Normal", 0.0))

        if not tensors:
            return results

        batch_tensor = np.concatenate(tensors, axis=0).astype(np.float32)  # (B, 32, 17, 3)

        try:
            input_feed = {self.runtime_input_tensor_name: batch_tensor}
            outputs = self.session.run(self.runtime_output_tensor_names, input_feed)
            logits = np.array(outputs[0], dtype=np.float32)  # (B, 11)

            for i, idx in enumerate(valid_indices):
                if logits.ndim == 2 and i < logits.shape[0]:
                    probs = self._softmax(logits[i])
                    top_index = int(np.argmax(probs))
                    confidence = float(probs[top_index])
                    label = self.DIRECT_CLASSES[top_index] if top_index < len(self.DIRECT_CLASSES) else "Normal"
                    results.append((label, round(confidence, 2)))
                else:
                    results.append(("Normal", 0.0))
        except Exception:
            for _ in valid_indices:
                results.append(("Normal", 0.0))

        return results

    def _append_pose(self, track_id: str, pose_keypoints: list[list[float]]) -> int:
        # 空姿态/异常形状直接跳过：写 17 点零数组会与 133 点混入缓冲，
        # 导致 np.stack 抛异常被静默吞掉、PoseClassifier 永远回退 Normal
        if not pose_keypoints:
            return len(self.track_pose_buffers[track_id])
        pose_array = np.array(pose_keypoints, dtype=np.float32)
        if pose_array.ndim != 2 or pose_array.shape[1] < 3:
            return len(self.track_pose_buffers[track_id])
        if pose_array.shape[1] > 3:
            pose_array = pose_array[:, :3]
        # D4: conf 通道归一化到 [0,1]（ViTPose heatmap 原始峰值可超 1）
        pose_array[:, 2] = np.clip(pose_array[:, 2], 0.0, 1.0)
        self.track_pose_buffers[track_id].append(pose_array)
        return len(self.track_pose_buffers[track_id])

    def append_pose(self, track_id: str, pose_keypoints: list[list[float]],
                    frame_width: float = 1920.0, frame_height: float = 1080.0) -> int:
        """主路径公开入口：每处理帧调用。坐标归一化与训练侧一致（x/W, y/H, conf 0-1）。"""
        if not pose_keypoints:
            return len(self.track_pose_buffers.get(track_id, []))
        arr = np.array(pose_keypoints, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            arr[:, 0] /= max(1.0, frame_width)
            arr[:, 1] /= max(1.0, frame_height)
            arr[:, 2] = np.clip(arr[:, 2], 0.0, 1.0)
        return self._append_pose(track_id, arr.tolist())

    def _append_frame_clip(self, track_id: str, frame: np.ndarray | None, track: dict[str, Any]) -> int:
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return len(self.track_frame_buffers[track_id])

        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = [int(value) for value in track.get("bbox", [0, 0, frame_w, frame_h])]
        left = int(np.clip(x, 0, frame_w - 1))
        top = int(np.clip(y, 0, frame_h - 1))
        right = int(np.clip(x + w, left + 1, frame_w))
        bottom = int(np.clip(y + h, top + 1, frame_h))

        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return len(self.track_frame_buffers[track_id])

        resized = cv2.resize(crop, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        chw = (chw - self.normalize_mean) / self.normalize_std
        self.track_frame_buffers[track_id].append(chw.astype(np.float32))
        return len(self.track_frame_buffers[track_id])

    def _build_input_tensor(self, track_id: str) -> np.ndarray | None:
        frames = list(self.track_pose_buffers[track_id])
        if len(frames) < self.input_clip_len:
            return None

        input_meta = self.session.get_inputs()[0]
        shape = input_meta.shape if hasattr(input_meta, "shape") else []
        dims = [dim if isinstance(dim, int) and dim > 0 else None for dim in shape]

        # candidate tensor from pose buffer: [T, V, C]
        clip = np.stack(frames[-self.input_clip_len :], axis=0).astype(np.float32)
        t_len = clip.shape[0]
        v_len = clip.shape[1]

        if len(dims) == 4:
            # [N, T, V, C]
            t_target = dims[1] or t_len
            v_target = dims[2] or v_len
            c_target = dims[3] or 3
            tensor = np.zeros((1, t_target, v_target, c_target), dtype=np.float32)
            clip_t = clip[:t_target, :v_target, :c_target]
            tensor[0, : clip_t.shape[0], : clip_t.shape[1], : clip_t.shape[2]] = clip_t
            return tensor

        if len(dims) == 5:
            # common ST-GCN shape: [N, C, T, V, M]
            c_target = dims[1] or 3
            t_target = dims[2] or t_len
            v_target = dims[3] or v_len
            m_target = dims[4] or 1
            tensor = np.zeros((1, c_target, t_target, v_target, m_target), dtype=np.float32)
            clip_t = clip[:t_target, :v_target, :c_target]
            clip_t = np.transpose(clip_t, (2, 0, 1))  # C,T,V
            tensor[0, : clip_t.shape[0], : clip_t.shape[1], : clip_t.shape[2], 0] = clip_t
            return tensor

        # fallback unknown shape: [1,T,V,C]
        tensor = np.expand_dims(clip, axis=0).astype(np.float32)
        return tensor

    @staticmethod
    def _fit_temporal(clip_tchw: np.ndarray, target_t: int | None) -> np.ndarray:
        if target_t is None or target_t <= 0:
            return clip_tchw
        current_t = clip_tchw.shape[0]
        if current_t == target_t:
            return clip_tchw
        if current_t > target_t:
            indices = np.linspace(0, current_t - 1, num=target_t, dtype=np.int32)
            return clip_tchw[indices]
        pad = np.repeat(clip_tchw[-1:], target_t - current_t, axis=0)
        return np.concatenate([clip_tchw, pad], axis=0)

    def _adapt_video_to_shape(self, clip_tchw: np.ndarray, shape: list[int | None]) -> np.ndarray:
        if len(shape) != 5:
            return np.expand_dims(np.transpose(clip_tchw, (1, 0, 2, 3)), axis=0).astype(np.float32)

        if shape[1] in (3, None):
            fitted = self._fit_temporal(clip_tchw, shape[2])
            tensor = np.transpose(fitted, (1, 0, 2, 3))
            return np.expand_dims(tensor, axis=0).astype(np.float32)

        if shape[2] in (3, None):
            fitted = self._fit_temporal(clip_tchw, shape[1])
            tensor = np.transpose(fitted, (0, 1, 2, 3))
            return np.expand_dims(tensor, axis=0).astype(np.float32)

        if shape[4] in (3, None):
            fitted = self._fit_temporal(clip_tchw, shape[1])
            tensor = np.transpose(fitted, (0, 2, 3, 1))
            return np.expand_dims(tensor, axis=0).astype(np.float32)

        return np.expand_dims(np.transpose(clip_tchw, (1, 0, 2, 3)), axis=0).astype(np.float32)

    def _build_video_input_feed(self, track_id: str) -> tuple[dict[str, np.ndarray], str] | tuple[None, str]:
        if not self.runtime_input_names:
            return None, "video_input_not_ready"

        video_inputs = [name for name in self.runtime_input_names if self._is_video_shape(self.runtime_input_shapes.get(name, []))]
        if not video_inputs:
            return None, "model_not_video_input"

        frames = list(self.track_frame_buffers[track_id])
        if len(frames) < self.input_clip_len:
            return None, "insufficient_video_clip_buffer"

        fast_clip = np.stack(frames[-self.input_clip_len :], axis=0).astype(np.float32)
        slow_clip = fast_clip[:: self.slowfast_alpha] if fast_clip.shape[0] > 1 else fast_clip

        input_feed: dict[str, np.ndarray] = {}
        for name in self.runtime_input_names:
            shape = self.runtime_input_shapes.get(name, [])
            if not self._is_video_shape(shape):
                continue

            use_slow = "slow" in name.lower()
            source = slow_clip if use_slow else fast_clip
            input_feed[name] = self._adapt_video_to_shape(source, shape)

        if not input_feed:
            return None, "video_feed_build_failed"

        return input_feed, ""

    def _ensure_track_buffers(self, track_id: str, pose_keypoints: list[list[float]] | None, frame: np.ndarray | None, track: dict[str, Any]) -> int:
        pose_buffer_len = self._append_pose(track_id, pose_keypoints or [])
        frame_buffer_len = self._append_frame_clip(track_id, frame, track)
        return max(pose_buffer_len, frame_buffer_len)

    def _build_input_feed_batch(self, track_ids: list[str]) -> tuple[dict[str, np.ndarray], str] | tuple[None, str]:
        if not self.runtime_input_names:
            return None, "video_input_not_ready"

        sample_feeds: list[dict[str, np.ndarray]] = []
        for track_id in track_ids:
            feed, reason = self._build_video_input_feed(track_id)
            if feed is None:
                tensor = self._build_input_tensor(track_id)
                if tensor is None:
                    return None, reason if reason else "insufficient_clip_buffer"
                feed = {self.runtime_input_tensor_name: tensor}
            sample_feeds.append(feed)

        if not sample_feeds:
            return None, "video_feed_build_failed"

        batched_feed: dict[str, np.ndarray] = {}
        for name in self.runtime_input_names:
            tensors = [np.asarray(feed[name], dtype=np.float32) for feed in sample_feeds if name in feed]
            if len(tensors) != len(sample_feeds):
                return None, "video_feed_build_failed"
            batched_feed[name] = np.concatenate(tensors, axis=0)
        return batched_feed, ""

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits)
        exps = np.exp(logits)
        denom = np.sum(exps)
        return exps / denom if denom > 0 else exps

    def infer(
        self,
        track: dict,
        frame_id: int,
        pose_keypoints: list[list[float]] | None = None,
        frame: np.ndarray | None = None,
    ) -> tuple[str, float]:
        track_id = track.get("track_id", "P000")
        pose_buffer_len = self._append_pose(track_id, pose_keypoints or [])
        frame_buffer_len = self._append_frame_clip(track_id, frame, track)
        buffer_len = max(pose_buffer_len, frame_buffer_len)

        self.last_debug.update(
            {
                "is_real_model_loaded": self.is_real_model_loaded,
                "input_tensor_name": self.input_tensor_name,
                "output_tensor_names": self.output_tensor_names,
                "runtime_input_tensor_name": self.runtime_input_tensor_name,
                "runtime_output_tensor_names": self.runtime_output_tensor_names,
                "runtime_input_names": self.runtime_input_names,
                "runtime_input_shapes": self.runtime_input_shapes,
                "buffer_len": buffer_len,
            }
        )

        if self.session is None:
            return self._fallback(track_id, frame_id, self.session_init_error or "onnx_session_not_ready")

        # 姿态输入模型：直接使用 pose buffer
        if self.is_pose_model:
            return self._infer_pose_direct(track_id, frame_id)

        input_feed, video_reason = self._build_video_input_feed(track_id)
        input_mode = "video"
        if input_feed is None:
            tensor = self._build_input_tensor(track_id)
            if tensor is None:
                reason = video_reason if video_reason else "insufficient_clip_buffer"
                return self._fallback(track_id, frame_id, reason)
            input_feed = {self.runtime_input_tensor_name: tensor}
            input_mode = "pose"

        try:
            outputs = self.session.run(self.runtime_output_tensor_names, input_feed)
            logits = np.array(outputs[0], dtype=np.float32)
            self.last_debug["raw_output_shape"] = list(logits.shape)

            logits_1d = np.squeeze(logits)
            if logits_1d.ndim != 1:
                logits_1d = logits_1d.reshape(-1)

            probs = self._softmax(logits_1d)
            top_index = int(np.argmax(probs))
            confidence = float(probs[top_index])

            external_label = self._resolve_external_label(top_index, int(probs.shape[0]))
            mapped_label = self.class_mapping.get(external_label, external_label)
            mapping_mode = "direct"

            if mapped_label not in self.PROJECT_LABELS:
                keyword_label, keyword_score, keyword_source = self._keyword_map_label(probs)
                if keyword_label is not None:
                    mapped_label = keyword_label
                    confidence = max(confidence, keyword_score)
                    mapping_mode = "keyword"
                    external_label = keyword_source or external_label
                else:
                    mapped_label = "Normal"
                    mapping_mode = "fallback_normal"

            self.last_debug.update(
                {
                    "used_fallback": False,
                    "error": "",
                    "input_mode": input_mode,
                    "mapping_mode": mapping_mode,
                    "external_top1_label": external_label,
                    "source_label_count": len(self.source_labels),
                    "top1_label": mapped_label,
                    "top1_score": round(confidence, 4),
                }
            )
            return mapped_label, round(confidence, 2)
        except Exception as exc:
            return self._fallback(track_id, frame_id, f"onnx_infer_failed: {exc}")

    def infer_batch(
        self,
        tracks: list[dict],
        frame_id: int,
        pose_keypoints_list: list[list[list[float]]] | None = None,
        frame: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        if not tracks:
            return []

        if pose_keypoints_list is None:
            pose_keypoints_list = [[] for _ in tracks]

        track_ids = [track.get("track_id", "P000") for track in tracks]

        def _prepare_buffers(args: tuple[dict, str, list[list[float]]]) -> tuple[int, str]:
            track, track_id, keypoints = args
            return self._ensure_track_buffers(track_id, keypoints, frame, track), track_id

        buffer_lens: list[int] = []
        if len(tracks) == 1:
            buffer_lens.append(self._ensure_track_buffers(track_ids[0], pose_keypoints_list[0], frame, tracks[0]))
        else:
            worker_count = min(len(tracks), 4)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for buffer_len, _track_id in executor.map(_prepare_buffers, zip(tracks, track_ids, pose_keypoints_list)):
                    buffer_lens.append(buffer_len)

        self.last_debug.update(
            {
                "is_real_model_loaded": self.is_real_model_loaded,
                "input_tensor_name": self.input_tensor_name,
                "output_tensor_names": self.output_tensor_names,
                "runtime_input_tensor_name": self.runtime_input_tensor_name,
                "runtime_output_tensor_names": self.runtime_output_tensor_names,
                "runtime_input_names": self.runtime_input_names,
                "runtime_input_shapes": self.runtime_input_shapes,
                "buffer_len": max(buffer_lens) if buffer_lens else 0,
            }
        )

        if self.session is None:
            return [self._fallback(track_id, frame_id, self.session_init_error or "onnx_session_not_ready") for track_id in track_ids]

        # 姿态输入模型的批次推理
        if self.is_pose_model:
            return self._infer_pose_direct_batch(track_ids, frame_id)

        input_feed, video_reason = self._build_input_feed_batch(track_ids)
        input_mode = "video"
        if input_feed is None:
            results: list[tuple[str, float]] = []
            for track, track_id in zip(tracks, track_ids):
                tensor = self._build_input_tensor(track_id)
                if tensor is None:
                    results.append(self._fallback(track_id, frame_id, video_reason if video_reason else "insufficient_clip_buffer"))
                    continue
                single_feed = {self.runtime_input_tensor_name: tensor}
                try:
                    outputs = self.session.run(self.runtime_output_tensor_names, single_feed)
                    logits = np.array(outputs[0], dtype=np.float32)
                    logits_1d = np.squeeze(logits)
                    if logits_1d.ndim != 1:
                        logits_1d = logits_1d.reshape(-1)
                    probs = self._softmax(logits_1d)
                    top_index = int(np.argmax(probs))
                    confidence = float(probs[top_index])
                    external_label = self._resolve_external_label(top_index, int(probs.shape[0]))
                    mapped_label = self.class_mapping.get(external_label, external_label)
                    mapping_mode = "direct"
                    if mapped_label not in self.PROJECT_LABELS:
                        keyword_label, keyword_score, keyword_source = self._keyword_map_label(probs)
                        if keyword_label is not None:
                            mapped_label = keyword_label
                            confidence = max(confidence, keyword_score)
                            mapping_mode = "keyword"
                            external_label = keyword_source or external_label
                        else:
                            mapped_label = "Normal"
                            mapping_mode = "fallback_normal"
                    self.last_debug.update(
                        {
                            "used_fallback": False,
                            "error": "",
                            "input_mode": input_mode,
                            "mapping_mode": mapping_mode,
                            "external_top1_label": external_label,
                            "source_label_count": len(self.source_labels),
                            "top1_label": mapped_label,
                            "top1_score": round(confidence, 4),
                        }
                    )
                    results.append((mapped_label, round(confidence, 2)))
                except Exception as exc:
                    results.append(self._fallback(track_id, frame_id, f"onnx_infer_failed: {exc}"))
            return results

        try:
            outputs = self.session.run(self.runtime_output_tensor_names, input_feed)
            logits = np.array(outputs[0], dtype=np.float32)
            self.last_debug["raw_output_shape"] = list(logits.shape)

            if logits.ndim == 1:
                logits = np.expand_dims(logits, axis=0)
            elif logits.ndim > 2:
                logits = np.reshape(logits, (logits.shape[0], -1))

            results: list[tuple[str, float]] = []
            for index in range(len(tracks)):
                row = logits[index] if index < logits.shape[0] else logits[-1]
                probs = self._softmax(np.asarray(row, dtype=np.float32).reshape(-1))
                top_index = int(np.argmax(probs))
                confidence = float(probs[top_index])
                external_label = self._resolve_external_label(top_index, int(probs.shape[0]))
                mapped_label = self.class_mapping.get(external_label, external_label)
                mapping_mode = "direct"

                if mapped_label not in self.PROJECT_LABELS:
                    keyword_label, keyword_score, keyword_source = self._keyword_map_label(probs)
                    if keyword_label is not None:
                        mapped_label = keyword_label
                        confidence = max(confidence, keyword_score)
                        mapping_mode = "keyword"
                        external_label = keyword_source or external_label
                    else:
                        mapped_label = "Normal"
                        mapping_mode = "fallback_normal"

                results.append((mapped_label, round(confidence, 2)))
                self.last_debug.update(
                    {
                        "used_fallback": False,
                        "error": "",
                        "input_mode": input_mode,
                        "mapping_mode": mapping_mode,
                        "external_top1_label": external_label,
                        "source_label_count": len(self.source_labels),
                        "top1_label": mapped_label,
                        "top1_score": round(confidence, 4),
                    }
                )
            return results
        except Exception as exc:
            return [self._fallback(track_id, frame_id, f"onnx_infer_failed: {exc}") for track_id in track_ids]

    def clear_stale_tracks(self, active_track_ids: set[str]) -> None:
        stale = [track_id for track_id in self.track_pose_buffers if track_id not in active_track_ids]
        for track_id in stale:
            self.track_pose_buffers.pop(track_id, None)
        stale_video = [track_id for track_id in self.track_frame_buffers if track_id not in active_track_ids]
        for track_id in stale_video:
            self.track_frame_buffers.pop(track_id, None)

    def get_debug_info(self) -> dict[str, Any]:
        return {
            "onnx_path": self.onnx_path,
            "onnx_path_resolved": self.onnx_path_resolved,
            "input_clip_len": self.input_clip_len,
            "buffer_tracks": len(self.track_pose_buffers),
            "frame_buffer_tracks": len(self.track_frame_buffers),
            "input_hw": [self.input_height, self.input_width],
            "slowfast_alpha": self.slowfast_alpha,
            **self.last_debug,
        }
