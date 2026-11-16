from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any
import json
import math


class TemporalAdjudicator:
    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.apply_decision = bool(config.get("apply_decision", False))
        self.enable_heuristic = bool(config.get("enable_heuristic", True))
        self.window_size = int(max(4, config.get("window_size", 16)))
        self.min_frames = int(max(4, config.get("min_frames", 8)))
        self.model_path = str(config.get("model_path", ""))

        self.edge_margin_ratio = float(config.get("edge_margin_ratio", 0.18))
        self.risk_then_fall_score = float(config.get("risk_then_fall_score", 0.58))
        self.direct_fall_score = float(config.get("direct_fall_score", 0.62))

        self.feature_names = [
            "h03_ratio",
            "h04_ratio",
            "h02_ratio",
            "h01_ratio",
            "h03_conf_avg",
            "h04_conf_avg",
            "edge_risk_ratio",
            "high_risk_ratio",
        ]
        self.classes = ["direct_fall", "risk_then_fall", "non_fall"]
        self.weights: list[list[float]] | None = None
        self.bias: list[float] | None = None
        self.temperature = 1.0

        self.frame_features: deque[list[float]] = deque(maxlen=self.window_size)
        self.frame_ids: deque[int] = deque(maxlen=self.window_size)

        self._load_model_if_any()

    def _load_model_if_any(self) -> None:
        if not self.model_path:
            return
        path = Path(self.model_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            path = (project_root / path).resolve()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            classes = list(data.get("classes", []))
            weights = list(data.get("weights", []))
            bias = list(data.get("bias", []))
            if classes and weights and bias and len(weights) == len(classes) and len(bias) == len(classes):
                self.classes = [str(item) for item in classes]
                self.weights = [[float(v) for v in row] for row in weights]
                self.bias = [float(v) for v in bias]
                self.temperature = float(max(1e-6, data.get("temperature", 1.0)))
                feature_names = list(data.get("feature_names", []))
                if feature_names:
                    self.feature_names = [str(item) for item in feature_names]
        except Exception:
            self.weights = None
            self.bias = None

    @staticmethod
    def _bbox_center_xywh(bbox: list[float]) -> tuple[float, float]:
        x, y, w, h = [float(v) for v in bbox]
        return x + w * 0.5, y + h * 0.5

    def _extract_frame_features(self, pending_tracks: list[dict[str, Any]], frame_shape: tuple[int, int]) -> list[float]:
        if not pending_tracks:
            return [0.0 for _ in self.feature_names]

        frame_h, frame_w = frame_shape
        total = float(max(1, len(pending_tracks)))

        h01 = h02 = h03 = h04 = 0.0
        h03_conf = 0.0
        h04_conf = 0.0
        edge_risk = 0.0
        high_risk = 0.0

        edge_x = frame_w * self.edge_margin_ratio
        edge_y = frame_h * self.edge_margin_ratio

        for item in pending_tracks:
            label = str(item.get("label", "Normal"))
            conf = float(item.get("confidence", 0.0))
            bbox = list(item.get("bbox", [0.0, 0.0, 0.0, 0.0]))
            cx, cy = self._bbox_center_xywh(bbox)
            near_edge = cx <= edge_x or cx >= frame_w - edge_x or cy <= edge_y or cy >= frame_h - edge_y

            if label.startswith("H"):
                high_risk += 1.0
            if label == "H01":
                h01 += 1.0
            elif label == "H02":
                h02 += 1.0
            elif label == "H03":
                h03 += 1.0
                h03_conf += conf
                if near_edge:
                    edge_risk += 1.0
            elif label == "H04":
                h04 += 1.0
                h04_conf += conf
                if near_edge:
                    edge_risk += 1.0

        return [
            h03 / total,
            h04 / total,
            h02 / total,
            h01 / total,
            h03_conf / max(1.0, h03),
            h04_conf / max(1.0, h04),
            edge_risk / total,
            high_risk / total,
        ]

    @staticmethod
    def _softmax(values: list[float]) -> list[float]:
        if not values:
            return []
        vmax = max(values)
        exp_values = [math.exp(v - vmax) for v in values]
        denom = sum(exp_values)
        if denom <= 0:
            return [0.0 for _ in values]
        return [v / denom for v in exp_values]

    def _predict_with_linear_model(self, feature_vector: list[float]) -> tuple[str, float] | None:
        if self.weights is None or self.bias is None:
            return None
        if not self.weights or not self.bias:
            return None

        logits: list[float] = []
        for row, b in zip(self.weights, self.bias):
            score = float(b)
            for i in range(min(len(row), len(feature_vector))):
                score += float(row[i]) * float(feature_vector[i])
            logits.append(score / self.temperature)

        probs = self._softmax(logits)
        if not probs:
            return None
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        label = self.classes[best_idx] if best_idx < len(self.classes) else "non_fall"
        return label, float(probs[best_idx])

    def _predict_with_heuristics(self, feature_vector: list[float]) -> tuple[str, float]:
        h03_ratio, h04_ratio, _, _, h03_conf, h04_conf, edge_risk_ratio, _ = feature_vector

        risk_score = 0.55 * h04_ratio + 0.35 * edge_risk_ratio + 0.10 * h04_conf
        direct_score = 0.62 * h03_ratio + 0.28 * h03_conf - 0.15 * h04_ratio - 0.10 * edge_risk_ratio

        if risk_score >= self.risk_then_fall_score and risk_score >= (direct_score + 0.05):
            return "risk_then_fall", float(min(0.99, max(0.0, risk_score)))
        if direct_score >= self.direct_fall_score:
            return "direct_fall", float(min(0.99, max(0.0, direct_score)))
        return "non_fall", float(min(0.99, max(0.0, 1.0 - max(risk_score, direct_score))))

    def _aggregate_window(self) -> list[float]:
        if not self.frame_features:
            return [0.0 for _ in self.feature_names]
        size = float(len(self.frame_features))
        dim = len(self.frame_features[0])
        agg = [0.0] * dim
        for vec in self.frame_features:
            for i in range(dim):
                agg[i] += float(vec[i])
        return [v / size for v in agg]

    def update(
        self,
        frame_id: int,
        pending_tracks: list[dict[str, Any]],
        frame_shape: tuple[int, int],
        current_label: str,
        current_confidence: float,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {
                "output_label": current_label,
                "output_confidence": current_confidence,
                "prediction": "disabled",
                "prediction_score": 0.0,
                "policy_applied": False,
                "debug": {"enabled": False},
            }

        frame_feature = self._extract_frame_features(pending_tracks, frame_shape)
        self.frame_features.append(frame_feature)
        self.frame_ids.append(int(frame_id))

        prediction = "insufficient_window"
        prediction_score = 0.0
        feature_vector = self._aggregate_window()

        if len(self.frame_features) >= self.min_frames:
            model_result = self._predict_with_linear_model(feature_vector)
            if model_result is not None:
                prediction, prediction_score = model_result
            elif self.enable_heuristic:
                prediction, prediction_score = self._predict_with_heuristics(feature_vector)

        output_label = str(current_label)
        output_confidence = float(current_confidence)
        policy_applied = False

        if self.apply_decision and prediction in {"direct_fall", "risk_then_fall"}:
            if prediction == "direct_fall" and output_label == "H04":
                output_label = "H03"
                output_confidence = max(output_confidence, min(0.98, prediction_score + 0.08))
                policy_applied = True
            elif prediction == "risk_then_fall" and output_label == "H03":
                output_label = "H04"
                output_confidence = max(output_confidence, min(0.98, prediction_score + 0.06))
                policy_applied = True

        debug = {
            "enabled": True,
            "apply_decision": self.apply_decision,
            "window_size": self.window_size,
            "min_frames": self.min_frames,
            "window_len": len(self.frame_features),
            "prediction": prediction,
            "prediction_score": round(float(prediction_score), 4),
            "feature_names": self.feature_names,
            "feature_vector": [round(float(v), 4) for v in feature_vector],
            "policy_applied": policy_applied,
            "output_label": output_label,
            "output_confidence": round(float(output_confidence), 4),
        }

        return {
            "output_label": output_label,
            "output_confidence": output_confidence,
            "prediction": prediction,
            "prediction_score": prediction_score,
            "policy_applied": policy_applied,
            "debug": debug,
        }
