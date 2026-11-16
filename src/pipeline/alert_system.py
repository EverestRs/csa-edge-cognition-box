from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np


class AlertSystem:
    def __init__(self, output_dir: str, config: dict | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.output_dir / "snapshots"
        config = config or {}
        self.min_consecutive_frames = int(max(1, config.get("min_consecutive_frames", 3)))
        self.cooldown_seconds = float(max(0, config.get("cooldown_seconds", 5)))
        self.high_risk_min_confidence = float(max(0, config.get("high_risk_min_confidence", 0.55)))
        self._consecutive_frames: dict[str, int] = {}
        self._last_alert_time: dict[str, float] = {}

    def build_alert(self, label: str, confidence: float, frame_id: int, timestamp: float | None = None,
                    frame: np.ndarray | None = None) -> dict:
        now = timestamp or time.time()
        if not (label.startswith("H") and confidence >= self.high_risk_min_confidence):
            self._consecutive_frames.clear()
            return self._no_alert()

        alert_type = f"{label.lower()}_detected"

        # cooldown check: skip if same alert type was recently triggered
        last_alert = self._last_alert_time.get(alert_type, 0.0)
        if last_alert > 0 and (now - last_alert) < self.cooldown_seconds:
            self._consecutive_frames[alert_type] = 0
            return self._no_alert()

        # accumulate consecutive frames
        count = self._consecutive_frames.get(alert_type, 0) + 1
        self._consecutive_frames[alert_type] = count
        if count < self.min_consecutive_frames:
            return self._no_alert()

        # alert triggered
        self._last_alert_time[alert_type] = now
        self._consecutive_frames[alert_type] = 0
        # 快照落盘（WEAKNESS_AUDIT A-1 修复）：frame 由引擎在触发点传入——
        # 此前 url 指向从未生成的 /snapshots/... 假路径，告警证据链断裂
        snapshot_url = ""
        if frame is not None:
            try:
                self.snapshots_dir.mkdir(parents=True, exist_ok=True)
                snap_path = (self.snapshots_dir / f"alert_{frame_id}.jpg").resolve()
                if cv2.imwrite(str(snap_path), frame):
                    snapshot_url = f"/api/v1/media?path={snap_path}"
            except Exception:
                snapshot_url = ""
        return {
            "triggered": True,
            "type": alert_type,
            "source": "base_model",
            "final_decision": "pending",
            "consecutive_frames": count,
            "cooldown_seconds": self.cooldown_seconds,
            "snapshot_url": snapshot_url,
            "clip_path": "",  # 视频切片功能未实现，如实留空
        }

    @staticmethod
    def _no_alert() -> dict:
        return {
            "triggered": False,
            "type": "none",
            "source": "base_model",
            "final_decision": "pending",
            "snapshot_url": "",
            "clip_path": "",
        }

    def write_event(self, event: dict) -> None:
        event_id = event.get("frame_id", "unknown")
        out_file = self.output_dir / f"event_{event_id}.json"
        out_file.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
