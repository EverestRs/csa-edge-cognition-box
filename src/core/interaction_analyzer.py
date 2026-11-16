from __future__ import annotations

from math import hypot
from typing import Any


class InteractionAnalyzer:
    @staticmethod
    def _mean_point(points: list[list[float]]) -> tuple[float, float] | None:
        visible = [point for point in points if len(point) >= 2 and (float(point[0]) != 0.0 or float(point[1]) != 0.0)]
        if not visible:
            return None
        return sum(float(point[0]) for point in visible) / len(visible), sum(float(point[1]) for point in visible) / len(visible)

    @staticmethod
    def _bbox_center(bbox: list[float]) -> tuple[float, float]:
        x, y, w, h = [float(value) for value in bbox]
        return x + w * 0.5, y + h * 0.5

    @staticmethod
    def _bbox_diag(bbox: list[float]) -> float:
        _, _, w, h = [float(value) for value in bbox]
        return max(1.0, (w * w + h * h) ** 0.5)

    @classmethod
    def _upper_body_activity(cls, keypoints: list[list[float]], bbox: list[float]) -> tuple[float, float]:
        diag = cls._bbox_diag(bbox)
        shoulders = [keypoints[idx] for idx in (5, 6) if idx < len(keypoints) and len(keypoints[idx]) >= 2]
        wrists = [keypoints[idx] for idx in (9, 10) if idx < len(keypoints) and len(keypoints[idx]) >= 2]
        head = [keypoints[idx] for idx in (0, 1, 2, 3, 4) if idx < len(keypoints) and len(keypoints[idx]) >= 2]

        shoulder_pt = cls._mean_point(shoulders) if shoulders else None
        head_pt = cls._mean_point(head) if head else None
        wrist_pt = cls._mean_point(wrists) if wrists else None

        if shoulder_pt is None or wrist_pt is None:
            activity = 0.0
        else:
            activity = hypot(wrist_pt[0] - shoulder_pt[0], wrist_pt[1] - shoulder_pt[1]) / diag

        if head_pt is None or wrist_pt is None:
            defense = 0.0
        else:
            defense = hypot(wrist_pt[0] - head_pt[0], wrist_pt[1] - head_pt[1]) / diag

        return activity, defense

    @classmethod
    def _defensive_posture(cls, keypoints: list[list[float]], bbox: list[float]) -> float:
        diag = cls._bbox_diag(bbox)
        head = [keypoints[idx] for idx in (0, 1, 2, 3, 4) if idx < len(keypoints) and len(keypoints[idx]) >= 2]
        hips = [keypoints[idx] for idx in (11, 12) if idx < len(keypoints) and len(keypoints[idx]) >= 2]
        wrists = [keypoints[idx] for idx in (9, 10) if idx < len(keypoints) and len(keypoints[idx]) >= 2]
        head_pt = cls._mean_point(head) if head else None
        hip_pt = cls._mean_point(hips) if hips else None
        wrist_pt = cls._mean_point(wrists) if wrists else None
        if head_pt is None or hip_pt is None or wrist_pt is None:
            return 0.0
        crouch = max(0.0, (hip_pt[1] - head_pt[1]) / diag)
        cover = max(0.0, (head_pt[1] - min(wrist_pt[1], head_pt[1])) / diag)
        return crouch + cover

    def infer_role(
        self,
        label: str,
        track_id: str,
        bbox: list[float] | None = None,
        keypoints: list[list[float]] | None = None,
        peer_tracks: list[dict[str, Any]] | None = None,
    ) -> str:
        bbox = bbox or [0.0, 0.0, 0.0, 0.0]
        keypoints = keypoints or []
        peer_tracks = peer_tracks or []

        if label in {"M02", "M03", "S01", "S02"}:
            return "bystander"

        if label == "H03":
            return "victim"

        if label not in {"H01", "H02"}:
            return "unknown"

        center = self._bbox_center(bbox)
        diag = self._bbox_diag(bbox)
        close_peers = 0
        peer_activity = 0.0
        for peer in peer_tracks:
            if peer.get("track_id") == track_id:
                continue
            peer_bbox = peer.get("bbox", [0.0, 0.0, 0.0, 0.0])
            peer_center = self._bbox_center(peer_bbox)
            if hypot(center[0] - peer_center[0], center[1] - peer_center[1]) <= max(diag, self._bbox_diag(peer_bbox)) * 1.1:
                close_peers += 1
            peer_activity = max(peer_activity, float(peer.get("activity_score", 0.0)))

        if close_peers == 0:
            return "unknown"

        activity, defense = self._upper_body_activity(keypoints, bbox)
        defensive_posture = self._defensive_posture(keypoints, bbox)

        if label == "H02":
            if defensive_posture >= 0.55 or defense >= 0.18:
                return "victim"
            if activity >= max(0.18, peer_activity * 0.75):
                return "attacker"
            return "victim" if close_peers > 0 else "bystander"

        if activity >= 0.18 or peer_activity >= 0.18:
            if defensive_posture >= 0.55:
                return "victim"
            return "attacker"

        return "bystander"

    def infer_roles(self, tracks: list[dict[str, Any]]) -> dict[str, str]:
        role_map: dict[str, str] = {}
        for track in tracks:
            role_map[str(track.get("track_id", ""))] = self.infer_role(
                str(track.get("label", "Normal")),
                str(track.get("track_id", "")),
                bbox=list(track.get("bbox", [0.0, 0.0, 0.0, 0.0])),
                keypoints=list(track.get("keypoints", [])),
                peer_tracks=tracks,
            )
        return role_map
