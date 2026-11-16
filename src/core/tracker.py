from __future__ import annotations

from typing import Any


class Tracker:
    """多目标跟踪器（ROADMAP Stage 0a 修复版）。

    相对旧"纯 IOU 贪心"版的三点增强（根因：纯 IOU 在快速运动下 IoU 归零，
    连环断 ID → 姿态缓冲混人 → 全类别精度受限）：
    1. 恒速运动预测：track 维护质心速度（EMA 平滑），匹配用速度外推的预测框；
    2. 两阶段匹配：先 IoU（预测框口径），失败后对剩余"track×检测"做
       质心距离贪心兜底，兜底受速度门控与尺寸合理性约束；
    3. 兜底命中同样学习速度，快速运动目标在 1-2 帧内被重新捕获后
       转入预测框 IoU 稳定匹配。

    接口与旧版完全一致（update / advance_without_detections）。
    """

    def __init__(self, config: dict[str, Any]):
        self.name = config.get("name", "bytetrack")
        self.match_threshold = float(config.get("match_threshold", 0.6))
        self.max_missed = int(max(1, config.get("track_buffer", 30)))
        # Stage 0a 参数
        self.use_motion_predict = bool(config.get("use_motion_predict", True))
        self.enable_fallback = bool(config.get("enable_fallback_match", True))
        # 兜底速度门控：质心位移 / 检测框对角线 ≤ gate × (missed+1) 才允许配对
        self.fallback_speed_gate = float(config.get("fallback_speed_gate", 0.8))
        self.velocity_ema = float(min(max(float(config.get("velocity_ema", 0.5)), 0.0), 1.0))
        self.min_area_ratio = float(config.get("fallback_min_area_ratio", 0.4))
        self.max_area_ratio = float(config.get("fallback_max_area_ratio", 2.5))
        self._next_id = 1
        self._tracks: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _iou_xywh(box_a: list[float], box_b: list[float]) -> float:
        ax, ay, aw, ah = [float(v) for v in box_a]
        bx, by, bw, bh = [float(v) for v in box_b]
        ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
        bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)

        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, aw) * max(0.0, ah)
        area_b = max(0.0, bw) * max(0.0, bh)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center(bbox: list[float]) -> tuple[float, float]:
        x, y, w, h = [float(v) for v in bbox]
        return x + w / 2.0, y + h / 2.0

    @staticmethod
    def _diag(bbox: list[float]) -> float:
        _, _, w, h = [max(0.0, float(v)) for v in bbox]
        return (w * w + h * h) ** 0.5

    def _predict_bbox(self, track: dict[str, Any]) -> list[float]:
        """速度外推预测框：外推步数 = 已丢失帧数 + 1（当前帧）。"""
        bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
        if not self.use_motion_predict:
            return bbox
        vx = track.get("vx")
        vy = track.get("vy")
        if vx is None or vy is None or (vx == 0.0 and vy == 0.0):
            return bbox
        miss = int(track.get("missed", 0)) + 1
        return [bbox[0] + vx * miss, bbox[1] + vy * miss, bbox[2], bbox[3]]

    def _clamp_velocity(self, track: dict[str, Any]) -> None:
        limit = self.fallback_speed_gate * max(1.0, self._diag(track["bbox"]))
        speed = (track["vx"] ** 2 + track["vy"] ** 2) ** 0.5
        if speed > limit > 0:
            scale = limit / speed
            track["vx"] *= scale
            track["vy"] *= scale

    def _attach(self, track: dict[str, Any], bbox: list[float], score: float, miss_span: int) -> None:
        """匹配成功：更新框/得分/速度。miss_span 为距上次匹配间隔的处理帧数。"""
        cx, cy = self._center(bbox)
        if self.use_motion_predict and track.get("cx") is not None:
            vx_new = (cx - float(track["cx"])) / max(1, miss_span + 1)
            vy_new = (cy - float(track["cy"])) / max(1, miss_span + 1)
            if track.get("vx") is None:
                track["vx"], track["vy"] = vx_new, vy_new
            else:
                a = self.velocity_ema
                track["vx"] = a * vx_new + (1 - a) * float(track["vx"])
                track["vy"] = a * vy_new + (1 - a) * float(track["vy"])
        track["bbox"] = [int(v) for v in bbox]
        track["cx"], track["cy"] = cx, cy
        track["score"] = score
        track["missed"] = 0
        track["age"] = int(track.get("age", 0)) + 1
        track["matched"] = True
        if track.get("vx") is not None:
            self._clamp_velocity(track)

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for track in self._tracks.values():
            track["matched"] = False

        # 阶段一：预测框 IoU 匹配（检测按置信度降序，降低处理顺序敏感性）
        ordered = sorted(detections, key=lambda d: -float(d.get("score", 0.0)))
        claimed: set[int] = set()
        for det in ordered:
            bbox = det.get("bbox", [0, 0, 0, 0])
            best_track_id = ""
            best_iou = 0.0
            for track_id, track in self._tracks.items():
                if track.get("matched"):
                    continue
                iou = self._iou_xywh(self._predict_bbox(track), bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            if best_track_id and best_iou >= self.match_threshold:
                self._attach(self._tracks[best_track_id], bbox, float(det.get("score", 0.0)), 0)
                claimed.add(id(det))

        # 阶段二：兜底匹配——剩余 track × 剩余检测，按归一化质心距离升序贪心，
        # 受速度门控（≤ gate×(missed+1)×对角线）与尺寸合理性约束
        if self.enable_fallback:
            open_tracks = [t for t in self._tracks.values() if not t.get("matched")]
            open_dets = [(i, d) for i, d in enumerate(detections) if id(d) not in claimed]
            pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
            for _, det in open_dets:
                bbox = det.get("bbox", [0, 0, 0, 0])
                dcx, dcy = self._center(bbox)
                det_diag = max(1.0, self._diag(bbox))
                det_area = max(1.0, abs(float(bbox[2]) * float(bbox[3])))
                for track in open_tracks:
                    miss = int(track.get("missed", 0)) + 1
                    pcx, pcy = self._center(self._predict_bbox(track))
                    dist = ((pcx - dcx) ** 2 + (pcy - dcy) ** 2) ** 0.5 / det_diag
                    if dist > self.fallback_speed_gate * miss:
                        continue
                    t_area = max(1.0, abs(float(track["bbox"][2]) * float(track["bbox"][3])))
                    ratio = t_area / det_area
                    if not (self.min_area_ratio <= ratio <= self.max_area_ratio):
                        continue
                    pairs.append((dist, track, det))
            pairs.sort(key=lambda item: item[0])
            for dist, track, det in pairs:
                if track.get("matched") or id(det) in claimed:
                    continue
                miss = int(track.get("missed", 0)) + 1
                self._attach(track, det.get("bbox", [0, 0, 0, 0]), float(det.get("score", 0.0)), miss - 1)
                claimed.add(id(det))

        # 阶段三：仍未认领的检测 → 新建 track
        for det in detections:
            if id(det) not in claimed:
                self._register(det.get("bbox", [0, 0, 0, 0]), float(det.get("score", 0.0)))

        # 生命周期：未匹配推进/超限删除
        stale_ids: list[str] = []
        for track_id, track in self._tracks.items():
            if not track.get("matched"):
                track["missed"] = int(track.get("missed", 0)) + 1
            if int(track.get("missed", 0)) > self.max_missed:
                stale_ids.append(track_id)
        for track_id in stale_ids:
            self._tracks.pop(track_id, None)

        active_tracks: list[dict[str, Any]] = []
        for track in self._tracks.values():
            if track.get("matched"):
                active_tracks.append(
                    {
                        "track_id": track["track_id"],
                        "bbox": list(track["bbox"]),
                        "score": float(track["score"]),
                        "age": int(track.get("age", 1)),
                    }
                )
        return active_tracks

    def _register(self, bbox: list[float], score: float) -> str:
        track_id = f"P{self._next_id:03d}"
        self._next_id += 1
        cx, cy = self._center(bbox)
        self._tracks[track_id] = {
            "track_id": track_id,
            "bbox": [int(v) for v in bbox],
            "score": score,
            "missed": 0,
            "age": 1,
            "matched": True,
            "cx": cx,
            "cy": cy,
            "vx": None,
            "vy": None,
        }
        return track_id

    def advance_without_detections(self) -> list[dict[str, Any]]:
        stale_ids: list[str] = []

        for track_id, track in self._tracks.items():
            track["matched"] = False
            track["missed"] = int(track.get("missed", 0)) + 1
            track["age"] = int(track.get("age", 0)) + 1
            if int(track.get("missed", 0)) > self.max_missed:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            self._tracks.pop(track_id, None)

        active_tracks: list[dict[str, Any]] = []
        for track in self._tracks.values():
            active_tracks.append(
                {
                    "track_id": track["track_id"],
                    "bbox": list(track["bbox"]),
                    "score": float(track["score"]),
                    "age": int(track.get("age", 1)),
                }
            )

        return active_tracks
