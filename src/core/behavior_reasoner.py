from __future__ import annotations

from collections import defaultdict, deque
from typing import Any


class BehaviorReasoner:
    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.loiter_min_frames = int(max(10, config.get("loiter_min_frames", 90)))
        self.loiter_radius_px = float(max(20.0, config.get("loiter_radius_px", 80.0)))
        self.fall_narrow_aspect_ratio = float(config.get("fall_narrow_aspect_ratio", 0.72))
        # S01 徘徊判定（视频时间口径）：pacing=来回走动；dwell=长时滞留（默认关）
        self.loiter_min_seconds = float(max(10.0, config.get("loiter_min_seconds", 45.0)))
        self.loiter_window_seconds = float(max(10.0, config.get("loiter_window_seconds", 45.0)))
        self.loiter_path_ratio = float(max(1.1, config.get("loiter_path_ratio", 2.5)))
        self.loiter_radius_ratio = float(max(0.0, config.get("loiter_radius_ratio", 0.35)))
        self.loiter_extent_ratio = float(max(0.02, config.get("loiter_extent_ratio", 0.10)))
        self.enable_dwell_loiter = bool(config.get("enable_dwell_loiter", False))
        self.loiter_dwell_min_seconds = float(max(30.0, config.get("loiter_dwell_min_seconds", 300.0)))
        self.loiter_conf_cap = float(min(0.99, max(0.5, config.get("loiter_conf_cap", 0.90))))
        self._source_fps = 25.0
        self.phone_min_frames = int(max(6, config.get("phone_min_frames", 24)))
        self.phone_hand_head_ratio = float(config.get("phone_hand_head_ratio", 0.16))
        self.phone_head_tilt_ratio = float(config.get("phone_head_tilt_ratio", 0.12))
        self.tamper_min_frames = int(max(6, config.get("tamper_min_frames", 18)))
        self.tamper_bbox_area_ratio = float(config.get("tamper_bbox_area_ratio", 0.22))
        self.tamper_center_radius_ratio = float(config.get("tamper_center_radius_ratio", 0.30))
        self.fall_min_frames = int(max(4, config.get("fall_min_frames", 8)))
        self.fall_aspect_ratio = float(config.get("fall_aspect_ratio", 0.95))
        self.fall_drop_ratio = float(config.get("fall_drop_ratio", 0.08))
        self.fall_motion_ratio = float(config.get("fall_motion_ratio", 0.15))
        self.fall_pose_horizontal_ratio = float(config.get("fall_pose_horizontal_ratio", 0.78))
        self.fall_floor_margin_ratio = float(config.get("fall_floor_margin_ratio", 0.12))
        self.fall_center_low_ratio = float(config.get("fall_center_low_ratio", 0.60))
        self.fall_still_motion_ratio = float(config.get("fall_still_motion_ratio", 0.08))
        self.fall_head_low_ratio = float(config.get("fall_head_low_ratio", 0.58))
        self.fall_event_ttl_calls = int(max(8, config.get("fall_event_ttl_calls", 24)))
        self.fall_event_radius_ratio = float(config.get("fall_event_radius_ratio", 1.0))
        self.h03_hold_frames = int(max(2, config.get("h03_hold_frames", 4)))
        self.h02_min_frames = int(max(3, config.get("h02_min_frames", 6)))
        self.h01_min_frames = int(max(3, config.get("h01_min_frames", 5)))
        self.h02_motion_ratio = float(config.get("h02_motion_ratio", 0.18))
        self.h01_motion_ratio = float(config.get("h01_motion_ratio", 0.22))
        self.interaction_close_ratio = float(config.get("interaction_close_ratio", 1.35))
        self.interaction_contact_iou = float(config.get("interaction_contact_iou", 0.02))
        self.h02_pair_min_frames = int(max(2, config.get("h02_pair_min_frames", self.h02_min_frames)))
        self.h01_pair_min_frames = int(max(2, config.get("h01_pair_min_frames", self.h01_min_frames)))
        self.h01_dual_active_ratio = float(config.get("h01_dual_active_ratio", max(0.10, self.h01_motion_ratio * 0.8)))
        self.h04_edge_impact_motion_ratio = float(config.get("h04_edge_impact_motion_ratio", 0.20))
        self.h04_edge_fast_motion_ratio = float(config.get("h04_edge_fast_motion_ratio", 0.30))
        self.h04_impact_min_frames = int(max(2, config.get("h04_impact_min_frames", 3)))
        self.h04_pre_fall_hold_frames = int(max(2, config.get("h04_pre_fall_hold_frames", 8)))
        self.h01_transition_min_base_conf = float(config.get("h01_transition_min_base_conf", 0.62))
        self.h04_transition_min_base_conf = float(config.get("h04_transition_min_base_conf", 0.62))
        self.h01_pair_transition_min_frames = int(max(2, config.get("h01_pair_transition_min_frames", 2)))
        self.h04_transition_min_frames = int(max(1, config.get("h04_transition_min_frames", 2)))
        self.h04_transition_hold_frames = int(max(1, config.get("h04_transition_hold_frames", 3)))
        self.h04_transition_cooldown_frames = int(max(2, config.get("h04_transition_cooldown_frames", 18)))
        self.h04_transition_min_impact_counter = int(max(1, config.get("h04_transition_min_impact_counter", 1)))
        self.h04_transition_same_risk_margin = float(config.get("h04_transition_same_risk_margin", 0.10))
        self.h03_risk_pre_frames = int(max(1, config.get("h03_risk_pre_frames", 2)))
        self.h03_mode_hold_frames = int(max(2, config.get("h03_mode_hold_frames", 10)))
        self.smoke_min_frames = int(max(4, config.get("smoke_min_frames", 12)))
        self.smoke_hand_mouth_ratio = float(config.get("smoke_hand_mouth_ratio", 0.14))
        self.smoke_unilateral_ratio = float(config.get("smoke_unilateral_ratio", 1.45))
        self.smoke_head_y_tolerance_ratio = float(config.get("smoke_head_y_tolerance_ratio", 0.18))
        self.smoke_hand_shoulder_offset_ratio = float(config.get("smoke_hand_shoulder_offset_ratio", 0.22))
        self.vandal_min_frames = int(max(4, config.get("vandal_min_frames", 8)))
        self.vandal_motion_ratio = float(config.get("vandal_motion_ratio", 0.14))
        self.danger_edge_margin_ratio = float(config.get("danger_edge_margin_ratio", 0.14))
        self.default_rule_conf = float(config.get("default_rule_conf", 0.72))
        self.forbidden_zones = list(config.get("forbidden_zones", []))
        self.danger_zones = list(config.get("danger_zones", []))

        self.track_centers: dict[str, deque[tuple[int, tuple[float, float]]]] = defaultdict(lambda: deque(maxlen=900))
        self.track_first_frame_id: dict[str, int] = {}
        self.track_bboxes: dict[str, deque[list[float]]] = defaultdict(lambda: deque(maxlen=120))
        self.track_last_center: dict[str, tuple[float, float]] = {}
        self.track_last_area: dict[str, float] = {}
        self.track_phone_counter: dict[str, int] = defaultdict(int)
        self.track_tamper_counter: dict[str, int] = defaultdict(int)
        self.track_fall_counter: dict[str, int] = defaultdict(int)
        self.track_h03_counter: dict[str, int] = defaultdict(int)
        self.track_h02_counter: dict[str, int] = defaultdict(int)
        self.track_h01_counter: dict[str, int] = defaultdict(int)
        self.track_base_h02_counter: dict[str, int] = defaultdict(int)
        self.track_base_h03_counter: dict[str, int] = defaultdict(int)
        self.track_motion_level: dict[str, float] = defaultdict(float)
        self.track_impact_counter: dict[str, int] = defaultdict(int)
        self.track_h04_hold_counter: dict[str, int] = defaultdict(int)
        self.track_h04_transition_counter: dict[str, int] = defaultdict(int)
        self.track_h04_transition_cooldown: dict[str, int] = defaultdict(int)
        self.track_h03_risk_pre_counter: dict[str, int] = defaultdict(int)
        self.track_h03_mode_hold_counter: dict[str, int] = defaultdict(int)
        self.track_h03_mode: dict[str, str] = defaultdict(str)
        self.pair_h02_counter: dict[str, int] = defaultdict(int)
        self.pair_h01_counter: dict[str, int] = defaultdict(int)
        self.pair_h01_transition_counter: dict[str, int] = defaultdict(int)
        self.pair_contact_counter: dict[str, int] = defaultdict(int)
        self.pair_dual_active_counter: dict[str, int] = defaultdict(int)
        self.pair_last_distance: dict[str, float] = {}
        self.track_smoke_counter: dict[str, int] = defaultdict(int)
        self.track_vandal_counter: dict[str, int] = defaultdict(int)
        self.track_danger_counter: dict[str, int] = defaultdict(int)
        self.reasoner_tick = 0
        self.recent_fall_events: deque[dict[str, float]] = deque(maxlen=96)

    @staticmethod
    def _bbox_center_xy(bbox: list[float]) -> tuple[float, float]:
        x, y, w, h = [float(v) for v in bbox]
        return x + w * 0.5, y + h * 0.5

    @staticmethod
    def _bbox_diag(bbox: list[float]) -> float:
        _, _, w, h = [float(v) for v in bbox]
        return max(1.0, (w * w + h * h) ** 0.5)

    @staticmethod
    def _point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
        x, y = point
        inside = False
        n = len(polygon)
        if n < 3:
            return False
        j = n - 1
        for i in range(n):
            xi, yi = float(polygon[i][0]), float(polygon[i][1])
            xj, yj = float(polygon[j][0]), float(polygon[j][1])
            intersects = (yi > y) != (yj > y)
            if intersects:
                cross_x = (xj - xi) * (y - yi) / max(1e-6, (yj - yi)) + xi
                if x < cross_x:
                    inside = not inside
            j = i
        return inside

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def update_zones(self, forbidden_zones: list[dict] | None = None,
                     danger_zones: list[dict] | None = None) -> None:
        """热更新布控区域（REST API 保存后调用，立即生效无需重启）。"""
        if forbidden_zones is not None:
            self.forbidden_zones = list(forbidden_zones)
        if danger_zones is not None:
            self.danger_zones = list(danger_zones)

    @staticmethod
    def _is_intrusion(center: tuple[float, float], zones: list[dict]) -> bool:
        if not zones:
            return False
        cx, cy = center
        for zone in zones:
            polygon = zone.get("polygon", [])
            if polygon and len(polygon) >= 3:
                if BehaviorReasoner._point_in_polygon((cx, cy), polygon):
                    return True
        return False

    @staticmethod
    def _close_peer_ids(
        track_id: str,
        bbox: list[float],
        peer_tracks: list[dict[str, Any]],
    ) -> set[str]:
        """返回与当前 track 距离较近的 peer track ID 集合。"""
        cx, cy = BehaviorReasoner._bbox_center_xy(bbox)
        diag = BehaviorReasoner._bbox_diag(bbox)
        close: set[str] = set()
        for peer in peer_tracks:
            pid = str(peer.get("track_id", ""))
            if pid == track_id or not pid:
                continue
            peer_bbox = peer.get("bbox", [0, 0, 0, 0])
            pcx, pcy = BehaviorReasoner._bbox_center_xy(peer_bbox)
            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            if dist <= max(diag, BehaviorReasoner._bbox_diag(peer_bbox)) * 1.5:
                close.add(pid)
        return close

    @staticmethod
    def _upper_limb_activity(
        keypoints: list[list[float]],
        bbox: list[float],
    ) -> float:
        """计算上肢活动度。"""
        wrists = BehaviorReasoner._valid_points(keypoints, [9, 10])
        shoulders = BehaviorReasoner._valid_points(keypoints, [5, 6])
        if not wrists or not shoulders:
            return 0.0
        wrist_mean = BehaviorReasoner._point_mean(wrists)
        shoulder_mean = BehaviorReasoner._point_mean(shoulders)
        if wrist_mean is None or shoulder_mean is None:
            return 0.0
        diag = BehaviorReasoner._bbox_diag(bbox)
        if diag < 1.0:
            return 0.0
        dx = wrist_mean[0] - shoulder_mean[0]
        dy = wrist_mean[1] - shoulder_mean[1]
        return (dx * dx + dy * dy) ** 0.5 / diag

    @staticmethod
    def _pair_key(id1: str, id2: str) -> str:
        return f"{min(id1, id2)}:{max(id1, id2)}"

    @staticmethod
    def _bbox_iou_xywh(a: list[float], b: list[float]) -> float:
        ax1, ay1, aw, ah = [float(v) for v in a]
        bx1, by1, bw, bh = [float(v) for v in b]
        ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
        bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
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
    def _valid_points(keypoints: list[list[float]], indices: list[int]) -> list[list[float]]:
        points: list[list[float]] = []
        for idx in indices:
            if idx < len(keypoints) and len(keypoints[idx]) >= 2:
                x = float(keypoints[idx][0])
                y = float(keypoints[idx][1])
                if x != 0.0 or y != 0.0:
                    points.append([x, y, float(keypoints[idx][2]) if len(keypoints[idx]) > 2 else 1.0])
        return points

    @staticmethod
    def _point_mean(points: list[list[float]]) -> tuple[float, float] | None:
        if not points:
            return None
        return (
            sum(float(point[0]) for point in points) / len(points),
            sum(float(point[1]) for point in points) / len(points),
        )

    def _pose_head_points(self, keypoints: list[list[float]]) -> list[list[float]]:
        return self._valid_points(keypoints, [0, 1, 2, 3, 4])

    def _pose_shoulder_points(self, keypoints: list[list[float]]) -> list[list[float]]:
        return self._valid_points(keypoints, [5, 6])

    def _pose_hand_points(self, keypoints: list[list[float]]) -> list[list[float]]:
        # 腕部 + 全身体手部关键点；肘部（7,8）剔除——坐姿托腮时肘-鼻投影距离近，是 M01 误触发的几何来源
        points = self._valid_points(keypoints, [9, 10])
        if len(keypoints) > 90:
            points.extend(self._valid_points(keypoints, list(range(91, min(133, len(keypoints))))))
        return points

    def _pose_hip_points(self, keypoints: list[list[float]]) -> list[list[float]]:
        return self._valid_points(keypoints, [11, 12])

    def _pose_foot_points(self, keypoints: list[list[float]]) -> list[list[float]]:
        return self._valid_points(keypoints, [15, 16])

    def set_source_fps(self, fps: float) -> None:
        """由引擎注入视频源帧率，驱动徘徊判定的视频时间口径。"""
        self._source_fps = float(fps if fps and fps > 1.0 else 25.0)

    def _loiter_dwell_seconds(self, track_id: str, frame_id: int) -> float:
        first_fid = self.track_first_frame_id.get(track_id)
        if first_fid is None or frame_id <= first_fid:
            return 0.0
        return (frame_id - first_fid) / max(1.0, self._source_fps)

    def _is_loitering(self, track_id: str, frame_id: int = 0, bbox_diag: float = 0.0) -> bool:
        """S01 判定（视频时间口径，取代旧的固定处理帧窗口）。

        pacing（默认）: 时间窗内"来回走动"——三个条件：
          1) 活动范围受限：轨迹离质心最远距离 ≤ 半径（随人体尺寸自适应）；
          2) 有真实移动：最远距离 ≥ loiter_extent_ratio×人体对角线
             （站立检测抖动的轨迹展开通常 <0.03×对角线，被此条件排除）；
          3) 路径远长于净位移（来回走动而非直线穿过）。
        dwell（默认关）: 同处驻留 ≥ loiter_dwell_min_seconds。
        """
        entries = list(self.track_centers[track_id])
        if len(entries) < 2:
            return False
        if frame_id <= 0:
            frame_id = int(entries[-1][0])
        fps = max(1.0, self._source_fps)
        window_frames = max(2, int(round(self.loiter_window_seconds * fps)))

        recent = [(fid, c) for fid, c in entries if frame_id - fid <= window_frames]
        if len(recent) < 2:
            return False

        xs = [c[0] for _, c in recent]
        ys = [c[1] for _, c in recent]
        cx = self._mean(xs)
        cy = self._mean(ys)
        max_dist = max((((x - cx) ** 2 + (y - cy) ** 2) ** 0.5) for x, y in zip(xs, ys))
        path_length = 0.0
        for idx in range(1, len(recent)):
            px, py = recent[idx - 1][1]
            cx2, cy2 = recent[idx][1]
            path_length += ((cx2 - px) ** 2 + (cy2 - py) ** 2) ** 0.5
        displacement = ((recent[-1][1][0] - recent[0][1][0]) ** 2 + (recent[-1][1][1] - recent[0][1][1]) ** 2) ** 0.5

        span_seconds = (recent[-1][0] - recent[0][0]) / fps
        # 容差 1s：吸收帧步长粒度（窗口过滤为闭区间，span 可能略小于 window_seconds）
        if span_seconds < self.loiter_window_seconds - 1.0:
            return False

        diag = max(1.0, bbox_diag)
        radius = max(self.loiter_radius_px, self.loiter_radius_ratio * diag)
        extent_min = self.loiter_extent_ratio * diag

        within_radius = max_dist <= radius
        has_real_movement = max_dist >= extent_min

        # pacing：来回走动（活动范围受限 + 有真实移动 + 路径远长于净位移）
        if (
            within_radius
            and has_real_movement
            and displacement <= radius
            and path_length / max(1.0, displacement) >= self.loiter_path_ratio
        ):
            return True

        # dwell：长时滞留（默认关闭）
        if self.enable_dwell_loiter:
            if self._loiter_dwell_seconds(track_id, frame_id) >= self.loiter_dwell_min_seconds and within_radius:
                return True
        return False

    def _head_hand_near(self, keypoints: list[list[float]], bbox: list[float]) -> bool:
        if not keypoints:
            return False

        head_pts = self._pose_head_points(keypoints)
        if not head_pts:
            return False
        head_mean = self._point_mean(head_pts)
        if head_mean is None:
            return False
        head_x, head_y = head_mean

        hand_candidates = self._pose_hand_points(keypoints)

        if not hand_candidates:
            return False

        bbox_diag = self._bbox_diag(bbox)
        min_dist = min((((pt[0] - head_x) ** 2 + (pt[1] - head_y) ** 2) ** 0.5) for pt in hand_candidates)
        return (min_dist / bbox_diag) <= self.phone_hand_head_ratio

    def hand_mouth_distance(self, keypoints: list[list[float]], bbox: list[float]) -> float:
        """手-嘴最小距离 / bbox 对角线（复核选帧信号；无有效点返回大值）。"""
        if not keypoints:
            return 1e9
        hand_pts = self._pose_hand_points(keypoints)
        nose = self._valid_points(keypoints, [0])
        mouth_ref = nose[0][:2] if nose else None
        if mouth_ref is None:
            head_pts = self._pose_head_points(keypoints)
            if not head_pts:
                return 1e9
            mouth_ref = self._point_mean(head_pts)
        if not hand_pts or mouth_ref is None:
            return 1e9
        bbox_diag = max(1e-6, self._bbox_diag(bbox))
        mouth_x, mouth_y = mouth_ref
        return min(((pt[0] - mouth_x) ** 2 + (pt[1] - mouth_y) ** 2) ** 0.5 for pt in hand_pts) / bbox_diag

    def _mouth_hand_near(self, keypoints: list[list[float]], bbox: list[float]) -> bool:
        if not keypoints:
            return False
        hand_pts = self._pose_hand_points(keypoints)
        if not hand_pts:
            return False
        # 嘴参考点：优先鼻尖（idx 0，比头部 5 点均值更贴近嘴部）
        nose = self._valid_points(keypoints, [0])
        mouth_ref = nose[0][:2] if nose else None
        if mouth_ref is None:
            head_pts = self._pose_head_points(keypoints)
            if not head_pts:
                return False
            mouth_ref = self._point_mean(head_pts)
        if mouth_ref is None:
            return False
        bbox_diag = self._bbox_diag(bbox)
        mouth_x, mouth_y = mouth_ref
        min_dist = min((((pt[0] - mouth_x) ** 2 + (pt[1] - mouth_y) ** 2) ** 0.5) for pt in hand_pts)
        return (min_dist / bbox_diag) <= self.smoke_hand_mouth_ratio

    def _is_camera_tamper_like(self, bbox: list[float], frame_shape: tuple[int, int], keypoints: list[list[float]]) -> bool:
        frame_h, frame_w = frame_shape
        x, y, w, h = [float(v) for v in bbox]
        area_ratio = (max(1.0, w) * max(1.0, h)) / max(1.0, frame_w * frame_h)
        cx, cy = self._bbox_center_xy(bbox)
        nx = abs(cx - frame_w * 0.5) / max(1.0, frame_w)
        ny = abs(cy - frame_h * 0.5) / max(1.0, frame_h)
        near_center = (nx <= self.tamper_center_radius_ratio) and (ny <= self.tamper_center_radius_ratio)

        head_y = None
        head_pts = self._pose_head_points(keypoints)
        head_mean = self._point_mean(head_pts)
        if head_mean is not None:
            head_y = float(head_mean[1])

        hand_above_head = False
        if keypoints and head_y is not None:
            for idx in range(len(keypoints)):
                if len(keypoints[idx]) >= 2 and float(keypoints[idx][0]) != 0.0 and float(keypoints[idx][1]) < head_y:
                    hand_above_head = True
                    break

        return area_ratio >= self.tamper_bbox_area_ratio and near_center and (hand_above_head or y <= frame_h * 0.15)

    def _is_danger_like(self, track_id: str, bbox: list[float], frame_shape: tuple[int, int], keypoints: list[list[float]], motion_ratio: float) -> bool:
        frame_h, frame_w = frame_shape
        x, y, w, h = [float(v) for v in bbox]
        cx, cy = self._bbox_center_xy(bbox)
        edge_margin_x = frame_w * self.danger_edge_margin_ratio
        edge_margin_y = frame_h * self.danger_edge_margin_ratio
        near_edge = cx <= edge_margin_x or cx >= frame_w - edge_margin_x or \
                    cy <= edge_margin_y or cy >= frame_h - edge_margin_y
        if not near_edge:
            return False
        curr_aspect = w / max(1.0, h)
        flat_strong = curr_aspect >= max(1.15, self.fall_aspect_ratio + 0.45)

        torso_ratio = self._pose_torso_vertical_ratio(keypoints)
        head_mean = self._point_mean(self._pose_head_points(keypoints))
        head_high = head_mean is not None and float(head_mean[1]) <= frame_h * 0.52
        upright_pose = torso_ratio is not None and torso_ratio >= 1.05
        not_flat = curr_aspect <= max(0.92, self.fall_aspect_ratio + 0.10)
        if upright_pose and head_high and not_flat and curr_aspect > self.fall_narrow_aspect_ratio:
            return False
        if torso_ratio is not None and torso_ratio >= 1.6 and curr_aspect >= 0.85 and curr_aspect <= 1.15:
            return False

        static_fall_posture = curr_aspect >= max(0.95, self.fall_aspect_ratio + 0.35) and (torso_ratio is None or torso_ratio <= 1.45)
        compact_lie_posture = 0.55 <= curr_aspect <= self.fall_narrow_aspect_ratio and cy >= frame_h * 0.62 and (torso_ratio is None or torso_ratio <= 1.20)
        medium_lie_posture = self.fall_narrow_aspect_ratio < curr_aspect <= 1.05 and cy >= frame_h * 0.66 and (torso_ratio is None or torso_ratio <= 0.60)
        if compact_lie_posture or medium_lie_posture:
            return True
        narrow_lie_posture = curr_aspect <= self.fall_narrow_aspect_ratio and cy >= frame_h * 0.66
        narrow_torso_ok = torso_ratio is None or torso_ratio <= 3.2
        if narrow_lie_posture and narrow_torso_ok:
            return True
        if static_fall_posture and (not head_high or cy >= frame_h * 0.62):
            return True

        bboxes = list(self.track_bboxes[track_id])
        window = max(4, self.fall_min_frames)
        if len(bboxes) < max(3, window // 2):
            return False

        recent = bboxes[-window:]
        flat_count = 0
        recent_aspects: list[float] = []
        for box in recent:
            _, _, w, h = [float(v) for v in box]
            if h <= 1.0:
                continue
            ratio = w / h
            recent_aspects.append(ratio)
            if ratio >= max(0.95, self.fall_aspect_ratio + 0.20):
                flat_count += 1

        flat_recent = flat_count >= max(2, len(recent_aspects) // 2)
        still_like = motion_ratio <= self.fall_still_motion_ratio

        descending_count = 0
        if len(recent) >= 2:
            prev = recent[-2]
            prev_cx, prev_cy = self._bbox_center_xy(prev)
            _, curr_cy = self._bbox_center_xy(recent[-1])
            prev_w, prev_h = [float(v) for v in prev[2:4]]
            _, curr_h = [float(v) for v in recent[-1][2:4]]
            prev_diag = max(1.0, (prev_w * prev_w + prev_h * prev_h) ** 0.5)
            if (curr_cy - prev_cy) >= prev_diag * self.fall_drop_ratio or (curr_h / max(1.0, prev_h)) <= 0.8:
                descending_count = 1

        pose_horizontal = torso_ratio is not None and torso_ratio <= self.fall_pose_horizontal_ratio

        if pose_horizontal and (flat_recent or still_like):
            return True

        if flat_recent and (motion_ratio >= self.fall_motion_ratio or descending_count > 0):
            return True
        if flat_recent and still_like:
            return True
        if flat_strong and still_like:
            return True
        return False

    def _fall_posture_candidate(self, bbox: list[float], frame_shape: tuple[int, int], keypoints: list[list[float]]) -> bool:
        frame_h, _ = frame_shape
        x, y, w, h = [float(v) for v in bbox]
        cx, cy = self._bbox_center_xy(bbox)
        near_floor = (y + h) >= frame_h * (1.0 - self.fall_floor_margin_ratio) or cy >= frame_h * self.fall_center_low_ratio
        if not near_floor:
            return False
        aspect = w / max(1.0, h)
        if aspect > 0.92:
            return False
        torso_ratio = self._pose_torso_vertical_ratio(keypoints)
        head_mean = self._point_mean(self._pose_head_points(keypoints))
        head_high = head_mean is not None and float(head_mean[1]) <= frame_h * 0.52
        upright_pose = torso_ratio is not None and torso_ratio >= 1.05
        if upright_pose and head_high and aspect > self.fall_narrow_aspect_ratio:
            return False
        if 0.50 <= aspect <= self.fall_narrow_aspect_ratio and cy >= frame_h * 0.62:
            return True
        if aspect <= self.fall_narrow_aspect_ratio and cy >= frame_h * 0.66:
            return True
        return False

    def _push_fall_event(self, bbox: list[float]) -> None:
        cx, cy = self._bbox_center_xy(bbox)
        diag = self._bbox_diag(bbox)
        self.recent_fall_events.append(
            {
                "cx": float(cx),
                "cy": float(cy),
                "diag": float(diag),
                "expire_tick": float(self.reasoner_tick + self.fall_event_ttl_calls),
            }
        )

    def _match_recent_fall_event(self, bbox: list[float]) -> bool:
        cx, cy = self._bbox_center_xy(bbox)
        diag = self._bbox_diag(bbox)
        aspect = float(bbox[2]) / max(1.0, float(bbox[3]))
        if aspect > 0.92:
            return False
        for event in self.recent_fall_events:
            if float(event.get("expire_tick", 0.0)) < float(self.reasoner_tick):
                continue
            ex = float(event.get("cx", 0.0))
            ey = float(event.get("cy", 0.0))
            ed = max(1.0, float(event.get("diag", 1.0)))
            radius = max(ed, diag) * self.fall_event_radius_ratio
            if ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5 <= radius:
                return True
        return False

    @staticmethod
    def _risk_priority(label: str) -> int:
        if label.startswith("H"):
            return 3
        if label.startswith("M"):
            return 2
        if label.startswith("S"):
            return 1
        return 0

    def _pose_torso_vertical_ratio(self, keypoints: list[list[float]]) -> float:
        """计算躯干垂直比例。值接近 1 表示站立，接近 0 表示水平。"""
        shoulders = self._pose_shoulder_points(keypoints)
        hips = self._pose_hip_points(keypoints)
        if not shoulders or not hips:
            return 1.0
        shoulder_mean = self._point_mean(shoulders)
        hip_mean = self._point_mean(hips)
        if shoulder_mean is None or hip_mean is None:
            return 1.0
        dy = abs(shoulder_mean[1] - hip_mean[1])
        dx = abs(shoulder_mean[0] - hip_mean[0])
        return dy / max(1.0, dx + dy)

    def _is_fall_like(self, bbox: list[float], frame_shape: tuple[int, int], keypoints: list[list[float]]) -> bool:
        return self._fall_posture_candidate(bbox, frame_shape, keypoints)

    def _is_smoking_like(self, track_id: str, bbox: list[float], keypoints: list[list[float]]) -> bool:
        if self._mouth_hand_near(keypoints, bbox):
            self.track_smoke_counter[track_id] += 1
        else:
            self.track_smoke_counter[track_id] = max(0, self.track_smoke_counter[track_id] - 1)
        return self.track_smoke_counter[track_id] >= self.smoke_min_frames

    def _is_vandal_like(self, track_id: str, bbox: list[float], keypoints: list[list[float]], motion_ratio: float = 0.0, **kwargs) -> bool:
        if motion_ratio >= self.vandal_motion_ratio:
            self.track_vandal_counter[track_id] += 1
        else:
            self.track_vandal_counter[track_id] = max(0, self.track_vandal_counter[track_id] - 1)
        return self.track_vandal_counter[track_id] >= self.vandal_min_frames

    def _is_edge_impact_like(self, track_id: str, bbox: list[float], frame_shape: tuple[int, int], keypoints: list[list[float]], motion_ratio: float = 0.0, limb_activity: float = 0.0) -> bool:
        return self._is_danger_like(track_id, bbox, frame_shape, keypoints, motion_ratio)

    def infer(
        self,
        track_id: str,
        bbox: list[float],
        keypoints: list[list[float]],
        frame_shape: tuple[int, int],
        base_label: str,
        base_confidence: float,
        peer_tracks: list[dict[str, Any]] | None = None,
        frame_id: int = 0,
    ) -> tuple[str, float, str, list[str]]:
        self.reasoner_tick += 1
        while self.recent_fall_events and float(self.recent_fall_events[0].get("expire_tick", 0.0)) < float(self.reasoner_tick):
            self.recent_fall_events.popleft()

        # 模型输出的 S01 不可信（训练数据 S01 占比过高），重映射为 Normal；
        # S01 只能由本模块的轨迹规则产生。
        if str(base_label) == "S01":
            base_label = "Normal"
            base_confidence = min(float(base_confidence), 0.6)

        center = self._bbox_center_xy(bbox)
        self.track_h04_transition_cooldown[track_id] = max(0, int(self.track_h04_transition_cooldown.get(track_id, 0)) - 1)
        if track_id not in self.track_first_frame_id:
            self.track_first_frame_id[track_id] = int(frame_id)
        self.track_centers[track_id].append((int(frame_id), center))
        self.track_bboxes[track_id].append([float(v) for v in bbox])
        area = max(1.0, float(bbox[2]) * float(bbox[3]))
        prev_center = self.track_last_center.get(track_id)
        prev_area = self.track_last_area.get(track_id)
        self.track_last_center[track_id] = center
        self.track_last_area[track_id] = area

        motion_ratio = 0.0
        if prev_center is not None:
            motion_ratio = ((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2) ** 0.5 / self._bbox_diag(bbox)
        area_change_ratio = 0.0
        if prev_area is not None:
            area_change_ratio = abs(area - prev_area) / max(1.0, prev_area)

        rule_candidates: list[tuple[str, float, str]] = []

        if self._is_loitering(track_id, frame_id, self._bbox_diag(bbox)):
            dwell_seconds = self._loiter_dwell_seconds(track_id, frame_id)
            extra = min(0.18, dwell_seconds / max(1.0, self.loiter_min_seconds) * 0.18)
            rule_candidates.append(("S01", min(self.loiter_conf_cap, self.default_rule_conf + extra), "loitering_rule"))

        if self._is_intrusion(center, self.forbidden_zones):
            rule_candidates.append(("S02", max(0.8, self.default_rule_conf), "forbidden_zone_rule"))

        near_head = self._head_hand_near(keypoints, bbox)
        if near_head:
            self.track_phone_counter[track_id] += 1
        else:
            self.track_phone_counter[track_id] = 0
        if self.track_phone_counter[track_id] >= self.phone_min_frames:
            rule_candidates.append(("M02", max(0.72, self.default_rule_conf), "phone_posture_rule"))

        motion_level = float(motion_ratio + area_change_ratio * 0.5)
        self.track_motion_level[track_id] = motion_level

        if str(base_label) == "H02":
            self.track_base_h02_counter[track_id] += 1
        else:
            self.track_base_h02_counter[track_id] = max(0, self.track_base_h02_counter[track_id] - 1)
        if str(base_label) == "H03":
            self.track_base_h03_counter[track_id] += 1
        else:
            self.track_base_h03_counter[track_id] = max(0, self.track_base_h03_counter[track_id] - 1)

        close_peers = self._close_peer_ids(track_id, bbox, peer_tracks)
        peer_active = sum(1 for peer_id in close_peers if float(self.track_motion_level.get(peer_id, 0.0)) >= self.h02_motion_ratio)
        limb_activity = self._upper_limb_activity(keypoints, bbox)

        bully_like = len(close_peers) >= 1 and motion_level >= self.h02_motion_ratio and limb_activity >= 0.16
        fight_like = len(close_peers) >= 1 and peer_active >= 1 and motion_level >= self.h01_motion_ratio and limb_activity >= 0.18

        if bully_like:
            self.track_h02_counter[track_id] += 1
        else:
            self.track_h02_counter[track_id] = max(0, self.track_h02_counter[track_id] - 1)

        if fight_like:
            self.track_h01_counter[track_id] += 1
        else:
            self.track_h01_counter[track_id] = max(0, self.track_h01_counter[track_id] - 1)

        if self.track_h02_counter[track_id] >= self.h02_min_frames:
            rule_candidates.append(("H02", max(0.77, self.default_rule_conf), "bully_interaction_rule"))
            if motion_level >= self.h01_motion_ratio * 1.10 and limb_activity >= 0.12:
                rule_candidates.append(("H01", max(0.88, self.default_rule_conf), "fight_escalation_track_rule"))
        if (
            str(base_label) == "H02"
            and base_confidence >= self.h01_transition_min_base_conf
            and len(close_peers) >= 1
            and peer_active >= 1
            and motion_level >= (self.h01_motion_ratio * 0.95)
            and limb_activity >= 0.10
        ):
            h01_transition_conf = max(0.96, min(0.99, base_confidence + 0.08))
            rule_candidates.append(("H01", h01_transition_conf, "fight_from_base_h02_instant_rule"))
        if (
            str(base_label) == "H02"
            and base_confidence >= self.h01_transition_min_base_conf
            and self.track_base_h02_counter[track_id] >= 2
            and len(close_peers) >= 1
            and (peer_active >= 1 or limb_activity >= 0.12)
        ):
            h01_transition_conf = max(0.95, min(0.99, base_confidence + 0.07))
            rule_candidates.append(("H01", h01_transition_conf, "fight_from_base_h02_transition_rule"))
        if (
            str(base_label) == "H02"
            and base_confidence >= self.h01_transition_min_base_conf
            and self.track_base_h02_counter[track_id] >= 3
            and len(close_peers) >= 1
            and peer_active >= 1
            and motion_level >= self.h01_dual_active_ratio
        ):
            h01_transition_conf = max(0.97, min(0.99, base_confidence + 0.10))
            rule_candidates.append(("H01", h01_transition_conf, "fight_from_base_h02_sustain_rule"))
        if (
            str(base_label) == "H02"
            and base_confidence >= self.h01_transition_min_base_conf
            and self.track_base_h02_counter[track_id] >= 2
            and motion_level >= (self.h01_dual_active_ratio * 0.9)
            and limb_activity >= 0.08
        ):
            h01_single_track_conf = max(0.95, min(0.99, base_confidence + 0.09))
            rule_candidates.append(("H01", h01_single_track_conf, "fight_h02_single_track_followup_rule"))
        if str(base_label) == "H02" and self.track_base_h02_counter[track_id] >= 2:
            h01_persistence_conf = max(0.94, min(0.99, base_confidence + 0.06))
            rule_candidates.append(("H01", h01_persistence_conf, "fight_h02_persistence_rule"))
        if self.track_h02_counter[track_id] >= 2:
            h01_rule_h02_conf = max(0.93, min(0.99, base_confidence + 0.05))
            rule_candidates.append(("H01", h01_rule_h02_conf, "fight_from_rule_h02_persistence_rule"))
        if self.track_h01_counter[track_id] >= self.h01_min_frames:
            rule_candidates.append(("H01", max(0.79, self.default_rule_conf), "fight_interaction_rule"))
        elif self.track_base_h02_counter[track_id] >= 2 and motion_level >= self.h01_dual_active_ratio:
            rule_candidates.append(("H01", max(0.90, self.default_rule_conf), "fight_from_base_h02_rule"))

        for peer_id in close_peers:
            pair_key = self._pair_key(track_id, peer_id)
            peer_motion = float(self.track_motion_level.get(peer_id, 0.0))

            peer_bbox = None
            for peer in (peer_tracks or []):
                if str(peer.get("track_id", "")) == peer_id:
                    peer_bbox = list(peer.get("bbox", [0.0, 0.0, 0.0, 0.0]))
                    break

            iou_contact = 0.0
            distance_now = 0.0
            approaching = False
            if peer_bbox is not None:
                iou_contact = self._bbox_iou_xywh(bbox, peer_bbox)
                pcx, pcy = self._bbox_center_xy(peer_bbox)
                distance_now = ((center[0] - pcx) ** 2 + (center[1] - pcy) ** 2) ** 0.5
                prev_distance = self.pair_last_distance.get(pair_key)
                if prev_distance is not None:
                    approaching = distance_now < (prev_distance * 0.94)
                self.pair_last_distance[pair_key] = distance_now

            contact_like = iou_contact >= self.interaction_contact_iou or approaching
            if contact_like:
                self.pair_contact_counter[pair_key] += 1
            else:
                self.pair_contact_counter[pair_key] = max(0, self.pair_contact_counter[pair_key] - 1)

            dual_active_like = motion_level >= self.h01_dual_active_ratio and peer_motion >= self.h01_dual_active_ratio
            if dual_active_like:
                self.pair_dual_active_counter[pair_key] += 1
            else:
                self.pair_dual_active_counter[pair_key] = max(0, self.pair_dual_active_counter[pair_key] - 1)

            pair_h02_like = (
                motion_level >= self.h02_motion_ratio
                and (limb_activity >= 0.10 or contact_like)
            )
            pair_h01_like = (
                motion_level >= self.h01_motion_ratio
                and peer_motion >= self.h01_dual_active_ratio
                and (limb_activity >= 0.12 or self.pair_contact_counter[pair_key] >= 2)
            )

            if pair_h02_like:
                self.pair_h02_counter[pair_key] += 1
            else:
                self.pair_h02_counter[pair_key] = max(0, self.pair_h02_counter[pair_key] - 1)

            if pair_h01_like:
                self.pair_h01_counter[pair_key] += 1
            else:
                self.pair_h01_counter[pair_key] = max(0, self.pair_h01_counter[pair_key] - 1)

            pair_h01_transition_like = (
                str(base_label) == "H02"
                and base_confidence >= self.h01_transition_min_base_conf
                and self.track_base_h02_counter[track_id] >= 1
                and self.pair_contact_counter[pair_key] >= 1
                and self.pair_dual_active_counter[pair_key] >= 1
            )
            if pair_h01_transition_like:
                self.pair_h01_transition_counter[pair_key] += 1
            else:
                self.pair_h01_transition_counter[pair_key] = max(0, self.pair_h01_transition_counter[pair_key] - 1)

            if self.pair_h02_counter[pair_key] >= self.h02_pair_min_frames or (
                self.pair_contact_counter[pair_key] >= self.h02_pair_min_frames and motion_level >= self.h02_motion_ratio
            ):
                rule_candidates.append(("H02", max(0.80, self.default_rule_conf), "bully_pair_rule"))
            if self.pair_h01_counter[pair_key] >= self.h01_pair_min_frames or (
                self.pair_contact_counter[pair_key] >= self.h01_pair_min_frames
                and self.pair_dual_active_counter[pair_key] >= self.h01_pair_min_frames
            ):
                rule_candidates.append(("H01", max(0.82, self.default_rule_conf), "fight_pair_rule"))
            elif self.pair_h02_counter[pair_key] >= self.h02_pair_min_frames and self.pair_dual_active_counter[pair_key] >= 2:
                rule_candidates.append(("H01", max(0.80, self.default_rule_conf), "fight_escalation_rule"))
            elif (
                self.pair_h02_counter[pair_key] >= max(2, self.h02_pair_min_frames - 1)
                and self.pair_dual_active_counter[pair_key] >= 1
                and motion_level >= (self.h01_motion_ratio * 0.9)
                and (limb_activity >= 0.10 or self.pair_contact_counter[pair_key] >= 1)
            ):
                rule_candidates.append(("H01", max(0.79, self.default_rule_conf), "fight_transition_pair_rule"))
            if self.pair_h01_transition_counter[pair_key] >= self.h01_pair_transition_min_frames:
                h01_transition_conf = max(0.97, min(0.99, base_confidence + 0.10))
                rule_candidates.append(("H01", h01_transition_conf, "fight_pair_state_machine_rule"))
            if (
                self.track_base_h02_counter[track_id] >= 1
                and self.pair_contact_counter[pair_key] >= 1
                and (self.pair_dual_active_counter[pair_key] >= 1 or self.pair_h02_counter[pair_key] >= self.h02_pair_min_frames)
            ):
                h01_followup_conf = max(0.96, min(0.99, base_confidence + 0.12))
                rule_candidates.append(("H01", h01_followup_conf, "fight_pair_followup_rule"))

        bbox_aspect = float(bbox[2]) / max(1.0, float(bbox[3]))
        head_mean = self._point_mean(self._pose_head_points(keypoints))
        head_not_low = head_mean is None or float(head_mean[1]) <= frame_shape[0] * 0.65
        smoke_allowed = (float(bbox[3]) <= frame_shape[0] * 0.72) and (center[1] <= frame_shape[0] * 0.82) and (bbox_aspect <= 0.90) and head_not_low
        smoke_evidence = False
        if smoke_allowed and self._is_smoking_like(track_id, bbox, keypoints):
            smoke_evidence = True
            rule_candidates.append(("M01", max(0.74, self.default_rule_conf), "smoking_posture_rule"))

        tamper_like = self._is_camera_tamper_like(bbox, frame_shape, keypoints)
        if tamper_like:
            self.track_tamper_counter[track_id] += 1
        else:
            self.track_tamper_counter[track_id] = 0
        if self.track_tamper_counter[track_id] >= self.tamper_min_frames:
            rule_candidates.append(("M03", max(0.75, self.default_rule_conf), "camera_tamper_rule"))

        fall_like = self._is_fall_like(bbox, frame_shape, keypoints)
        fall_memory_match = False
        if fall_like:
            self.track_fall_counter[track_id] += 2
            self._push_fall_event(bbox)
        else:
            if self._fall_posture_candidate(bbox, frame_shape, keypoints) and self._match_recent_fall_event(bbox):
                fall_memory_match = True
                self.track_fall_counter[track_id] += 1
            else:
                self.track_fall_counter[track_id] = max(0, self.track_fall_counter[track_id] - 1)
        if fall_like or fall_memory_match:
            self.track_h03_counter[track_id] += 2
        else:
            self.track_h03_counter[track_id] = max(0, self.track_h03_counter[track_id] - 1)

        h03_continuity = self.track_h03_counter[track_id] >= self.h03_hold_frames
        if h03_continuity and (fall_like or fall_memory_match) and self._fall_posture_candidate(bbox, frame_shape, keypoints):
            rule_candidates.append(("H03", max(0.80, self.default_rule_conf), "fall_continuity_rule"))

        if (fall_like or fall_memory_match) and self.track_fall_counter[track_id] >= max(3, self.fall_min_frames // 2):
            rule_candidates.append(("H03", max(0.78, self.default_rule_conf), "fall_posture_rule"))

        base_label_adj = str(base_label)
        base_conf_adj = float(base_confidence)

        h03_active = fall_like or fall_memory_match or self.track_h03_counter[track_id] >= self.h03_hold_frames

        edge_impact_like = self._is_edge_impact_like(track_id, bbox, frame_shape, keypoints, motion_ratio=motion_level, limb_activity=limb_activity)
        if edge_impact_like:
            self.track_impact_counter[track_id] += 1
            self.track_h04_hold_counter[track_id] = self.h04_pre_fall_hold_frames
        else:
            self.track_impact_counter[track_id] = max(0, self.track_impact_counter[track_id] - 1)
            self.track_h04_hold_counter[track_id] = max(0, self.track_h04_hold_counter[track_id] - 1)

        if not h03_active and self.track_impact_counter[track_id] >= self.h04_impact_min_frames:
            rule_candidates.append(("H04", max(0.80, self.default_rule_conf), "impact_risk_rule"))
        elif self.track_h04_hold_counter[track_id] > 0 and self.track_impact_counter[track_id] >= 1 and self.track_h03_counter[track_id] <= max(3, self.h03_hold_frames):
            rule_candidates.append(("H04", max(0.78, self.default_rule_conf), "pre_fall_impact_rule"))

        frame_h, frame_w = frame_shape
        near_edge_now = center[0] <= frame_w * 0.24 or center[0] >= frame_w * 0.76 or center[1] <= frame_h * 0.42
        has_edge_impact_evidence = near_edge_now and (
            self.track_impact_counter[track_id] >= self.h04_transition_min_impact_counter
            or edge_impact_like
            or motion_level >= (self.h04_edge_impact_motion_ratio * 1.15)
        )
        if has_edge_impact_evidence:
            self.track_h03_risk_pre_counter[track_id] += 1
        else:
            self.track_h03_risk_pre_counter[track_id] = max(0, self.track_h03_risk_pre_counter[track_id] - 1)

        if fall_like or fall_memory_match:
            if self.track_h03_risk_pre_counter[track_id] >= self.h03_risk_pre_frames:
                self.track_h03_mode[track_id] = "risk_then_fall"
            else:
                self.track_h03_mode[track_id] = "direct_fall"
            self.track_h03_mode_hold_counter[track_id] = self.h03_mode_hold_frames
        else:
            self.track_h03_mode_hold_counter[track_id] = max(0, self.track_h03_mode_hold_counter[track_id] - 1)
            if self.track_h03_mode_hold_counter[track_id] <= 0:
                self.track_h03_mode[track_id] = ""

        risk_then_fall_mode = self.track_h03_mode[track_id] == "risk_then_fall"
        early_h03_transition = (
            str(base_label) == "H03"
            and base_confidence >= self.h04_transition_min_base_conf
            and self.track_base_h03_counter[track_id] <= 2
            and has_edge_impact_evidence
            and self.track_h04_transition_cooldown[track_id] <= 0
        )
        if early_h03_transition:
            self.track_h04_transition_counter[track_id] += 1
            self.track_h04_transition_cooldown[track_id] = self.h04_transition_cooldown_frames
        else:
            self.track_h04_transition_counter[track_id] = max(0, self.track_h04_transition_counter[track_id] - 1)

        transition_h04_active = self.track_h04_transition_counter[track_id] >= self.h04_transition_min_frames
        if (
            transition_h04_active
            and risk_then_fall_mode
            and has_edge_impact_evidence
            and self.track_h04_transition_counter[track_id] <= (self.h04_transition_min_frames + self.h04_transition_hold_frames)
        ):
            h04_transition_conf = max(0.96, min(0.99, base_confidence + 0.08))
            rule_candidates.append(("H04", h04_transition_conf, "h03_transition_state_machine_rule"))
        if (
            h03_active
            and risk_then_fall_mode
            and self.track_h03_counter[track_id] <= (self.h03_hold_frames + 2)
            and near_edge_now
            and motion_level >= self.h04_edge_impact_motion_ratio
            and (self.track_impact_counter[track_id] >= self.h04_impact_min_frames or transition_h04_active)
        ):
            rule_candidates.append(("H04", max(0.90, self.default_rule_conf), "h03_transition_risk_rule"))
        elif (
            risk_then_fall_mode
            and
            self.track_base_h03_counter[track_id] <= 3
            and str(base_label) == "H03"
            and near_edge_now
            and motion_level >= self.h04_edge_impact_motion_ratio
            and (self.track_impact_counter[track_id] >= self.h04_impact_min_frames or transition_h04_active)
        ):
            rule_candidates.append(("H04", max(0.90, self.default_rule_conf), "h04_from_base_h03_transition_rule"))

        if self._is_intrusion(center, self.danger_zones):
            rule_candidates.append(("H04", max(0.80, self.default_rule_conf), "danger_zone_rule"))
        elif not h03_active and self._is_danger_like(track_id, bbox, frame_shape, keypoints, motion_ratio=motion_level):
            self.track_danger_counter[track_id] += 1
            if self.track_danger_counter[track_id] >= max(4, self.fall_min_frames // 2):
                rule_candidates.append(("H04", max(0.79, self.default_rule_conf), "danger_edge_rule"))
        else:
            self.track_danger_counter[track_id] = max(0, self.track_danger_counter[track_id] - 1)

        if self._is_vandal_like(track_id, bbox, keypoints, motion_ratio=motion_ratio + area_change_ratio * 0.5):
            rule_candidates.append(("H05", max(0.76, self.default_rule_conf), "vandal_motion_rule"))

        best_label = base_label_adj
        best_conf = float(base_conf_adj)
        source = "action_model"
        tags: list[str] = []

        # S01 为位置类弱证据：不得覆盖任何具体行为（M02 手机站立 / M01 / H* / S02）
        if rule_candidates and str(base_label_adj) not in {"Normal", "S01"}:
            rule_candidates = [item for item in rule_candidates if item[0] != "S01"]

        if rule_candidates:
            rule_candidates.sort(key=lambda item: (self._risk_priority(item[0]), item[1]), reverse=True)
            rule_label, rule_conf, rule_tag = rule_candidates[0]
            tags = [item[2] for item in rule_candidates]
            rule_risk = self._risk_priority(rule_label)
            base_risk = self._risk_priority(base_label_adj)
            protected_h03 = base_label_adj == "H03" and (h03_active or self.track_base_h03_counter[track_id] >= 2)
            protected_m01 = base_label_adj == "M01" and (smoke_evidence or self.track_smoke_counter[track_id] >= max(2, self.smoke_min_frames // 2))
            can_override = False
            if rule_risk > base_risk:
                can_override = True
            elif rule_risk == base_risk and rule_conf > (best_conf + 0.03):
                can_override = True
            elif base_label_adj == "Normal" and rule_conf > best_conf:
                can_override = True

            if protected_h03 and rule_label != "H03":
                can_override = (
                    rule_label == "H04"
                    and risk_then_fall_mode
                    and has_edge_impact_evidence
                    and (self.track_impact_counter[track_id] >= self.h04_transition_min_impact_counter or transition_h04_active)
                    and rule_conf >= (best_conf + self.h04_transition_same_risk_margin)
                )
            if protected_m01 and rule_label != "M01":
                can_override = rule_risk >= 3 and rule_conf >= max(0.90, best_conf + 0.18)

            if can_override:
                best_label = rule_label
                best_conf = float(min(0.99, rule_conf))
                source = "rule_engine"

        return best_label, round(float(best_conf), 2), source, tags

    def refine_light(
        self,
        track_id: str,
        bbox: list[float],
        keypoints: list[list[float]],
        frame_shape: tuple[int, int],
        base_label: str,
        base_confidence: float,
        peer_tracks: list[dict[str, Any]] | None = None,
        frame_id: int = 0,
    ) -> tuple[str, float, str, list[str]]:
        """轻量级规则推理 —— 仅使用位置规则（S01 徘徊、S02 翻越围栏）。

        当 PoseClassifier 主路径激活时调用，PoseClassifier 已提供高质量标签，
        此处仅补充位置相关的规则覆盖。
        """
        # 模型输出的 S01 不可信（训练数据 S01 占比过高），重映射为 Normal；
        # S01 只能由本模块的轨迹规则产生。
        tags: list[str] = []
        if str(base_label) == "S01":
            base_label = "Normal"
            base_confidence = min(float(base_confidence), 0.6)
            tags.append("model_s01_remapped")

        # 更新轨迹历史
        center = self._bbox_center_xy(bbox)
        if track_id not in self.track_first_frame_id:
            self.track_first_frame_id[track_id] = int(frame_id)
        self.track_centers[track_id].append((int(frame_id), center))
        self.track_bboxes[track_id].append([float(v) for v in bbox])
        area = max(1.0, float(bbox[2]) * float(bbox[3]))
        self.track_last_center[track_id] = center
        self.track_last_area[track_id] = area

        # 收集规则候选（仅位置规则）
        rule_candidates: list[tuple[str, float, str]] = []

        if self._is_loitering(track_id, frame_id, self._bbox_diag(bbox)):
            dwell_seconds = self._loiter_dwell_seconds(track_id, frame_id)
            extra = min(0.18, dwell_seconds / max(1.0, self.loiter_min_seconds) * 0.18)
            rule_candidates.append(("S01", min(self.loiter_conf_cap, self.default_rule_conf + extra), "loitering_rule"))

        if self._is_intrusion(center, self.forbidden_zones):
            rule_candidates.append(("S02", max(0.8, self.default_rule_conf), "forbidden_zone_rule"))

        # ── 吸烟姿态规则（从 infer() 移植）：手近嘴持续 N 帧 → M01 候选 ──
        # 规则命中只给出中等置信度（0.72 落灰区），由 VLM 复核最终确认，
        # 避免把摸脸/揉眼睛等误判成吸烟
        if self._mouth_hand_near(keypoints, bbox):
            self.track_smoke_counter[track_id] += 1
        else:
            self.track_smoke_counter[track_id] = max(0, int(self.track_smoke_counter.get(track_id, 0)) - 1)
        if int(self.track_smoke_counter.get(track_id, 0)) >= self.smoke_min_frames:
            rule_candidates.append(("M01", 0.72, "smoking_posture_rule"))

        # ── M02 手机姿态规则（从 infer() 移植） ──
        if self._head_hand_near(keypoints, bbox):
            self.track_phone_counter[track_id] = int(self.track_phone_counter.get(track_id, 0)) + 1
        else:
            self.track_phone_counter[track_id] = max(0, int(self.track_phone_counter.get(track_id, 0)) - 1)
        if int(self.track_phone_counter.get(track_id, 0)) >= self.phone_min_frames:
            rule_candidates.append(("M02", max(0.72, self.default_rule_conf), "phone_posture_rule"))

        # 决策：PoseClassifier 标签为基础，仅当规则风险更高时覆盖
        best_label = base_label
        best_conf = float(base_confidence)
        source = "action_model"

        # S01 为位置类弱证据：不得覆盖任何具体行为（M02 手机站立 / M01 / H* / S02）
        if rule_candidates and str(base_label) not in {"Normal", "S01"}:
            rule_candidates = [item for item in rule_candidates if item[0] != "S01"]

        if rule_candidates:
            rule_candidates.sort(key=lambda item: (self._risk_priority(item[0]), item[1]), reverse=True)
            rule_label, rule_conf, rule_tag = rule_candidates[0]
            tags = [item[2] for item in rule_candidates] + tags
            rule_risk = self._risk_priority(rule_label)
            base_risk = self._risk_priority(base_label)
            can_override = False
            if rule_risk > base_risk:
                can_override = True
            elif rule_risk == base_risk and rule_conf > (best_conf + 0.03):
                can_override = True
            elif base_label == "Normal" and rule_conf > best_conf:
                can_override = True

            if can_override:
                best_label = rule_label
                best_conf = float(min(0.99, rule_conf))
                source = "rule_engine"

        return best_label, round(float(best_conf), 2), source, tags

    def clear_stale_tracks(self, active_track_ids: set[str]) -> None:
        for table in [
            self.track_centers,
            self.track_bboxes,
            self.track_last_center,
            self.track_last_area,
            self.track_phone_counter,
            self.track_tamper_counter,
            self.track_fall_counter,
            self.track_h03_counter,
            self.track_h02_counter,
            self.track_h01_counter,
            self.track_base_h02_counter,
            self.track_base_h03_counter,
            self.track_motion_level,
            self.track_impact_counter,
            self.track_h04_hold_counter,
            self.track_h04_transition_counter,
            self.track_h04_transition_cooldown,
            self.track_h03_risk_pre_counter,
            self.track_h03_mode_hold_counter,
            self.track_h03_mode,
            self.track_smoke_counter,
            self.track_vandal_counter,
            self.track_danger_counter,
        ]:
            stale = [track_id for track_id in table if track_id not in active_track_ids]
            for track_id in stale:
                table.pop(track_id, None)

        stale_pairs_h02 = [key for key in self.pair_h02_counter if any(part not in active_track_ids for part in key.split("|"))]
        for key in stale_pairs_h02:
            self.pair_h02_counter.pop(key, None)

        stale_pairs_h01 = [key for key in self.pair_h01_counter if any(part not in active_track_ids for part in key.split("|"))]
        for key in stale_pairs_h01:
            self.pair_h01_counter.pop(key, None)

        stale_pairs_h01_transition = [key for key in self.pair_h01_transition_counter if any(part not in active_track_ids for part in key.split("|"))]
        for key in stale_pairs_h01_transition:
            self.pair_h01_transition_counter.pop(key, None)

        stale_pairs_contact = [key for key in self.pair_contact_counter if any(part not in active_track_ids for part in key.split("|"))]
        for key in stale_pairs_contact:
            self.pair_contact_counter.pop(key, None)

        stale_pairs_dual = [key for key in self.pair_dual_active_counter if any(part not in active_track_ids for part in key.split("|"))]
        for key in stale_pairs_dual:
            self.pair_dual_active_counter.pop(key, None)

        stale_pairs_dist = [key for key in self.pair_last_distance if any(part not in active_track_ids for part in key.split("|"))]
        for key in stale_pairs_dist:
            self.pair_last_distance.pop(key, None)

        stale_first_frames = [key for key in self.track_first_frame_id if key not in active_track_ids]
        for key in stale_first_frames:
            self.track_first_frame_id.pop(key, None)
