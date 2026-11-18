from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np

from src.core.action_recognizer import ActionRecognizer
from src.core.alert_adjudicator import AlertAdjudicator
from src.core.behavior_reasoner import BehaviorReasoner
from src.core.detector import Detector
from src.core.interaction_analyzer import InteractionAnalyzer
from src.core.nl_rule_engine import NLAlertRuleEngine
from src.core.vlm_cache import VLMCropCache
from src.core.pose_estimator import PoseEstimator
from src.core.scene_narrator import SceneNarrator
from src.core.temporal_adjudicator import TemporalAdjudicator
from src.core.tracker import Tracker
from src.core.vlm_reviewer import VLMReviewer
from src.pipeline.alert_system import AlertSystem
from src.pipeline.frame_buffer import FrameRingBuffer
from src.pipeline.vlm_router import VLMRouter
from src.utils.schemas import Behavior, Detection, FrameResult, VLMReview

try:  # edge 决策模式可选依赖（edge_migration 包缺失时自动回退 server 行为）
    from edge_migration.crop_classifier import CropClassifier
    from edge_migration.decision_trace import DecisionTrace
    from edge_migration.demo_pack import DemoPack
    from edge_migration.mqtt_bridge import MqttBridge
    from edge_migration.track_narrator import TrackNarrator
    _EDGE_AVAILABLE = True
except Exception:
    _EDGE_AVAILABLE = False


RISK_MAP = {
    "Normal": "LOW",
    "H01": "HIGH",
    "H02": "HIGH",
    "H03": "HIGH",
    "H04": "HIGH",
    "H05": "HIGH",
    "M01": "MEDIUM",
    "M02": "MEDIUM",
    "M03": "MEDIUM",
    "S01": "MEDIUM",
    "S02": "MEDIUM",
}


class InferenceEngine:
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

    def __init__(self, model_cfg: dict, vlm_cfg: dict, output_dir: str = "results"):
        runtime_cfg = dict(model_cfg.get("runtime", {}))
        detector_cfg = dict(model_cfg.get("detector", {}))
        detector_cfg["runtime"] = runtime_cfg
        pose_cfg = dict(model_cfg.get("pose", {}))
        pose_cfg["runtime"] = runtime_cfg
        action_cfg = dict(model_cfg.get("action", {}))
        # 仅 action 模型走 CPU 推理，规避 cuDNN 在 CUDA 13.0 下的 Conv+bias 兼容性问题
        action_runtime = dict(runtime_cfg)
        action_runtime["execution_provider"] = "CPUExecutionProvider"
        action_cfg["runtime"] = action_runtime

        self.detector = Detector(detector_cfg)
        self.tracker = Tracker(model_cfg.get("tracker", {}))
        self.pose_estimator = PoseEstimator(pose_cfg)
        self.action_recognizer = ActionRecognizer(action_cfg)
        inference_cfg = model_cfg.get("inference", {})
        self.pose_stride = int(max(1, inference_cfg.get("pose_stride", 8)))
        self.max_pose_tracks = int(max(1, inference_cfg.get("max_pose_tracks", 2)))
        self.action_stride = int(max(1, inference_cfg.get("action_stride", 6)))
        self.max_action_tracks = int(max(1, inference_cfg.get("max_action_tracks", 1)))
        self.action_min_track_score = float(max(0.0, inference_cfg.get("action_min_track_score", 0.0)))
        self.action_warmup_frames = int(max(0, inference_cfg.get("action_warmup_frames", 0)))
        self.action_min_track_age = int(max(1, inference_cfg.get("action_min_track_age", 1)))
        # 方案 A：快速几何触发（不依赖 32 帧 PC 缓冲），立即发 pending 告警
        self._fast_trigger_enabled = bool(inference_cfg.get("fast_trigger_enabled", True))
        self._fast_trigger_aspect = float(inference_cfg.get("fast_trigger_aspect", 0.95))
        self._fast_trigger_max_height_ratio = float(inference_cfg.get("fast_trigger_max_height_ratio", 0.40))
        self._fast_trigger_torso_horizontal = float(inference_cfg.get("fast_trigger_torso_horizontal", 0.60))
        self._fast_trigger_drop_ratio = float(inference_cfg.get("fast_trigger_drop_ratio", 0.08))
        self._fast_trigger_cooldown_seconds = float(inference_cfg.get("fast_trigger_cooldown_seconds", 5.0))
        self._fast_trigger_last: dict[str, int] = {}
        self._fast_trigger_hits: dict[str, int] = {}
        self._fast_trigger_prev_bbox: dict[str, tuple[int, list[float]]] = {}
        self._fast_confirm_geometry_seconds = float(inference_cfg.get("fast_confirm_geometry_seconds", 1.5))
        self._fast_reject_geometry_seconds = float(inference_cfg.get("fast_reject_geometry_seconds", 1.0))
        self._pending_alerts: dict[str, dict] = {}
        self._pending_alert_updates: deque = deque()
        self._action_pc_last_result: dict = {}
        self.action_default_label = str(inference_cfg.get("action_default_label", "Normal"))
        self.action_default_confidence = float(min(1.0, max(0.0, inference_cfg.get("action_default_confidence", 0.60))))
        self.pose_motion_gate_iou_threshold = float(inference_cfg.get("pose_motion_gate_iou_threshold", 1.01))
        self.enable_pose_motion_gate = bool(inference_cfg.get("enable_pose_motion_gate", False))
        self.pose_force_refresh_gap = int(max(1, inference_cfg.get("pose_force_refresh_gap", max(1, self.pose_stride * 2))))
        self.enable_detector_gating = bool(inference_cfg.get("enable_detector_gating", False))
        self.detector_stride = int(max(1, inference_cfg.get("detector_stride", 1)))
        self.detector_max_skip_frames = int(max(self.detector_stride, inference_cfg.get("detector_max_skip_frames", self.detector_stride * 2)))
        self.detector_motion_threshold = float(max(0.0, inference_cfg.get("detector_motion_threshold", 0.012)))
        self.detector_low_motion_threshold = float(max(0.0, inference_cfg.get("detector_low_motion_threshold", self.detector_motion_threshold * 0.5)))
        self.detector_warmup_frames = int(max(0, inference_cfg.get("detector_warmup_frames", 1)))
        self.behavior_reasoner = BehaviorReasoner(model_cfg.get("behavior_heuristics", {}))
        self.temporal_adjudicator = TemporalAdjudicator(model_cfg.get("temporal_adjudicator", {}))
        self.pose_cache: dict[str, list[list[float]]] = {}
        self.pose_last_bbox: dict[str, list[float]] = {}
        self.pose_last_frame: dict[str, int] = {}
        self.action_cache: dict[str, tuple[str, float]] = {}
        self.track_seen_frames: dict[str, int] = {}
        self.interaction_analyzer = InteractionAnalyzer()
        self.vlm_reviewer = VLMReviewer(vlm_cfg)
        self.vlm_router = VLMRouter(vlm_cfg)
        alert_policy_cfg = dict(model_cfg.get("alert_policy", {})) if isinstance(model_cfg, dict) else {}
        if not alert_policy_cfg:
            alert_policy_cfg = {
                "min_consecutive_frames": inference_cfg.get("min_consecutive_frames", 3),
                "cooldown_seconds": inference_cfg.get("cooldown_seconds", 5),
                "high_risk_min_confidence": inference_cfg.get("high_risk_min_confidence", 0.55),
            }
        self.alert_system = AlertSystem(output_dir, config=alert_policy_cfg)
        self._last_detector_gray: np.ndarray | None = None
        self._detector_skip_count = 0

        # ── B2: ORT 模型 warmup（消除首帧 5-10x 慢启动） ──
        self._warmup_models()

        # ── C1: ID 切换检测（每个 track 记录上一帧 bbox，跳变时清零缓存） ──
        self._track_prev_bbox: dict[str, list[float]] = {}
        self._id_switch_distance_ratio = 2.0  # 质心位移/bbox对角线 > 此值判定为切换
        self._id_switch_area_ratio = 3.0     # bbox面积突变 > 此值判定为切换

        # ── B1: VLM 异步复核线程池 + 缓存锁 ──
        self._vlm_async_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm-async")
        self._vlm_cache = VLMCropCache(max_size=100, ttl_seconds=30.0)
        self._cache_lock = threading.Lock()

        # ── B6: 三阶段流水线（detector/tracker → pose → action/reasoner/post） ──
        _pipeline_cfg = dict((model_cfg or {}).get("inference", {}).get("pipeline", {}))
        self._pipeline_enabled = bool(_pipeline_cfg.get("enabled", False))
        self._pipe_queue_size = max(2, int(_pipeline_cfg.get("queue_size", 4)))
        self._pipe_generation = 0
        self._pipe_dropped = 0
        self._pipe_stage_ms = {1: 0.0, 2: 0.0, 3: 0.0}
        self._pipe_stage_max = {1: 0.0, 2: 0.0, 3: 0.0}
        self._pipe_stage_count = {1: 0, 2: 0, 3: 0}
        self._pipe_stage_samples = {1: deque(maxlen=240), 2: deque(maxlen=240), 3: deque(maxlen=240)}
        self._assemble_samples: deque = deque(maxlen=240)
        self._stage_action_samples: deque = deque(maxlen=240)
        self._stage_post_samples: deque = deque(maxlen=240)
        self._pose_batch_runs = 0
        # 异步 PoseClassifier worker：单线程，避免 ORT session 并发；只算 PC 批次，
        # 三阶段决策/crop 复核仍留在 stage3 线程。
        self._action_pc_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="action-pc")
        self._action_pc_lock = threading.Lock()
        self._action_pc_result: dict = {}
        self._action_pc_inflight = False
        self._action_pc_async_samples: deque = deque(maxlen=60)
        self._pipe_results: deque = deque()
        self._pipe_results_lock = threading.Lock()
        self._last_pipeline_error = ""
        if self._pipeline_enabled:
            from queue import Queue
            self._pipe_q1 = Queue(maxsize=self._pipe_queue_size)
            self._pipe_q2 = Queue(maxsize=self._pipe_queue_size)
            self._pipe_q3 = Queue(maxsize=self._pipe_queue_size)
            for _stage_no in (1, 2, 3):
                threading.Thread(
                    target=self._pipeline_worker, args=(_stage_no,),
                    daemon=True, name=f"pipe-stage{_stage_no}",
                ).start()

        # ── 场景理解与告警裁决（均为异步旁路，不阻塞推理流水线） ──
        vlm_cfg = vlm_cfg if isinstance(vlm_cfg, dict) else {}
        fb_cfg = dict(vlm_cfg.get("frame_buffer", {}))
        self.frame_buffer = FrameRingBuffer(
            max_seconds=float(fb_cfg.get("max_seconds", 30.0)),
            samples_per_second=float(fb_cfg.get("samples_per_second", 2.0)),
            target_width=int(fb_cfg.get("target_width", 640)),
        )
        self.scene_narrator = SceneNarrator(vlm_cfg, output_dir=output_dir)
        self.scene_narrator.attach_vlm_router(self.vlm_router)  # B3
        self.scene_narrator.attach_buffer(self.frame_buffer)
        self.alert_adjudicator = AlertAdjudicator(vlm_cfg, output_dir=output_dir)
        self.alert_adjudicator.attach_vlm_router(self.vlm_router)  # B3
        self.alert_adjudicator.attach_buffer(self.frame_buffer)
        self.alert_adjudicator.attach_narrator(self.scene_narrator)
        self.nl_rule_engine = NLAlertRuleEngine(vlm_cfg, self.frame_buffer, output_dir=output_dir)
        # ROADMAP Stage 1/2：新行为备忘 + 行为原型库（旁路，事件级触发；缓冲在下方创建后再 attach）
        from src.core.novel_behavior_memo import NovelBehaviorMemo
        from src.core.behavior_prototypes import BehaviorPrototypes
        self.novel_behavior_memo = NovelBehaviorMemo(vlm_cfg, output_dir=output_dir)
        # 原型是跨任务累积的全局知识，固定存 results/prototypes/，不随任务 output_dir 隔离
        self.behavior_prototypes = BehaviorPrototypes(vlm_cfg, output_dir="results")
        self.novel_behavior_memo.attach_prototypes(self.behavior_prototypes)
        self.nl_rule_engine.attach_vlm_router(self.vlm_router)  # B3
        # ── T3.x 语言层：点谁说谁 + 场景问答（文本 LLM，facts 为唯一信息源）──
        self.track_narrator = TrackNarrator(vlm_cfg)
        self.last_track_snapshot: list[dict] = []  # 最近处理帧逐目标终态（narrate 输入）
        self._last_frame_id = 0
        # ── T3.4 MQTT：遥测上行 / rule/set 下行（broker 缺失时 dry-run 优雅降级）──
        self.mqtt_bridge = MqttBridge(dict(model_cfg.get("mqtt", {})),
                                      on_rule_set=self._on_mqtt_rule_set)
        self.mqtt_bridge.register_status_provider(self._mqtt_status_payload)
        self.mqtt_bridge.start()
        # ── T6.5 演示剧本开关（连排/拍摄日防翻车：真实路径的降级入口，非假数据）──
        self.demo_pack = DemoPack(dict(model_cfg.get("demo", {})))
        # ── 存储治理（§4.5）：results/ 截图/事件轮转 + 磁盘水位（边缘盘有限）──
        try:
            from edge_migration.storage_gc import StorageGcLoop
            self._storage_gc = StorageGcLoop(output_dir, dict(model_cfg.get("storage_gc", {})))
            self._storage_gc.start()
        except Exception:
            self._storage_gc = None
        self.track_label_history: dict[str, deque] = {}
        # 诊断计数：裁决队列满被拒的告警数（WEAKNESS_AUDIT #1 修复后仍可能有界丢弃）
        self._adjudicate_submit_dropped = 0
        # 强制 VLM 复核（如 H03 跌倒，与坐姿/半躺易混淆）：越过三级决策直接复核，逐目标节流
        fvr_cfg = dict(vlm_cfg.get("force_vlm_review", {}))
        self.force_vlm_classes = {str(c) for c in fvr_cfg.get("classes", []) if str(c)}
        self.force_vlm_throttle_frames = int(max(1, fvr_cfg.get("throttle_frames", 25)))
        self.track_vlm_review_cache: dict[str, tuple[int, str, float]] = {}
        # ── edge 决策模式（EDGE v3.0 §4 / edge_migration/01 §2.4）──────────
        # crop 多分类仲裁替代 VLM 复核：同步毫秒级，复用同一套缓存/否决/确认信任层
        self.decision_mode = str(inference_cfg.get("decision_mode", "server"))
        self._slowfast_enabled = bool(inference_cfg.get("slowfast_enabled", True))
        self._edge_mode = False
        self.crop_classifier = None
        if self.decision_mode == "edge":
            if _EDGE_AVAILABLE:
                self._edge_mode = True
                self.crop_classifier = CropClassifier(dict(vlm_cfg.get("crop_review", {})))
            else:
                self.decision_mode = "server"  # 优雅降级：包缺失 → server 行为
        self._frame_traces: dict[str, "DecisionTrace"] = {}  # 时刻④：本帧各目标决策链
        # C2: 时序标签平滑参数
        self._smooth_window = 5
        # A: 检测事实注入——近窗异常寄存器（track → 最近一次异常判定），叙述注入按窗口聚合，抗标签瞬时波动
        self._recent_abnormal: dict[str, dict] = {}
        # 归因诊断：action_cache 中由 VLM 异步回写产生的 track（用于区分 cache 沿用的来源）
        self._vlm_cache_sourced: set[str] = set()
        # 归因修复：VLM 否决锁存（track_id -> {label: until_frame}），规则被复核否决后压制约15s
        self._vlm_veto: dict[str, dict[str, int]] = {}
        self._vlm_veto_seconds = 15.0
        self._source_fps = 25.0
        # 双钥匙 A：M01 的 VLM 确认寄存器（track -> until_frame），与规则候选缺一不可
        self._vlm_confirmed: dict[str, int] = {}
        self._vlm_confirm_hold_seconds = 12.0
        # 问题1：原帧复核通道——环形缓冲存 960 降采样帧（叙述/裁决用），
        # 本缓冲直存原始分辨率帧（person-crop 复核专用），真实细节翻倍。
        # 复核只需取近帧裁剪，独立收口（默认 48 帧 ≈ 300MB@1080p）：
        # 旧实现继承 30s×10fps=300 帧，1080p 下常驻约 1.9GB（WEAKNESS_AUDIT #9）
        _fbo_cfg = dict(vlm_cfg.get("frame_buffer_orig", {})) if isinstance(vlm_cfg, dict) else {}
        self.frame_buffer_orig = FrameRingBuffer(
            max_seconds=float(_fbo_cfg.get("max_seconds", 12.0)),
            samples_per_second=float(_fbo_cfg.get("samples_per_second", 2.0)),
            target_width=0,
            max_frames=int(_fbo_cfg.get("max_frames", 48)),
        )
        # Stage 1/2 缓冲接线（frame_buffer_orig 创建后）
        self.novel_behavior_memo.attach_vlm_router(self.vlm_router)
        self.novel_behavior_memo.attach_buffer(self.frame_buffer)
        self.novel_behavior_memo.attach_buffer_orig(self.frame_buffer_orig)
        self.novel_behavior_memo.attach_pose_buffers(self.action_recognizer.track_pose_buffers)
        # 复核判定平滑：同一 track 最近 3 次复核多数投票，防单次抖动翻转确认/否决状态
        self._review_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=3))
        self._veto_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=2))
        # 确认/否决的空间锚定：ID 切换频繁场景（P007→P009→P008 连环重编号）下，
        # track_id 键的确认窗口随 ID 丢失——附加质心网格键（80px）作空间备份
        self._vlm_confirmed_spatial: dict[tuple, int] = {}
        self._vlm_veto_spatial: dict[tuple, dict[str, int]] = {}
        self._spatial_grid = 80.0
        self._smooth_min_votes = 3
        self.track_smooth_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=self._smooth_window))
        # 消融开关：关闭后灰区复核不注入场景上下文（评估用）
        self.review_context_enabled = True

    def begin_task(self, task_id: str) -> None:
        """新任务开始：重置所有按帧号调度的旁路模块。

        帧号在任务间归零重来，环形缓冲/叙述调度/裁决等待/规则引擎调度
        若跨任务残留，会导致新任务前半段无叙述、开放规则不触发等问题。
        """
        self.frame_buffer.clear()
        self.frame_buffer_orig.clear()
        self.scene_narrator.begin_task(task_id)
        self.alert_adjudicator.reset_for_task()
        self.nl_rule_engine.reset_scheduling()
        self.novel_behavior_memo.begin_task(task_id)
        # C2/B5 状态同样按任务隔离：上一任务的平滑历史与 crop 缓存不得泄漏
        self.track_smooth_history.clear()
        self._vlm_cache.clear()
        self._adjudicate_submit_dropped = 0
        self._recent_abnormal.clear()
        self._vlm_cache_sourced.clear()
        self._vlm_veto.clear()
        self._vlm_confirmed.clear()
        self._review_history.clear()
        self._veto_history.clear()
        self._vlm_confirmed_spatial.clear()
        self._vlm_veto_spatial.clear()
        # 任务级缓存隔离：ByteTrack track_id 会从 1 重新分配，不清理会把上一任务的
        # action/pose/label 缓存套到新目标上，导致 action_runs 恒 0、行为标签沿用旧任务。
        self.action_cache.clear()
        self.pose_cache.clear()
        self.pose_last_bbox.clear()
        self.pose_last_frame.clear()
        self.track_label_history.clear()
        self.track_smooth_history.clear()
        self.track_seen_frames.clear()
        self._track_prev_bbox.clear()
        self.track_vlm_review_cache.clear()
        self.action_recognizer.clear_stale_tracks(set())
        self.behavior_reasoner.clear_stale_tracks(set())
        # B6: 管道状态隔离——代际 +1 使各阶段丢弃旧任务残留帧，并清空队列/结果
        self._pipe_generation += 1
        # 每次任务重置 pipeline 计时/丢弃计数，避免跨任务累计污染诊断
        self._pipe_dropped = 0
        self._pipe_stage_ms = {1: 0.0, 2: 0.0, 3: 0.0}
        self._pipe_stage_max = {1: 0.0, 2: 0.0, 3: 0.0}
        self._pipe_stage_count = {1: 0, 2: 0, 3: 0}
        self._pipe_stage_samples = {1: deque(maxlen=240), 2: deque(maxlen=240), 3: deque(maxlen=240)}
        self._assemble_samples = deque(maxlen=240)
        self._stage_action_samples = deque(maxlen=240)
        self._stage_post_samples = deque(maxlen=240)
        self._action_pc_result = {}
        self._action_pc_async_samples = deque(maxlen=60)
        self._fast_trigger_last.clear()
        self._fast_trigger_hits.clear()
        self._fast_trigger_prev_bbox.clear()
        self._pending_alerts.clear()
        self._pending_alert_updates.clear()
        self._action_pc_last_result = {}
        self._pose_batch_runs = 0
        self._last_pipeline_error = ""
        for _q_name in ("_pipe_q1", "_pipe_q2", "_pipe_q3"):
            _q = getattr(self, _q_name, None)
            if _q is not None:
                try:
                    while True:
                        _q.get_nowait()
                except Exception:
                    pass
        with self._pipe_results_lock:
            self._pipe_results.clear()

    def set_source_fps(self, fps: float) -> None:
        """由任务层注入视频源帧率，驱动叙述/裁决的视频时间计算。"""
        self._source_fps = float(fps if fps and fps > 1.0 else 25.0)
        self.frame_buffer.set_source_fps(fps)
        self.scene_narrator.set_source_fps(fps)
        self.alert_adjudicator.set_source_fps(fps)
        self.behavior_reasoner.set_source_fps(fps)
        self.nl_rule_engine.set_source_fps(fps)
        self.novel_behavior_memo.set_source_fps(fps)

    def close(self) -> None:
        self.scene_narrator.stop()
        self.alert_adjudicator.stop()
        self.nl_rule_engine.stop()
        self.novel_behavior_memo.stop()
        self._vlm_async_executor.shutdown(wait=False, cancel_futures=True)
        try:
            self._action_pc_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        # 释放 NPU 设备（OM 后端 unload→finalize），否则档切换重建时新 OM 加载失败回退 CPU ONNX
        self.detector.close()

    # ── B2: ORT warmup ───────────────────────────────────────
    def _warmup_models(self) -> None:
        """启动时各模型跑一次 dummy 推理，触发 ONNX graph 优化与内存分配预热。"""
        try:
            dummy = np.zeros((360, 640, 3), dtype=np.uint8)
            if self.detector.is_real_model_loaded:
                self.detector.infer(dummy)
            if self.pose_estimator.is_real_model_loaded:
                fake_track = {"track_id": "warmup", "bbox": [100, 50, 200, 250], "score": 0.9}
                self.pose_estimator.infer_batch(dummy, [fake_track])
            # PoseClassifier warmup：填充 32 帧零姿态
            if self.action_recognizer.pose_classifier_loaded:
                zero_pose = np.zeros((133, 3), dtype=np.float32)
                for _ in range(self.action_recognizer.input_clip_len):
                    self.action_recognizer.append_pose("warmup", zero_pose.tolist())
                self.action_recognizer.infer_pose_classifier_batch(["warmup"])
                self.action_recognizer.clear_stale_tracks(set())  # 清理 warmup track
        except Exception as exc:
            print(f"[warmup] skipped: {exc}")

    # ── C1: ID 切换检测 + 缓冲清零 ──────────────────────────
    def _clear_track_caches(self, track_id: str) -> None:
        """ID 切换时清零该 track 的所有逐目标缓存，防止混人序列污染。"""
        self.action_recognizer.clear_stale_tracks({track_id})
        self.behavior_reasoner.clear_stale_tracks({track_id})
        with self._cache_lock:
            self.track_vlm_review_cache.pop(track_id, None)
            self.track_label_history.pop(track_id, None)

    def _detect_and_handle_id_switches(self, tracks: list[dict]) -> None:
        """检测 bbox 跳变（ID 切换），清零被污染的逐目标缓存。"""
        for track in tracks:
            track_id = str(track["track_id"])
            bbox = track.get("bbox", [0, 0, 0, 0])
            prev_bbox = self._track_prev_bbox.get(track_id)
            if prev_bbox is not None and len(prev_bbox) == 4 and len(bbox) == 4:
                # 质心位移
                cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
                px, py = prev_bbox[0] + prev_bbox[2] / 2, prev_bbox[1] + prev_bbox[3] / 2
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                diag = max(1.0, (bbox[2] ** 2 + bbox[3] ** 2) ** 0.5)
                ratio = dist / diag
                # 面积突变
                area = max(1.0, bbox[2] * bbox[3])
                prev_area = max(1.0, prev_bbox[2] * prev_bbox[3])
                area_ratio = max(area / prev_area, prev_area / area)
                if ratio > self._id_switch_distance_ratio or area_ratio > self._id_switch_area_ratio:
                    self._clear_track_caches(track_id)
            self._track_prev_bbox[track_id] = list(bbox)
        # 清理消失的 track + 确认窗口继承（同位置新 ID 延续 VLM 确认，防 ID 切换丢钥匙）
        active_ids = {str(t["track_id"]) for t in tracks}
        stale = [tid for tid in self._track_prev_bbox if tid not in active_ids]
        for tid in stale:
            self._track_prev_bbox.pop(tid, None)
        for tid in stale:
            conf_until = self._vlm_confirmed.pop(tid, None)
            if conf_until is None:
                continue
            # 找与消失 track 上一帧位置最近的现存 track，继承确认窗口
            prev_bbox = None  # 已 pop，从动作缓存尽力而为：用当前帧最近位置替代
            best_tid, best_d = None, float("inf")
            for t in tracks:
                cand = str(t["track_id"])
                if cand == tid:
                    continue
                bx, by, bw, bh = [float(v) for v in t.get("bbox", [0, 0, 0, 0])]
                d = (bx + bw / 2) ** 2 + (by + bh / 2) ** 2
                if d < best_d:
                    best_d, best_tid = d, cand
            if best_tid is not None:
                self._vlm_confirmed[best_tid] = max(
                    conf_until, self._vlm_confirmed.get(best_tid, 0))

    # ── B1: VLM 异步复核 ─────────────────────────────────────
    def _spatial_key(self, bbox: list[float]) -> tuple:
        x, y, w, h = [float(v) for v in bbox]
        return (int((x + w / 2) // self._spatial_grid), int((y + h / 2) // self._spatial_grid))

    def _select_review_frames_by_hand_mouth(self, track_id: str, bbox: list[float], k: int = 3,
                                              buffer=None):
        """复核主动选帧：按手-嘴距离排序环形缓冲帧，取距离最小（动作峰值）的 k 帧。

        吸烟是间歇动作，均匀采样常错过"手举烟"瞬间；姿态几何提供廉价的峰值信号。
        """
        entries = (buffer or self.frame_buffer).recent_entries(12)  # 最近 ~12 个缓冲帧
        if not entries:
            return []
        scored = []
        for fid, full_f in entries:
            kps = self.pose_cache.get(track_id)
            score = 0.0
            try:
                if kps:
                    score = -self.behavior_reasoner.hand_mouth_distance(kps, bbox)
            except Exception:
                score = 0.0
            scored.append((score, fid, full_f))
        scored.sort(key=lambda t: t[0])  # 距离最小在前
        return [(fid, f) for _, fid, f in scored[:k]]

    def _submit_async_vlm_review(self, track: dict, frame: np.ndarray, frame_id: int,
                                  pc_label: str, pc_conf: float, context: str = "") -> None:
        """提交异步 VLM 复核：立即返回，VLM 完成后回写 action_cache + vlm_review_cache。"""
        if self._edge_mode:
            # edge 决策模式：同步 crop 仲裁（毫秒级），复用同一套缓存/否决/确认回写语义
            self._crop_review_apply(track, frame, frame_id, pc_label, pc_conf)
            return
        track_id = str(track["track_id"])
        # 准备 person crop 帧——问题1：优先原帧缓冲（真实细节翻倍），无原帧回退降采样缓冲
        bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
        review_bbox = list(bbox)  # 闭包捕获：回调登记空间确认/否决用
        x, y, w, h = [int(round(v)) for v in bbox]
        frames = []
        src_buffer = self.frame_buffer
        sx = sy = 1.0
        if len(self.frame_buffer_orig) > 0:
            src_buffer = self.frame_buffer_orig
            proc_h, proc_w = (frame.shape[:2] if frame is not None else (540, 960))
            # 以最近一条原帧的实际尺寸换算（兼容不同分辨率源）
            _last = self.frame_buffer_orig.sample(1)
            if _last:
                orig_h, orig_w = _last[0][1].shape[:2]
                sx, sy = orig_w / max(1, proc_w), orig_h / max(1, proc_h)
        # 主动选帧：吸烟/手机类复核优先取该 track 手-嘴距离最近的历史帧（动作峰值更可能含烟）
        sampled = []
        if pc_label in ("M01", "M02"):
            try:
                sampled = self._select_review_frames_by_hand_mouth(track_id, bbox, k=3, buffer=src_buffer)
            except Exception:
                sampled = []
        if not sampled:
            sampled = src_buffer.sample(3)
        for _, full_f in sampled:
            h_f, w_f = full_f.shape[:2]
            _x, _y, _w, _h = x * sx, y * sy, w * sx, h * sy
            x1, y1 = max(0, int(_x)), max(0, int(_y))
            x2, y2 = min(w_f, int(_x + _w)), min(h_f, int(_y + _h))
            crop = full_f[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else full_f
            if crop.size > 0:
                frames.append(crop)
        if len(frames) < 2 and frame is not None:
            h_f, w_f = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_f, x + w), min(h_f, y + h)
            crop = frame[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else frame
            if crop.size > 0:
                frames = [crop]
        if not frames:
            return

        # 小 crop 上采样：960px 降采样源帧裁出的 person crop 常 <100px，VLM 视觉编码
        # 在该尺度下丢失细粒度动作证据（吸烟/手机），放大到短边 ≥256px 再送审
        upscaled = []
        for f in frames:
            fh, fw = f.shape[:2]
            short_side = min(fh, fw)
            if short_side > 0 and short_side < 256:
                scale = 256.0 / short_side
                f = cv2.resize(f, (max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))),
                               interpolation=cv2.INTER_CUBIC)
            upscaled.append(f)
        frames = upscaled

        # B5: 缓存检查——相同 crop 直接复用上次 VLM 结果
        cached_result = self._vlm_cache.get(frames[0])
        if cached_result is not None:
            with self._cache_lock:
                self.action_cache[track_id] = cached_result
                self.track_vlm_review_cache[track_id] = (int(frame_id), cached_result[0], cached_result[1])
                self._vlm_cache_sourced.add(track_id)  # 归因：该 cache 值源自 VLM
            return

        def callback(result: dict) -> None:
            """VLM 完成后回写缓存，下一帧自动使用复核结果。"""
            if result.get("status") == "success" and result.get("label"):
                new_label = str(result["label"])
                new_conf = float(result.get("confidence", pc_conf))
                # 双钥匙 B：VLM 单方面 Normal→M01 升级（无规则候选）不采信——防办公场景幻觉种子；
                # 规则触发后的复核确认（pc_label==M01，双钥匙第二把）直接采信，误报由规则几何把关
                if new_label == "M01" and pc_label != "M01":
                    new_label, new_conf = pc_label, pc_conf
                # 判定平滑：多数投票（最近3次≥2票）才允许改变方向，防单次抖动翻转状态
                if new_label != pc_label:
                    hist = self._review_history[track_id]
                    hist.append(new_label)
                    if len(hist) >= 2 and sum(1 for x in hist if x == new_label) >= 2:
                        pass  # 连续2次同向 → 采纳改变
                    else:
                        new_label, new_conf = pc_label, pc_conf  # 首次翻转 → 本次不采信，维持原判
                else:
                    self._review_history[track_id].append(new_label)
                # 归因诊断：判定留痕
                try:
                    import json as _json
                    from pathlib import Path as _Path
                    _vd = _Path("results/vlm_verdicts.jsonl")
                    with _vd.open("a", encoding="utf-8") as _vf:
                        _vf.write(_json.dumps({
                            "frame_id": int(frame_id), "track_id": track_id,
                            "claimed": pc_label, "returned": new_label, "conf": new_conf,
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                with self._cache_lock:
                    self.action_cache[track_id] = (new_label, new_conf)
                    self.track_vlm_review_cache[track_id] = (int(frame_id), new_label, new_conf)
                    self._vlm_cache_sourced.add(track_id)  # 归因：该 cache 值源自 VLM
                    # 归因修复：复核结论与声称不一致 → 否决锁存（连续2次同向才生效，防单次抖动误锁）
                    if new_label != pc_label and pc_label in self.force_vlm_classes:
                        veto_frames = int(self._vlm_veto_seconds * max(1.0, self._source_fps))
                        veto_hist = self._veto_history.setdefault(track_id, deque(maxlen=2))
                        veto_hist.append(new_label)
                        if len(veto_hist) >= 2 and all(x == new_label for x in veto_hist):
                            self._vlm_veto.setdefault(track_id, {})[pc_label] = int(frame_id) + veto_frames
                            if review_bbox is not None:
                                self._vlm_veto_spatial.setdefault(self._spatial_key(review_bbox), {})[pc_label] = int(frame_id) + veto_frames
                        else:
                            new_label, new_conf = pc_label, pc_conf  # 单次否决不生效也不回写
                    # 双钥匙 A：复核明确确认 M01 → 登记确认窗口（规则候选在其内可输出 M01）
                    if new_label == "M01":
                        hold = int(self._vlm_confirm_hold_seconds * max(1.0, self._source_fps))
                        self._vlm_confirmed[track_id] = int(frame_id) + hold
                        if review_bbox is not None:
                            key = self._spatial_key(review_bbox)
                            self._vlm_confirmed_spatial[key] = max(int(frame_id) + hold,
                                                                    self._vlm_confirmed_spatial.get(key, 0))
                if frames:
                    self._vlm_cache.put(frames[0], new_label, new_conf)

        self._vlm_async_executor.submit(self._async_vlm_review_job, frames, track_id, context, callback)

    # ── edge 决策模式：crop 仲裁（01 §2.4）─────────────────────────────
    def _crop_arbitrate(self, track: dict, frame, frame_id: int,
                        pc_label: str, pc_conf: float) -> tuple[str, float, "DecisionTrace"]:
        """crop 多分类对 PC 结果做确认/翻案。语义：

        - crop=Normal 且 PC 异常 → 翻案 Normal（置信度取均值，抑制误报）；
        - crop 异常且过该类阈值 → 采纳 crop 标签（支持跨类改判）；
        - crop 未达阈值 / 模型不可用 → 维持 PC 结果（trace 记 below_threshold/unavailable）。
        """
        tid = str(track["track_id"])
        trace = self._frame_traces.setdefault(tid, DecisionTrace(tid, frame_id))
        if not trace.steps:
            trace.add("pose_classifier", pc_label, pc_conf)
        cc = self.crop_classifier
        verdict = None
        if cc is not None and frame is not None:
            verdict = cc.cached(tid, frame_id)
            if verdict is None:
                bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
                try:
                    verdict = cc.classify(frame, [bbox], track_ids=[tid], frame_id=frame_id)[0]
                except Exception:
                    verdict = None
        if verdict is None or verdict.cls == "Unknown":
            trace.add("crop_arbiter", "unavailable", 0.0)
            return pc_label, pc_conf, trace
        if verdict.cls == "Normal":
            if (pc_label != "Normal" and pc_label not in cc.no_rebate_classes
                    and verdict.conf >= cc.normal_rebate_conf):
                trace.add("crop_arbiter", "Normal", verdict.conf, rejected=pc_label)
                return "Normal", round((pc_conf + verdict.conf) / 2.0, 3), trace
            trace.add("crop_arbiter", "Normal", verdict.conf, kept=pc_label)
            return pc_label, max(pc_conf, verdict.conf), trace
        if verdict.cls not in cc.arbiter_classes:
            # 演示类不参与仲裁：视为未确认，维持 PC（trace 留痕供排查）
            trace.add("crop_arbiter", verdict.cls, verdict.conf, non_arbiter=True)
            return pc_label, pc_conf, trace
        if cc.confirm(verdict):
            trace.add("crop_arbiter", verdict.cls, verdict.conf, confirmed=True)
            return verdict.cls, verdict.conf, trace
        trace.add("crop_arbiter", verdict.cls, verdict.conf, below_threshold=True)
        return pc_label, pc_conf, trace

    def _crop_review_apply(self, track: dict, frame, frame_id: int,
                           pc_label: str, pc_conf: float) -> None:
        """edge 模式下替代异步 VLM 任务：同步 crop 复核 + 与 VLM 回调同语义回写。

        回写语义对齐 _async_vlm_review_job.callback：节流缓存、否决锁存（连续2次同向）、
        M01 确认窗口——规则路径的 veto/双钥匙逻辑因此对 crop 结论同样生效。
        """
        tid = str(track["track_id"])
        new_label, new_conf, _trace = self._crop_arbitrate(track, frame, frame_id, pc_label, pc_conf)
        # no_rebate 保护的 veto 通道封堵：Normal 结论不得经否决锁存写入受保护类的缓存
        # （否则后续帧从缓存读回 Normal，绕过 _crop_arbitrate 的保护）
        cc = self.crop_classifier
        if cc is not None and new_label == "Normal" and pc_label in cc.no_rebate_classes:
            return  # 不写缓存：维持 PC 原判，后续帧继续仲裁
        effective_label, effective_conf = new_label, new_conf
        if new_label != pc_label and pc_label in self.force_vlm_classes:
            veto_frames = int(self._vlm_veto_seconds * max(1.0, self._source_fps))
            veto_hist = self._veto_history.setdefault(tid, deque(maxlen=2))
            veto_hist.append(new_label)
            if len(veto_hist) >= 2 and all(x == new_label for x in veto_hist):
                self._vlm_veto.setdefault(tid, {})[pc_label] = int(frame_id) + veto_frames
            else:
                effective_label, effective_conf = pc_label, pc_conf  # 单次翻转不生效
        with self._cache_lock:
            self.action_cache[tid] = (effective_label, effective_conf)
            self.track_vlm_review_cache[tid] = (int(frame_id), effective_label, effective_conf)
            self._vlm_cache_sourced.add(tid)
            if effective_label == "M01":
                hold = int(self._vlm_confirm_hold_seconds * max(1.0, self._source_fps))
                self._vlm_confirmed[tid] = int(frame_id) + hold

    def _async_vlm_review_job(self, frames: list, track_id: str, context: str,
                              callback) -> None:
        """异步 VLM 复核工作线程。"""
        # B3: 全局并发槽位——复核类调用阻塞排队（10s 超时后放弃，PC 结果兜底）
        if not self.vlm_router.try_acquire(timeout=10.0):
            return
        try:
            result = self.vlm_reviewer.review_clip(
                frames=frames, track_id=track_id, context=context,
            )
            callback(result)
        except Exception:
            # VLM 失败 = 不修正（PC 结果留在 cache 里，已经是对的）
            pass
        finally:
            self.vlm_router.release()

    def _forced_vlm_review(self, track: dict, frame, frame_id: int, label: str, conf: float) -> tuple[str, float]:
        """对强制复核类别（如 H03）做 VLM 确认；同目标节流，失败回退原结果。"""
        track_id = str(track["track_id"])
        cached = self.track_vlm_review_cache.get(track_id)
        if cached is not None and (frame_id - cached[0]) < self.force_vlm_throttle_frames:
            return cached[1], cached[2]
        # person crop 模式：从环形缓冲全帧按 bbox 裁 3 帧高清 crop
        # （吸烟/跌倒等细粒度行为需要看清手部与烟，全场景 480px 不够）
        bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
        x, y, w, h = [int(round(v)) for v in bbox]
        frames = []
        sampled = self.frame_buffer.sample(3)
        for _, full_f in sampled:
            h_f, w_f = full_f.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_f, x + w), min(h_f, y + h)
            crop = full_f[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else full_f
            if crop.size > 0:
                frames.append(crop)
        if len(frames) < 2:
            if frame is not None:
                h_f, w_f = frame.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_f, x + w), min(h_f, y + h)
                crop = frame[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else frame
                if crop.size > 0:
                    frames = [crop]
        result = self.vlm_reviewer.review_clip(
            frames=frames,
            track_id=track_id,
            context=self._build_review_context(track_id),
        )
        if result.get("status") == "success" and result.get("label"):
            label = str(result["label"])
            conf = float(result.get("confidence", conf))
        self.track_vlm_review_cache[track_id] = (int(frame_id), label, conf)
        return label, conf

    def _build_review_context(self, track_id: str) -> str:
        """组装灰区复核上下文：场景叙述摘要 + 该目标近期行为序列。"""
        if not getattr(self, "review_context_enabled", True):
            return ""
        parts: list[str] = []
        scene_ctx = self.scene_narrator.get_context()
        if scene_ctx:
            parts.append(scene_ctx)
        history = self.track_label_history.get(track_id)
        if history:
            seq = " → ".join(f"{label}({conf:.2f})" for label, conf in list(history)[-12:])
            parts.append(f"该目标近期模型输出序列：{seq}")
        return "\n".join(parts)

    # ── T3.x 语言层 / MQTT（01 §2.5-2.6）────────────────────────────
    def _on_mqtt_rule_set(self, payload: dict) -> None:
        """MQTT rule/set 下行：一句话布控热更新（与 REST create_alert_rule 同一 add_rule 路径）。"""
        try:
            text = str(payload.get("text", "")).strip()
            if text:
                self.nl_rule_engine.add_rule(text)
        except Exception as exc:
            self.mqtt_bridge.last_debug["rule_apply_error"] = str(exc)

    def _mqtt_publish_alert(self, alert: dict, frame_id: int) -> None:
        """告警事件 MQTT 上行（event topic，区别于 1s 周期的 status）。"""
        try:
            event = {
                "event_id": f"evt_{int(frame_id)}",
                "frame_id": int(frame_id),
                "type": alert.get("type", "none"),
                "source": alert.get("source", ""),
                "label": alert.get("label") or alert.get("matched_class") or "",
                "confidence": alert.get("confidence"),
                "final_decision": alert.get("final_decision", "pending"),
                "rule_id": alert.get("rule_id"),
                "rule_text": alert.get("rule_text", ""),
                "ts": time.time(),
            }
            self.mqtt_bridge.publish_event(event)
        except Exception as exc:
            try:
                self.mqtt_bridge.last_debug["event_publish_error"] = str(exc)
            except Exception:
                pass

    def llm_online(self) -> bool:
        """文本 LLM 在线探测（60s 缓存）。

        盒上适配: `localhost` 在盒上优先解析 ::1，而 llama-server 只监听 127.0.0.1，
        原实现直接 GET localhost → 恒判离线；这里统一归一化到 127.0.0.1。
        """
        now = time.time()
        if now - getattr(self, "_llm_probe_ts", 0.0) <= 60.0:
            return bool(getattr(self, "_llm_online", False))
        self._llm_probe_ts = now
        try:
            import requests as _rq
            # engine 自身不持有 vllm_url（在各子组件里），统一取 VLMReviewer 的配置值
            url = str(getattr(self.vlm_reviewer, "vllm_url", "http://127.0.0.1:8001/v1/chat/completions"))
            probe_url = url.rsplit("/chat/", 1)[0].replace("//localhost", "//127.0.0.1")
            self._llm_online = _rq.get(f"{probe_url}/models", timeout=1.5).ok
        except Exception:
            self._llm_online = False
        return bool(self._llm_online)

    def _mqtt_status_payload(self) -> dict:
        """status 上行载荷：板级遥测（npu-smi）+ 引擎摘要；llm 在线探测带 60s 缓存。"""
        try:
            from edge_migration import npu_telemetry
        except Exception:
            npu_telemetry = None
        payload: dict = {"box_id": self.mqtt_bridge.box_id, "ts": time.time(),
                         "decision_mode": self.decision_mode,
                         "persons": len(self.last_track_snapshot)}
        if npu_telemetry is not None:
            tel = npu_telemetry.read()
            if tel:
                payload.update({k: tel[k] for k in ("power_w", "temp_c", "ai_core_util_pct")
                                if k in tel})
        now = time.time()
        payload["llm_online"] = self.llm_online()
        return payload

    def _attach_event_facts(self, alert: dict, frame_id: int, detections: list) -> None:
        """T3.5 后半：告警附带结构化事实——裁决第一趟不发图、用事实文本（边缘省一次多模态调用）。"""
        try:
            from edge_migration.fact_compiler import (
                build_event_facts,
                build_scene_facts,
                facts_to_text,
            )
            tracks = []
            for d in detections or []:
                try:
                    if isinstance(d, dict):
                        tracks.append({"track_id": d.get("track_id"),
                                       "label": d.get("label", "Normal"),
                                       "confidence": float(d.get("confidence", 0.0) or 0.0)})
                    else:
                        behavior = getattr(d, "behavior", None)
                        tracks.append({"track_id": getattr(d, "track_id", None),
                                       "label": getattr(behavior, "type", "Normal"),
                                       "confidence": float(getattr(behavior, "confidence", 0.0) or 0.0)})
                except Exception:
                    continue
            facts = build_event_facts(
                alert={"type": alert.get("type", ""), "label": alert.get("label", ""),
                       "confidence": alert.get("confidence"), "frame_id": frame_id,
                       "rule_text": alert.get("rule_text", "")},
                scene_facts=build_scene_facts(frame_id=frame_id, tracks=tracks),
                traces=[t if isinstance(t, dict) else t.to_dict()
                        for t in (self._frame_traces or {}).values()])
            alert["facts"] = facts
            alert["facts_text"] = facts_to_text(facts).get("description", "")
        except Exception as exc:
            # 事实化失败不影响告警链路：仅记录，裁决回退为原图像两趟路径
            try:
                self.mqtt_bridge.last_debug["event_facts_error"] = str(exc)
            except Exception:
                pass

    def narrate_track(self, track_id: str) -> dict:
        """时刻②点谁说谁：最近帧快照 + 行为历史 → 结构化事实 → 文本旁白。"""
        from edge_migration.fact_compiler import build_track_facts
        det = next((d for d in self.last_track_snapshot
                    if str(d.get("track_id")) == str(track_id)), None)
        if det is None:
            return {"ok": False, "text": "", "source": "none",
                    "reason": "track_not_in_last_frame"}
        history = list(self.track_label_history.get(str(track_id), []))
        facts = build_track_facts(
            track_id=str(track_id), frame_id=self._last_frame_id,
            window_seconds=len(history) / max(self._source_fps, 1.0),
            label_history=history, bboxes=[det.get("bbox", [0, 0, 0, 0])])
        out = self.track_narrator.narrate_track(facts)
        # T6.5：LLM 失败/降级时回退预生成旁白（演示剧本；真实优先，气泡上 source 可见）
        if self.demo_pack.enabled and (not out.get("ok") or out.get("source") != "llm"):
            preset = self.demo_pack.next_narration()
            if preset:
                out = {**out, "ok": True, "text": preset, "source": "demo_preset"}
        out["facts"] = facts
        return out

    def _motion_score(self, frame) -> float:
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._last_detector_gray is None:
            self._last_detector_gray = gray
            return 1.0

        diff = cv2.absdiff(gray, self._last_detector_gray)
        self._last_detector_gray = gray
        return float(np.mean(diff) / 255.0)

    def _should_run_detector(self, frame_id: int, motion_score: float, has_tracks: bool) -> bool:
        if not self.enable_detector_gating:
            return True
        if frame_id <= self.detector_warmup_frames:
            return True
        if not has_tracks:
            return True
        if self._detector_skip_count >= self.detector_max_skip_frames:
            return True
        if motion_score >= self.detector_motion_threshold:
            return True
        if motion_score <= self.detector_low_motion_threshold and (frame_id % self.detector_stride) != 0:
            return False
        return (frame_id % self.detector_stride) == 0

    def _should_refresh_pose(self, frame_id: int, track: dict, track_age: int) -> bool:
        track_id = track["track_id"]
        if not self.enable_pose_motion_gate:
            return False

        if track_id not in self.pose_cache:
            return True

        last_pose_frame = int(self.pose_last_frame.get(track_id, 0))
        if last_pose_frame > 0 and (frame_id - last_pose_frame) >= self.pose_force_refresh_gap:
            return True

        if (frame_id % self.pose_stride) == 0:
            return True

        prev_bbox = self.pose_last_bbox.get(track_id)
        if prev_bbox is None:
            return False

        iou = self._bbox_iou_xywh(prev_bbox, track.get("bbox", [0, 0, 0, 0]))
        if iou < self.pose_motion_gate_iou_threshold:
            return True
        if track_age <= self.action_min_track_age + 1:
            return True
        return False

    def process(self, frame, frame_id: int, orig_frame=None) -> FrameResult:
        """同步推理一帧：render / 评估 / 管道禁用时使用（行为与重构前一致）。"""
        ctx = self._make_ctx(frame, frame_id)
        if orig_frame is not None:
            ctx["orig_frame"] = orig_frame
        self._stage1_detect(ctx)
        self._stage2_pose(ctx)
        self._stage3_recognize(ctx)
        return self._assemble_result(ctx)

    # ── B6: 三阶段流水线 ─────────────────────────────────────
    # 同步路径（render/评估/管道禁用）：process() 顺序调用四个阶段方法，行为与旧版完全一致。
    # 管道路径：三个工作线程各自消费队列跑一个阶段，帧间重叠执行
    # （stage1 跑第 N+1 帧检测时，stage2 跑第 N 帧姿态、stage3 跑第 N-1 帧行为）。
    # 跨阶段状态通过 ctx dict 传递；各阶段只独占自己的模块，共享 dict 均为单键原子操作。

    def _make_ctx(self, frame, frame_id: int) -> dict:
        return {"frame": frame, "frame_id": frame_id, "start": time.perf_counter()}

    def _stage1_detect(self, ctx: dict) -> None:
        """阶段1：运动门控 + 检测/跟踪 + C1 ID切换检测。"""
        frame = ctx["frame"]
        frame_id = ctx["frame_id"]

        stage_detector_t0 = time.perf_counter()
        motion_score = self._motion_score(frame)
        run_detector = self._should_run_detector(frame_id, motion_score, bool(self.track_seen_frames))
        if run_detector:
            detections_raw = self.detector.infer(frame)
            tracks = self.tracker.update(detections_raw)
            self._detector_skip_count = 0
        else:
            tracks = self.tracker.advance_without_detections()
            self._detector_skip_count += 1
        stage_detector_ms = (time.perf_counter() - stage_detector_t0) * 1000.0

        # C1: ID 切换检测——bbox 跳变时清零该 track 的逐目标缓存（防混人序列污染）
        self._detect_and_handle_id_switches(tracks)
        ctx["tracks"] = tracks
        ctx["run_detector"] = run_detector
        ctx["stage_detector_ms"] = stage_detector_ms

    def _stage2_pose(self, ctx: dict) -> None:
        """阶段2：track 状态装配 + 姿态批次推理（pose_cache 更新）。"""
        frame = ctx["frame"]
        frame_id = ctx["frame_id"]
        tracks = ctx["tracks"]
        stage_pose_ms = 0.0
        stage_action_ms = 0.0

        detections: list[Detection] = []
        # B1: VLM 异步复核——本帧标记 pending，VLM 完成后下一帧从 cache 自动更新
        vlm_review = VLMReview(enabled=self.vlm_reviewer.enabled, status="pending" if self.vlm_reviewer.enabled else "skipped")

        main_label = "Normal"
        main_confidence = 0.0
        pending_tracks: list[dict] = []
        sorted_tracks = sorted(tracks, key=lambda item: float(item.get("score", 0.0)), reverse=True)

        pose_compute_set = set()
        if (frame_id % self.pose_stride) == 0:
            pose_compute_set = {item.get("track_id") for item in sorted_tracks[: self.max_pose_tracks]}

        action_candidates = []
        for item in sorted_tracks:
            score = float(item.get("score", 0.0))
            if score < self.action_min_track_score:
                continue
            action_candidates.append(item)
            if len(action_candidates) >= self.max_action_tracks:
                break
        action_compute_set = {item.get("track_id") for item in action_candidates}

        action_runs = 0
        pose_skip_motion_gate = 0
        action_skip_warmup = 0
        action_skip_track_age = 0
        action_skip_gate = 0
        action_skip_cached = 0
        action_skip_default = 0
        rule_override_count = 0
        reasoner_source_counts = {"action_model": 0, "rule_engine": 0}
        reasoner_rule_tag_counts: dict[str, int] = {}
        h01_h04_mapping = {
            "base_h01": 0,
            "base_h04": 0,
            "refined_h01": 0,
            "refined_h04": 0,
            "base_h01_conf_sum": 0.0,
            "base_h04_conf_sum": 0.0,
            "refined_h01_conf_sum": 0.0,
            "refined_h04_conf_sum": 0.0,
        }

        track_states: list[dict[str, object]] = []
        pose_batch_tracks: list[dict] = []
        pose_batch_state_indices: list[int] = []

        for track in tracks:
            track_id = track["track_id"]
            self.track_seen_frames[track_id] = self.track_seen_frames.get(track_id, 0) + 1
            track_age = self.track_seen_frames[track_id]

            if self.enable_pose_motion_gate:
                should_pose_infer = self._should_refresh_pose(frame_id, track, track_age)
                if (
                    not should_pose_infer
                    and track_id in self.pose_cache
                    and track_id in pose_compute_set
                    and self.pose_motion_gate_iou_threshold <= 1.0
                ):
                    prev_bbox = self.pose_last_bbox.get(track_id)
                    if prev_bbox is not None:
                        iou = self._bbox_iou_xywh(prev_bbox, track.get("bbox", [0, 0, 0, 0]))
                        if iou >= self.pose_motion_gate_iou_threshold:
                            should_pose_infer = False
                            pose_skip_motion_gate += 1
            else:
                should_pose_infer = track_id in pose_compute_set or track_id not in self.pose_cache

            state = {
                "track": track,
                "track_id": track_id,
                "track_age": track_age,
                "keypoints": self.pose_cache.get(track_id, []),
                "pose_should_infer": should_pose_infer,
                "label": self.action_cache.get(track_id, (self.action_default_label, self.action_default_confidence))[0],
                "confidence": self.action_cache.get(track_id, (self.action_default_label, self.action_default_confidence))[1],
            }
            track_states.append(state)

            if should_pose_infer:
                pose_batch_state_indices.append(len(track_states) - 1)
                pose_batch_tracks.append(track)

        if pose_batch_tracks:
            self._pose_batch_runs += 1
            pose_t0 = time.perf_counter()
            pose_batch_results = self.pose_estimator.infer_batch(frame, pose_batch_tracks)
            stage_pose_ms += (time.perf_counter() - pose_t0) * 1000.0
            for state_index, keypoints in zip(pose_batch_state_indices, pose_batch_results):
                state = track_states[state_index]
                track = state["track"]
                track_id = str(state["track_id"])
                keypoints = keypoints or []
                self.pose_cache[track_id] = keypoints
                self.pose_last_bbox[track_id] = list(track.get("bbox", [0, 0, 0, 0]))
                self.pose_last_frame[track_id] = frame_id
                state["keypoints"] = keypoints
        ctx["detections"] = detections
        ctx["vlm_review"] = vlm_review
        ctx["main_label"] = main_label
        ctx["main_confidence"] = main_confidence
        ctx["pending_tracks"] = pending_tracks
        ctx["track_states"] = track_states
        ctx["stage_pose_ms"] = stage_pose_ms
        ctx["stage_action_ms"] = stage_action_ms
        ctx["action_compute_set"] = action_compute_set
        ctx["action_runs"] = action_runs
        ctx["pose_skip_motion_gate"] = pose_skip_motion_gate
        ctx["rule_override_count"] = rule_override_count
        ctx["reasoner_source_counts"] = reasoner_source_counts
        ctx["reasoner_rule_tag_counts"] = reasoner_rule_tag_counts
        ctx["h01_h04_mapping"] = h01_h04_mapping
        ctx["action_skip_warmup"] = action_skip_warmup
        ctx["action_skip_track_age"] = action_skip_track_age
        ctx["action_skip_gate"] = action_skip_gate
        ctx["action_skip_cached"] = action_skip_cached
        ctx["action_skip_default"] = action_skip_default

    def _drain_action_pc(self, frame_id: int, tracks: list[dict]) -> list[tuple[str, float]]:
        """取异步 PC 结果；过期/代际不符则退回 action_cache/default。"""
        items = None
        with self._action_pc_lock:
            result = self._action_pc_result
            self._action_pc_result = {}
        if result and result.get("generation") == self._pipe_generation:
            max_age = max(2 * int(self.action_stride), 24)
            if int(frame_id) - int(result.get("frame_id", frame_id)) <= max_age:
                items = result.get("items") or {}
        default = (self.action_default_label, self.action_default_confidence)
        out: list[tuple[str, float]] = []
        for track in tracks:
            tid = str(track["track_id"])
            if items and tid in items:
                out.append(items[tid])
            else:
                out.append(self.action_cache.get(tid, default))
        return out

    def _submit_action_pc(self, tracks: list[dict], frame_id: int) -> None:
        """把 PC 批次提交给单线程 worker；姿态 clip 先复制，避免读主线程 deque。"""
        with self._action_pc_lock:
            if self._action_pc_inflight:
                return
            self._action_pc_inflight = True
        ids: list[str] = []
        tensors: list[np.ndarray] = []
        for track in tracks:
            tid = str(track["track_id"])
            tensor = self.action_recognizer.build_pose_tensor(tid)
            if tensor is not None:
                ids.append(tid)
                tensors.append(tensor)
        if not tensors:
            with self._action_pc_lock:
                self._action_pc_inflight = False
            return
        generation = self._pipe_generation

        def _job() -> None:
            started = time.perf_counter()
            results = self.action_recognizer.infer_pose_classifier_tensors(tensors)
            elapsed = (time.perf_counter() - started) * 1000.0
            with self._action_pc_lock:
                if generation == self._pipe_generation:
                    self._action_pc_result = {
                        "generation": generation,
                        "frame_id": int(frame_id),
                        "items": dict(zip(ids, results)),
                    }
                    self._action_pc_last_result = dict(self._action_pc_result)
                self._action_pc_async_samples.append(elapsed)
                self._action_pc_inflight = False

        self._action_pc_pool.submit(_job)

    def _fast_geometry_candidate(self, state: dict, frame: np.ndarray) -> str | None:
        track = state.get("track") or {}
        bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
        w, h = bbox[2], bbox[3]
        if w <= 0 or h <= 0:
            return None
        aspect = w / h
        height_ratio = h / max(1.0, float(frame.shape[0]))
        keypoints = state.get("keypoints") or []
        torso = (self.behavior_reasoner._pose_torso_vertical_ratio(keypoints)
                 if keypoints else None)
        if aspect >= self._fast_trigger_aspect and height_ratio <= self._fast_trigger_max_height_ratio:
            return "shape"
        if torso is not None and torso <= self._fast_trigger_torso_horizontal:
            return "torso"
        return None

    def _update_pending_confirmations(self, track_states: list[dict], frame_id: int,
                                      frame: np.ndarray | None) -> None:
        """板端闭环：PC 结果优先，几何持续性兜底；状态变化写入更新队列。"""
        if not self._pending_alerts or frame is None:
            return
        fps = max(1.0, float(getattr(self, "_source_fps", 8.0)))
        ttl_frames = int(15.0 * fps)
        confirm_frames = max(2, int(self._fast_confirm_geometry_seconds * fps))
        reject_frames = max(2, int(self._fast_reject_geometry_seconds * fps))
        states = {str(s.get("track_id")): s for s in track_states}
        with self._action_pc_lock:
            pc_result = dict(self._action_pc_last_result or {})
        for tid, pending in list(self._pending_alerts.items()):
            trigger_frame = int(pending.get("trigger_frame", frame_id))
            if frame_id - trigger_frame > ttl_frames:
                self._pending_alerts.pop(tid, None)
                continue
            alert = pending.get("alert") or {}
            decision = None
            source = ""
            label = "H04"
            conf = float(alert.get("confidence", 0.6))
            items = None
            if (pc_result.get("generation") == self._pipe_generation
                    and int(pc_result.get("frame_id", -1)) >= trigger_frame):
                items = pc_result.get("items") or {}
            if items and tid in items:
                pc_label, pc_conf = items[tid]
                if pc_label == "H04" and float(pc_conf) >= 0.4:
                    decision, source, label, conf = "confirmed", "pose_classifier", pc_label, float(pc_conf)
                elif pc_label == "Normal" and float(pc_conf) >= 0.7:
                    decision, source, label, conf = "rejected", "pose_classifier", pc_label, float(pc_conf)
            if decision is None:
                state = states.get(tid)
                if state is not None and self._fast_geometry_candidate(state, frame):
                    pending["missing"] = 0
                    if frame_id - trigger_frame >= confirm_frames:
                        decision, source, label, conf = "confirmed", "geometry_persist", "H04", max(conf, 0.6)
                else:
                    pending["missing"] = int(pending.get("missing", 0)) + 1
                    if pending["missing"] >= reject_frames:
                        decision = "rejected"
                        source = "geometry_lost" if state is not None else "track_lost"
                        conf = 0.0
            if decision is None:
                continue
            self._pending_alerts.pop(tid, None)
            update = dict(alert)
            update.update({
                "triggered": True,
                "final_decision": decision,
                "source": source,
                "label": label,
                "confidence": round(float(conf), 2),
                "decided_frame": int(frame_id),
                "reason": source,
            })
            self._pending_alert_updates.append(update)

    def _fast_fall_trigger(self, track_states: list[dict], frame_id: int,
                           frame: np.ndarray | None) -> dict | None:
        """方案 A：框形/躯干/下沉三选一，连续 2 帧即发 pending 告警。

        不依赖 PoseClassifier 的 32 帧缓冲；只用当前 bbox + 缓存 keypoints，
        8fps 下约 0.25s 报警。
        """
        if not self._fast_trigger_enabled or frame is None:
            return None
        frame_h = float(frame.shape[0])
        fps = max(1.0, float(getattr(self, "_source_fps", 8.0)))
        cooldown = max(1, int(self._fast_trigger_cooldown_seconds * fps))
        for state in track_states:
            track = state.get("track") or {}
            bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
            w, h = bbox[2], bbox[3]
            if w <= 0 or h <= 0:
                continue
            tid = str(state.get("track_id"))
            aspect = w / h
            height_ratio = h / max(1.0, frame_h)
            cy = bbox[1] + h / 2.0
            keypoints = state.get("keypoints") or []
            torso = (self.behavior_reasoner._pose_torso_vertical_ratio(keypoints)
                     if keypoints else None)
            prev = self._fast_trigger_prev_bbox.get(tid)
            drop = 0.0
            if prev is not None and (frame_id - prev[0]) <= max(2, int(0.5 * fps)):
                prev_cy = prev[1][1] + prev[1][3] / 2.0
                drop = (cy - prev_cy) / max(1.0, frame_h)
            self._fast_trigger_prev_bbox[tid] = (int(frame_id), list(bbox))

            shape_candidate = aspect >= self._fast_trigger_aspect and height_ratio <= self._fast_trigger_max_height_ratio
            torso_candidate = torso is not None and torso <= self._fast_trigger_torso_horizontal
            drop_candidate = drop >= self._fast_trigger_drop_ratio
            if not (shape_candidate or torso_candidate or drop_candidate):
                self._fast_trigger_hits[tid] = 0
                continue
            hits = int(self._fast_trigger_hits.get(tid, 0)) + 1
            self._fast_trigger_hits[tid] = hits
            if hits < 2:
                continue
            last = int(self._fast_trigger_last.get(tid, -10**9))
            if frame_id - last < cooldown:
                continue
            self._fast_trigger_last[tid] = int(frame_id)
            self._fast_trigger_hits[tid] = 0
            snapshot_url = ""
            try:
                snap_dir = getattr(self.alert_system, "snapshots_dir", None)
                if snap_dir is not None:
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    snap_path = (snap_dir / f"fast_alert_{int(frame_id)}.jpg").resolve()
                    if cv2.imwrite(str(snap_path), frame):
                        snapshot_url = f"/api/v1/media?path={snap_path}"
            except Exception:
                snapshot_url = ""
            reason = "shape" if shape_candidate else ("torso" if torso_candidate else "drop")
            print(f"[fast_trigger] frame={frame_id} tid={tid} reason={reason} "
                  f"aspect={aspect:.2f} h_ratio={height_ratio:.2f} torso={torso} drop={drop:.3f}", flush=True)
            alert = {
                "triggered": True,
                "type": "h04_detected",
                "source": "fast_trigger",
                "final_decision": "pending",
                "consecutive_frames": hits,
                "snapshot_url": snapshot_url,
                "clip_path": "",
                "track_id": tid,
                "bbox": [round(v, 1) for v in bbox],
                "label": "H04",
                "confidence": 0.6,
                "reason": reason,
                "frame_id": int(frame_id),
                "timestamp": time.time(),
            }
            self._pending_alerts[tid] = {
                "alert": dict(alert),
                "trigger_frame": int(frame_id),
                "created": time.time(),
                "missing": 0,
            }
            try:
                self._submit_action_pc([track], frame_id)
            except Exception:
                pass
            return alert
        return None

    def _stage3_recognize(self, ctx: dict) -> None:
        """阶段3：三级决策 + 规则精化 + C2平滑 + 告警/场景旁路 + 姿态缓冲。"""
        self._frame_traces = {}  # edge 时刻④：每帧重建决策链寄存器
        frame = ctx["frame"]
        frame_id = ctx["frame_id"]
        tracks = ctx["tracks"]
        start = ctx["start"]
        stage_action_ms = ctx["stage_action_ms"]
        track_states = ctx["track_states"]
        action_compute_set = ctx["action_compute_set"]
        detections = ctx["detections"]
        vlm_review = ctx["vlm_review"]
        main_label = ctx["main_label"]
        main_confidence = ctx["main_confidence"]
        pending_tracks = ctx["pending_tracks"]
        action_runs = ctx["action_runs"]
        pose_skip_motion_gate = ctx["pose_skip_motion_gate"]
        rule_override_count = ctx["rule_override_count"]
        reasoner_source_counts = ctx["reasoner_source_counts"]
        reasoner_rule_tag_counts = ctx["reasoner_rule_tag_counts"]
        h01_h04_mapping = ctx["h01_h04_mapping"]
        action_skip_warmup = ctx["action_skip_warmup"]
        action_skip_track_age = ctx["action_skip_track_age"]
        action_skip_gate = ctx["action_skip_gate"]
        action_skip_cached = ctx["action_skip_cached"]
        action_skip_default = ctx["action_skip_default"]
        action_batch_tracks: list[dict] = []
        action_batch_keypoints: list[list[list[float]]] = []
        action_batch_state_indices: list[int] = []

        for state_index, state in enumerate(track_states):
            track = state["track"]
            track_id = str(state["track_id"])
            track_age = int(state["track_age"])
            keypoints = state["keypoints"]

            can_run_action = True
            if frame_id <= self.action_warmup_frames:
                can_run_action = False
                action_skip_warmup += 1
            elif track_age < self.action_min_track_age:
                can_run_action = False
                action_skip_track_age += 1
            elif track_id not in action_compute_set:
                can_run_action = False
                action_skip_gate += 1

            should_run_by_stride = (frame_id % self.action_stride) == 0 or track_id not in self.action_cache
            if can_run_action and should_run_by_stride:
                action_batch_state_indices.append(state_index)
                action_batch_tracks.append(track)
                action_batch_keypoints.append(keypoints)
            elif track_id in self.action_cache:
                state["label"], state["confidence"] = self.action_cache[track_id]
                action_skip_cached += 1
            else:
                state["label"] = self.action_default_label
                state["confidence"] = self.action_default_confidence
                self.action_cache[track_id] = (self.action_default_label, self.action_default_confidence)
                action_skip_default += 1

        # D4: 时间粒度对齐——每个处理帧都为所有 track 追加姿态到缓冲
        # （与训练数据的连续帧口径一致；此前只在 action 步长帧追加，时间粒度不匹配）
        frame_h, frame_w = frame.shape[:2] if frame is not None else (1080, 1920)
        # B6: keypoints 随 ctx 传递，stage3 不跨线程读 stage2 的 pose_cache
        for _state in track_states:
            self.action_recognizer.append_pose(str(_state["track_id"]), _state["keypoints"],
                                               frame_width=frame_w, frame_height=frame_h)

        batch_label_sources: dict[str, str] = {}
        if action_batch_tracks:
            action_t0 = time.perf_counter()
            # ── 三级决策：PoseClassifier 主路径 ──────────────────
            if self.action_recognizer.pose_classifier_loaded:
                # Step 1: 取上一次异步 PC 结果；当帧只提交新任务，不等待 300ms
                try:
                    pc_results = self._drain_action_pc(frame_id, action_batch_tracks)
                    self._submit_action_pc(action_batch_tracks, frame_id)
                except Exception:
                    pc_results = [("Normal", 0.0) for _ in action_batch_tracks]
                # Step 2: 三级决策
                three_level_results: list[tuple[str, float]] = []
                vlm_fallback_tracks: list[dict] = []
                vlm_fallback_indices: list[int] = []
                batch_label_sources: dict[str, str] = {}
                for state_idx, (track, pc_result) in enumerate(zip(action_batch_tracks, pc_results)):
                    pc_label, pc_conf = pc_result
                    src_tag = "pc_high" if pc_conf >= self.action_recognizer.pc_conf_high else (
                        "pc_gray" if pc_conf >= self.action_recognizer.pc_conf_low else "slowfast")
                    # B1: 强制复核改为异步——立即用 PC 结果，复核回写 cache 供下一帧
                    # edge 模式：复核即同步 crop 仲裁，回写后当帧读回
                    if pc_label in self.force_vlm_classes:
                        cached = self.track_vlm_review_cache.get(str(track["track_id"]))
                        if cached is not None and (frame_id - cached[0]) < self.force_vlm_throttle_frames:
                            pc_label, pc_conf = cached[1], cached[2]
                            src_tag = "crop_cache" if self._edge_mode else "vlm_cache"  # 归因：取自复核节流缓存
                        else:
                            self._submit_async_vlm_review(track, frame, frame_id, pc_label, pc_conf,
                                                          context=self._build_review_context(str(track["track_id"])))
                            if self._edge_mode:
                                cached = self.track_vlm_review_cache.get(str(track["track_id"]))
                                if cached is not None and cached[0] == int(frame_id):
                                    pc_label, pc_conf = cached[1], cached[2]
                                src_tag = "crop_forced"
                    batch_label_sources[str(track["track_id"])] = src_tag
                    if pc_conf >= self.action_recognizer.pc_conf_high:
                        # 级别1：高置信度，直接采纳
                        three_level_results.append((pc_label, pc_conf))
                    elif pc_conf >= self.action_recognizer.pc_conf_low:
                        # 级别2：灰色地带 → server: 异步VLM复核 / edge: 同步crop仲裁
                        if self._edge_mode:
                            self._submit_async_vlm_review(track, frame, frame_id, pc_label, pc_conf,
                                                          context=self._build_review_context(str(track["track_id"])))
                            cached = self.track_vlm_review_cache.get(str(track["track_id"]))
                            if cached is not None and cached[0] == int(frame_id):
                                pc_label, pc_conf = cached[1], cached[2]
                            three_level_results.append((pc_label, pc_conf))
                            src_tag = "crop_arbiter"
                        else:
                            three_level_results.append((pc_label, pc_conf))
                            vlm_fallback_tracks.append(track)
                            vlm_fallback_indices.append(state_idx)
                    else:
                        # 级别3：低置信度 → server: SlowFast 关键词映射 / edge: 禁用兜底保持 PC
                        three_level_results.append((pc_label, pc_conf))
                        if self._slowfast_enabled:
                            # 用 SlowFast 覆盖
                            sf_result = self.action_recognizer.infer(
                                track, frame_id,
                                action_batch_keypoints[state_idx] if state_idx < len(action_batch_keypoints) else None,
                                frame,
                            )
                            three_level_results[state_idx] = sf_result
                        elif self._edge_mode:
                            # edge 无兜底：记录"低置信保持"决策链（认知视图可见模型不确定态）
                            self._frame_traces.setdefault(
                                str(track["track_id"]),
                                DecisionTrace(str(track["track_id"]), frame_id),
                            ).add("pose_classifier", pc_label, pc_conf)

                # B1: 灰区 VLM 复核改为异步——立即用 PC 结果，VLM 完成后回写 cache
                # （edge 模式已在上方同步 crop 仲裁，跳过该异步路径）
                if vlm_fallback_tracks and self.vlm_reviewer.enabled and not self._edge_mode:
                    for vlm_idx, track in enumerate(vlm_fallback_tracks):
                        track_id = str(track["track_id"])
                        # 提取 person crop 帧用于 VLM 时序上下文
                        vlm_frames = []
                        if frame is not None:
                            bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
                            x, y, w, h = [int(round(v)) for v in bbox]
                            h_f, w_f = frame.shape[:2]
                            x1, y1 = max(0, x), max(0, y)
                            x2, y2 = min(w_f, x + w), min(h_f, y + h)
                            crop = frame[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else frame
                            if crop.size > 0:
                                vlm_frames.append(crop)

                        # 异步提交：PC 结果保留在本帧，VLM 回写 action_cache 供下一帧
                        self._submit_async_vlm_review(track, frame, frame_id,
                                                      three_level_results[vlm_fallback_indices[vlm_idx]][0],
                                                      three_level_results[vlm_fallback_indices[vlm_idx]][1],
                                                      context=self._build_review_context(track_id))

                action_batch_results = three_level_results
            else:
                # PoseClassifier 未加载：server 走 SlowFast 批量；edge 且禁用 SlowFast 时保持默认
                if self._edge_mode and not self._slowfast_enabled:
                    action_batch_results = [(self.action_default_label, self.action_default_confidence)
                                            for _ in action_batch_tracks]
                else:
                    action_batch_results = self.action_recognizer.infer_batch(
                        action_batch_tracks,
                        frame_id,
                        pose_keypoints_list=action_batch_keypoints,
                        frame=frame,
                    )
            stage_action_ms += (time.perf_counter() - action_t0) * 1000.0
            action_runs += len(action_batch_results)
            for state_index, (label, confidence) in zip(action_batch_state_indices, action_batch_results):
                state = track_states[state_index]
                track_id = str(state["track_id"])
                state["label"] = label
                state["confidence"] = confidence
                self.action_cache[track_id] = (label, confidence)
                # 行为序列历史（供 VLM 复核上下文使用）
                history = self.track_label_history.setdefault(track_id, deque(maxlen=16))
                history.append((label, float(confidence)))

        label_sources: dict[str, str] = {}
        for state in track_states:
            track = state["track"]
            track_id = str(state["track_id"])
            keypoints = state["keypoints"]
            label = str(state["label"])
            confidence = float(state["confidence"])
            base_label = label
            base_confidence = confidence
            # 归因：初始来源——批次内标记；非批次 track 为缓存沿用（区分是否源自 VLM 回写）
            src_tag = batch_label_sources.get(track_id)
            if src_tag is None:
                src_tag = "vlm_async" if track_id in self._vlm_cache_sourced else "cache"

            if base_label == "H01":
                h01_h04_mapping["base_h01"] += 1
                h01_h04_mapping["base_h01_conf_sum"] += base_confidence
            elif base_label == "H04":
                h01_h04_mapping["base_h04"] += 1
                h01_h04_mapping["base_h04_conf_sum"] += base_confidence

            # 使用轻量规则推理（PoseClassifier 加载后仅位置规则）
            if self.action_recognizer.pose_classifier_loaded:
                refined_label, refined_confidence, source, rule_tags = self.behavior_reasoner.refine_light(
                    track_id=track_id,
                    bbox=track.get("bbox", [0, 0, 0, 0]),
                    keypoints=keypoints,
                    frame_shape=frame.shape[:2],
                    base_label=label,
                    base_confidence=confidence,
                    peer_tracks=tracks,
                    frame_id=frame_id,
                )
            else:
                refined_label, refined_confidence, source, rule_tags = self.behavior_reasoner.infer(
                    track_id=track_id,
                    bbox=track.get("bbox", [0, 0, 0, 0]),
                    keypoints=keypoints,
                    frame_shape=frame.shape[:2],
                    base_label=label,
                    base_confidence=confidence,
                    peer_tracks=tracks,
                    frame_id=frame_id,
                )
            label, confidence = refined_label, refined_confidence
            if source == "rule_engine":
                rule_override_count += 1
                src_tag = "rule"  # 归因：规则引擎覆盖

            # 归因修复 + 双钥匙：规则触发的高危类必须经 VLM 复核；M01 需"规则候选 ∧ VLM确认"双钥匙
            if label in self.force_vlm_classes and source == "rule_engine":
                skey = self._spatial_key(track.get("bbox", [0, 0, 0, 0]))
                veto_map = {**self._vlm_veto_spatial.get(skey, {}),
                             **self._vlm_veto.get(track_id, {})}
                cached = self.track_vlm_review_cache.get(track_id)
                if frame_id < int(veto_map.get(label, 0)):
                    # 锁存期内：VLM 近期已否决该类 → 压制规则输出
                    label, confidence = base_label, base_confidence
                    src_tag = "vlm_veto"
                elif cached is not None and (frame_id - cached[0]) < self.force_vlm_throttle_frames:
                    if cached[1] != label:
                        # 复核缓存近期否决 → 登记锁存并压制
                        veto_frames = int(self._vlm_veto_seconds * max(1.0, self._source_fps))
                        self._vlm_veto.setdefault(track_id, {})[label] = frame_id + veto_frames
                        label, confidence = base_label, base_confidence
                        src_tag = "vlm_veto"
                    # cached[1] == label → 复核已确认，放行
                else:
                    # 提交规则触发的复核（claimed = 规则标签，复用 person-crop 通道）
                    self._submit_async_vlm_review(track, frame, frame_id, label, confidence,
                                                  context=self._build_review_context(track_id))
                # 双钥匙 A：确认窗口取 track 与空间键的 max（任一有效即放行，抗 ID 切换与判定抖动）
                hist = list(self._review_history.get(track_id, []))
                confirmed_until = max(int(self._vlm_confirmed.get(track_id, 0)),
                                      int(self._vlm_confirmed_spatial.get(skey, 0)))
                if frame_id > confirmed_until:
                    recent = hist[-3:]
                    if sum(1 for x in recent if x == "M01") >= 2:
                        hold = int(self._vlm_confirm_hold_seconds * max(1.0, self._source_fps))
                        self._vlm_confirmed[track_id] = frame_id + hold  # 自续期
                        self._vlm_confirmed_spatial[skey] = max(frame_id + hold,
                                                                int(self._vlm_confirmed_spatial.get(skey, 0)))
                    else:
                        label, confidence = base_label, base_confidence
                        src_tag = "rule_pending"
            reasoner_source_counts[source] = int(reasoner_source_counts.get(source, 0)) + 1
            # edge 时刻④：决策链定论步（final = 规则精化+复核制衡后的最终标签）
            if self._edge_mode:
                tr = self._frame_traces.get(track_id)
                if tr is not None:
                    tr.add("final", label, confidence)
            for tag in rule_tags:
                reasoner_rule_tag_counts[tag] = int(reasoner_rule_tag_counts.get(tag, 0)) + 1

            # ── C2: 时序标签平滑（多数投票，基于 refine 后的最终标签） ──
            smooth_hist = self.track_smooth_history.get(track_id)
            if smooth_hist is None:
                smooth_hist = deque(maxlen=self._smooth_window)
                self.track_smooth_history[track_id] = smooth_hist
            smooth_hist.append(label)
            if len(smooth_hist) >= 3:
                vote_counts = Counter(smooth_hist)
                top_label, top_count = vote_counts.most_common(1)[0]
                if top_count >= self._smooth_min_votes and label != top_label:
                    label = top_label
                    confidence = max(confidence, 0.6)
                    src_tag += "+smooth"  # 归因：C2 平滑改判
            label_sources[track_id] = src_tag

            if label == "H01":
                h01_h04_mapping["refined_h01"] += 1
                h01_h04_mapping["refined_h01_conf_sum"] += confidence
            elif label == "H04":
                h01_h04_mapping["refined_h04"] += 1
                h01_h04_mapping["refined_h04_conf_sum"] += confidence
            risk_level = RISK_MAP.get(label, "LOW")

            if confidence > main_confidence:
                main_label, main_confidence = label, confidence
            if label != "Normal" and risk_level in ("HIGH", "MEDIUM"):
                self._recent_abnormal[track_id] = {
                    "track_id": track_id, "label": label, "confidence": confidence,
                    "bbox": list(track.get("bbox", [0, 0, 0, 0])), "frame_id": frame_id,
                }
            pending_tracks.append(
                {
                    "track_id": track_id,
                    "bbox": list(track["bbox"]),
                    "keypoints": keypoints,
                    "label": label,
                    "confidence": confidence,
                    "risk_level": risk_level,
                    "activity_score": round(max(0.0, confidence - 0.45), 3),
                }
            )

        ctx["label_sources"] = label_sources
        role_map = self.interaction_analyzer.infer_roles(pending_tracks)

        temporal_result = self.temporal_adjudicator.update(
            frame_id=frame_id,
            pending_tracks=pending_tracks,
            frame_shape=frame.shape[:2],
            current_label=main_label,
            current_confidence=main_confidence,
        )
        main_label = str(temporal_result.get("output_label", main_label))
        main_confidence = float(temporal_result.get("output_confidence", main_confidence))

        for item in pending_tracks:
            role = role_map.get(item["track_id"], "unknown")
            detections.append(
                Detection(
                    track_id=item["track_id"],
                    bbox=item["bbox"],
                    pose_keypoints=item["keypoints"],
                    behavior=Behavior(
                        type=item["label"],
                        confidence=item["confidence"],
                        risk_level=item["risk_level"],
                        role=role,
                    ),
                )
            )

        stage_post_t0 = time.perf_counter()

        # ── 场景理解旁路（非阻塞）：环形缓冲入队 + 周期叙述 + 告警裁决进度 + 规则引擎 ──
        self.frame_buffer.push(frame, frame_id)
        _orig = ctx.get("orig_frame")
        if _orig is not None:
            self.frame_buffer_orig.push(_orig, frame_id)  # 问题1：原帧复核通道
        # A: 检测事实注入——当前快照 + 近窗异常聚合（叙述描述的是时间窗，参考也按窗口给，抗瞬时波动）
        window_seconds = float(getattr(self.scene_narrator, "interval_seconds", 20))
        window_frames = int(window_seconds * self._source_fps) if getattr(self, "_source_fps", None) else int(window_seconds * 25)
        injection = []
        for item in pending_tracks:
            entry = {"track_id": item["track_id"], "label": item["label"],
                     "confidence": item["confidence"], "bbox": item["bbox"]}
            recent = self._recent_abnormal.get(item["track_id"])
            if recent is not None and frame_id - int(recent.get("frame_id", 0)) <= window_frames:
                entry["label"] = recent["label"]
                entry["confidence"] = recent["confidence"]
            injection.append(entry)
        # ID 切换场景：近窗异常 track 若已不在当前帧（被重新编号），其事实仍须注入
        covered = {e["track_id"] for e in injection}
        for tid, recent in self._recent_abnormal.items():
            if tid in covered:
                continue
            if frame_id - int(recent.get("frame_id", 0)) <= window_frames:
                injection.append({
                    "track_id": tid, "label": recent["label"],
                    "confidence": recent["confidence"], "bbox": recent.get("bbox", [0, 0, 0, 0]),
                })
        self.scene_narrator.on_frame(frame_id, person_count=len(tracks),
                                     detections=injection, frame_shape=frame.shape[:2])
        self.alert_adjudicator.on_frame(frame_id)
        self.nl_rule_engine.on_frame(frame_id)
        self.novel_behavior_memo.on_frame(frame_id, pending_tracks)

        current_track_ids = {track["track_id"] for track in tracks}
        stale_ids = [track_id for track_id in self.pose_cache if track_id not in current_track_ids]
        for track_id in stale_ids:
            self.pose_cache.pop(track_id, None)
            self.pose_last_bbox.pop(track_id, None)
            self.pose_last_frame.pop(track_id, None)
        stale_action_ids = [track_id for track_id in self.action_cache if track_id not in current_track_ids]
        for track_id in stale_action_ids:
            self.action_cache.pop(track_id, None)
        stale_seen_ids = [track_id for track_id in self.track_seen_frames if track_id not in current_track_ids]
        for track_id in stale_seen_ids:
            self.track_seen_frames.pop(track_id, None)
        stale_prev_bbox_ids = [track_id for track_id in self._track_prev_bbox if track_id not in current_track_ids]
        for track_id in stale_prev_bbox_ids:
            self._track_prev_bbox.pop(track_id, None)
        stale_smooth_ids = [track_id for track_id in self.track_smooth_history if track_id not in current_track_ids]
        for track_id in stale_smooth_ids:
            self.track_smooth_history.pop(track_id, None)
        stale_history_ids = [track_id for track_id in self.track_label_history if track_id not in current_track_ids]
        for track_id in stale_history_ids:
            self.track_label_history.pop(track_id, None)
        stale_vlm_ids = [track_id for track_id in self.track_vlm_review_cache if track_id not in current_track_ids]
        for track_id in stale_vlm_ids:
            self.track_vlm_review_cache.pop(track_id, None)
            self._vlm_cache_sourced.discard(track_id)
            self._vlm_veto.pop(track_id, None)
            # 确认窗口/复核历史豁免：ID 切换时人的确认状态应延续（空间键兜底），只清易污染的序列缓存
            self._review_history.pop(track_id, None)
            self._veto_history.pop(track_id, None)
        self.action_recognizer.clear_stale_tracks(current_track_ids)
        self.behavior_reasoner.clear_stale_tracks(current_track_ids)

        self._update_pending_confirmations(track_states, frame_id, frame)
        alert = self.alert_system.build_alert(main_label, main_confidence, frame_id, timestamp=time.time(),
                                              frame=frame)
        fast_triggered = False
        update_triggered = False
        if not alert["triggered"]:
            fast_alert = self._fast_fall_trigger(track_states, frame_id, frame)
            if fast_alert is not None:
                alert = fast_alert
                fast_triggered = True
        if not alert["triggered"] and self._pending_alert_updates:
            alert = self._pending_alert_updates.popleft()
            update_triggered = True
        if alert["triggered"]:
            self._attach_event_facts(alert, frame_id, detections)
            self._mqtt_publish_alert(alert, frame_id)
            # fast_trigger/确认更新是板端闭环，不送 VLM/裁决；base 告警仍走原裁决链。
            if not fast_triggered and not update_triggered:
                if not self.alert_adjudicator.submit(alert, frame_id):
                    self._adjudicate_submit_dropped += 1

        # PoseClassifier 未加载时，使用旧 VLM 复核路径作为后备
        if not self.action_recognizer.pose_classifier_loaded:
            if self.vlm_reviewer.should_review(main_label, main_confidence):
                review = self.vlm_reviewer.review(
                    {
                        "event_id": f"evt_{frame_id}",
                        "behavior_candidates": [{"type": main_label, "confidence": main_confidence}],
                        "risk_level": RISK_MAP.get(main_label, "LOW"),
                    }
                )
                vlm_review = VLMReview(
                    enabled=True,
                    status=review["review_status"],
                    reviewed_behavior=review["reviewed_behavior"]["type"],
                    review_confidence=review["reviewed_behavior"]["confidence"],
                    reason_tags=review["reason_tags"],
                    latency_ms=review["latency_ms"],
                )
                if alert["triggered"]:
                    alert["final_decision"] = "confirmed" if review["review_status"] == "accepted" else "pending"

        stage_post_ms = (time.perf_counter() - stage_post_t0) * 1000.0
        pipeline_compute_ms = (time.perf_counter() - start) * 1000.0

        scene_narration = self.scene_narrator.consume_pending_push()
        alert_adjudication = self.alert_adjudicator.consume_pending_push()

        # 自然语言规则引擎：闭集映射逐帧匹配（零成本）+ 推送队列
        rule_hits: list[dict] = []
        for item in pending_tracks:
            rule_hits.extend(
                self.nl_rule_engine.match_closed(item["label"], item["confidence"], frame_id)
            )
        detections_snapshot = [
            {
                "track_id": item["track_id"],
                "bbox": item["bbox"],
                "label": item["label"],
                "confidence": item["confidence"],
            }
            for item in pending_tracks
        ]
        rule_hits = self.nl_rule_engine.commit_hits(rule_hits, frame_id, detections_snapshot)
        if scene_narration:
            self.nl_rule_engine.note_activities(scene_narration.get("notable_activities", []), frame_id)
            self.novel_behavior_memo.note_narrator_activities(scene_narration.get("notable_activities", []), frame_id)
        rule_alerts = self.nl_rule_engine.consume_pending_pushes()
        # 规则告警接入裁决：旧实现只有主告警进裁决器，规则事件永远停在 pending（WEAKNESS_AUDIT #1）
        for rule_hit in rule_alerts:
            rid = str(rule_hit.get("rule_id", "")).strip()
            if not rid:
                continue
            rule_alert = {
                "type": "nl_rule_hit",
                "source": "nl_rule",
                "rule_id": rid,
                "rule_text": str(rule_hit.get("rule_text", "")),
                "label": str(rule_hit.get("matched_class", "")),
                "confidence": rule_hit.get("confidence"),
            }
            self._attach_event_facts(rule_alert, frame_id, detections)
            self._mqtt_publish_alert(rule_alert, frame_id)
            if not self.alert_adjudicator.submit(
                rule_alert, frame_id,
                event_filename=f"event_{int(frame_id)}_rule_{rid}.json",
            ):
                self._adjudicate_submit_dropped += 1
        ctx["detections"] = detections
        ctx["vlm_review"] = vlm_review
        ctx["main_label"] = main_label
        ctx["main_confidence"] = main_confidence
        ctx["stage_action_ms"] = stage_action_ms
        ctx["stage_post_ms"] = stage_post_ms
        ctx["pipeline_compute_ms"] = pipeline_compute_ms
        ctx["alert"] = alert
        ctx["action_runs"] = action_runs
        ctx["scene_narration"] = scene_narration
        ctx["alert_adjudication"] = alert_adjudication
        ctx["rule_alerts"] = rule_alerts
        # edge 时刻④：本帧决策链随 FrameResult 下发（WS 全量序列化自动带出）
        ctx["decision_traces"] = [t.to_dict() for t in self._frame_traces.values()] if self._edge_mode else []
        ctx["rule_override_count"] = rule_override_count
        ctx["reasoner_source_counts"] = reasoner_source_counts
        ctx["reasoner_rule_tag_counts"] = reasoner_rule_tag_counts
        ctx["h01_h04_mapping"] = h01_h04_mapping
        ctx["action_skip_warmup"] = action_skip_warmup
        ctx["action_skip_track_age"] = action_skip_track_age
        ctx["action_skip_gate"] = action_skip_gate
        ctx["action_skip_cached"] = action_skip_cached
        ctx["action_skip_default"] = action_skip_default
        ctx["temporal_result"] = temporal_result
        ctx["pose_skip_motion_gate"] = pose_skip_motion_gate

    def _assemble_result(self, ctx: dict) -> FrameResult:
        """阶段4：组装 FrameResult（含调试指标）。"""
        frame_id = ctx["frame_id"]
        detections = ctx["detections"]
        alert = ctx["alert"]
        vlm_review = ctx["vlm_review"]
        scene_narration = ctx["scene_narration"]
        alert_adjudication = ctx["alert_adjudication"]
        rule_alerts = ctx["rule_alerts"]
        stage_detector_ms = ctx["stage_detector_ms"]
        stage_pose_ms = ctx["stage_pose_ms"]
        stage_action_ms = ctx["stage_action_ms"]
        stage_post_ms = ctx["stage_post_ms"]
        pipeline_compute_ms = ctx["pipeline_compute_ms"]
        action_runs = ctx["action_runs"]
        pose_skip_motion_gate = ctx["pose_skip_motion_gate"]
        rule_override_count = ctx["rule_override_count"]
        reasoner_source_counts = ctx["reasoner_source_counts"]
        reasoner_rule_tag_counts = ctx["reasoner_rule_tag_counts"]
        h01_h04_mapping = ctx["h01_h04_mapping"]
        action_skip_warmup = ctx["action_skip_warmup"]
        action_skip_track_age = ctx["action_skip_track_age"]
        action_skip_gate = ctx["action_skip_gate"]
        action_skip_cached = ctx["action_skip_cached"]
        action_skip_default = ctx["action_skip_default"]
        temporal_result = ctx["temporal_result"]
        label_sources = ctx.get("label_sources", {})
        decision_traces = ctx.get("decision_traces", [])
        # T3.x 语言层输入快照：最近处理帧逐目标终态（点谁说谁/问答/场景事实共用）
        self.last_track_snapshot = [
            {"track_id": d.track_id, "bbox": list(d.bbox),
             "label": d.behavior.type, "confidence": d.behavior.confidence}
            for d in detections]
        self._last_frame_id = frame_id
        result = FrameResult(
            timestamp=datetime.utcnow().isoformat() + "Z",
            frame_id=frame_id,
            detections=detections,
            alert=alert,
            vlm_review=vlm_review,
            scene_narration=scene_narration,
            alert_adjudication=alert_adjudication,
            rule_alerts=rule_alerts,
            decision_traces=decision_traces,
            metrics={
                "pipeline_latency_ms": int(pipeline_compute_ms),
                "label_sources": label_sources,  # 归因诊断：各 track 最终标签来源
                "pipeline_stage_ms": {
                    "detector": round(stage_detector_ms, 3),
                    "pose": round(stage_pose_ms, 3),
                    "action": round(stage_action_ms, 3),
                    "post": round(stage_post_ms, 3),
                    "sum_stages": round(stage_detector_ms + stage_pose_ms + stage_action_ms + stage_post_ms, 3),
                    "pipeline_compute": round(pipeline_compute_ms, 3),
                },
                "alert_adjudicate_dropped": int(self._adjudicate_submit_dropped),
                "scene_debug": {
                    "narrator": self.scene_narrator.status(),
                    "adjudicator": self.alert_adjudicator.status(),
                    "nl_rule_engine": self.nl_rule_engine.status(),
                    "novel_memo": self.novel_behavior_memo.status(),
                    "prototypes": self.behavior_prototypes.status(),
                },
                "model_load_status": {
                    "detector": self.detector.is_real_model_loaded,
                    "pose": self.pose_estimator.is_real_model_loaded,
                    "action": self.action_recognizer.is_real_model_loaded,
                    "pose_classifier": self.action_recognizer.pose_classifier_loaded,
                    "decision_mode": self.decision_mode,
                    "crop_arbiter": (self.crop_classifier.is_real_model_loaded
                                     if self.crop_classifier is not None else None),
                },
                "detector_debug": {
                    "used_fallback": self.detector.get_debug_info().get("used_fallback", True),
                    "raw_output_shape": self.detector.get_debug_info().get("raw_output_shape"),
                    "effective_person_boxes": self.detector.get_debug_info().get("effective_person_boxes", 0),
                    "error": self.detector.get_debug_info().get("error", ""),
                },
                "pose_debug": {
                    "used_fallback": self.pose_estimator.get_debug_info().get("used_fallback", True),
                    "raw_output_shape": self.pose_estimator.get_debug_info().get("raw_output_shape"),
                    "num_keypoints": self.pose_estimator.get_debug_info().get("num_keypoints", 0),
                    "error": self.pose_estimator.get_debug_info().get("error", ""),
                    "pose_stride": self.pose_stride,
                    "max_pose_tracks": self.max_pose_tracks,
                    "pose_cache_size": len(self.pose_cache),
                    "pose_motion_gate_iou_threshold": self.pose_motion_gate_iou_threshold,
                    "pose_skip_motion_gate": pose_skip_motion_gate,
                },
                "action_debug": {
                    "used_fallback": self.action_recognizer.get_debug_info().get("used_fallback", True),
                    "pose_classifier_loaded": self.action_recognizer.pose_classifier_loaded,
                    "raw_output_shape": self.action_recognizer.get_debug_info().get("raw_output_shape"),
                    "top1_label": self.action_recognizer.get_debug_info().get("top1_label", "Normal"),
                    "top1_score": self.action_recognizer.get_debug_info().get("top1_score", 0.0),
                    "buffer_len": self.action_recognizer.get_debug_info().get("buffer_len", 0),
                    "error": self.action_recognizer.get_debug_info().get("error", ""),
                    "mapping_mode": self.action_recognizer.get_debug_info().get("mapping_mode", "unknown"),
                    "external_top1_label": self.action_recognizer.get_debug_info().get("external_top1_label", ""),
                    "source_label_count": self.action_recognizer.get_debug_info().get("source_label_count", 0),
                    "action_stride": self.action_stride,
                    "max_action_tracks": self.max_action_tracks,
                    "action_min_track_score": self.action_min_track_score,
                    "action_warmup_frames": self.action_warmup_frames,
                    "action_min_track_age": self.action_min_track_age,
                    "action_runs": action_runs,
                    "action_skip_warmup": action_skip_warmup,
                    "action_skip_track_age": action_skip_track_age,
                    "action_skip_gate": action_skip_gate,
                    "action_skip_cached": action_skip_cached,
                    "action_skip_default": action_skip_default,
                    "rule_override_count": rule_override_count,
                    "reasoner_source_counts": reasoner_source_counts,
                    "reasoner_rule_tag_counts": reasoner_rule_tag_counts,
                    "h01_h04_mapping_summary": {
                        "base_h01": h01_h04_mapping["base_h01"],
                        "base_h04": h01_h04_mapping["base_h04"],
                        "refined_h01": h01_h04_mapping["refined_h01"],
                        "refined_h04": h01_h04_mapping["refined_h04"],
                        "base_h01_avg_conf": round(h01_h04_mapping["base_h01_conf_sum"] / max(1, h01_h04_mapping["base_h01"]), 4),
                        "base_h04_avg_conf": round(h01_h04_mapping["base_h04_conf_sum"] / max(1, h01_h04_mapping["base_h04"]), 4),
                        "refined_h01_avg_conf": round(h01_h04_mapping["refined_h01_conf_sum"] / max(1, h01_h04_mapping["refined_h01"]), 4),
                        "refined_h04_avg_conf": round(h01_h04_mapping["refined_h04_conf_sum"] / max(1, h01_h04_mapping["refined_h04"]), 4),
                    },
                },
                "temporal_adjudicator_debug": temporal_result.get("debug", {}),
            },
        )

        if result.alert.triggered:
            self.alert_system.write_event(result.model_dump())

        return result

    # ── B6: 管道模式接口 ──────────────────────────────────────

    def _pipeline_worker(self, stage_no: int) -> None:
        q_in = {1: self._pipe_q1, 2: self._pipe_q2, 3: self._pipe_q3}[stage_no]
        q_out = {1: self._pipe_q2, 2: self._pipe_q3, 3: None}[stage_no]
        while True:
            ctx = q_in.get()
            if ctx is None:
                return
            if ctx.get("generation") != self._pipe_generation:
                continue  # 上一任务残留帧，丢弃
            t0 = time.perf_counter()
            try:
                if stage_no == 1:
                    self._stage1_detect(ctx)
                elif stage_no == 2:
                    self._stage2_pose(ctx)
                else:
                    self._stage3_recognize(ctx)
                    self._stage_action_samples.append(float(ctx.get("stage_action_ms", 0.0)))
                    self._stage_post_samples.append(float(ctx.get("stage_post_ms", 0.0)))
                    t_assemble = time.perf_counter()
                    result = self._assemble_result(ctx)
                    self._assemble_samples.append((time.perf_counter() - t_assemble) * 1000.0)
                    with self._pipe_results_lock:
                        if ctx.get("generation") == self._pipe_generation:
                            self._pipe_results.append(result)
                    continue
            except Exception as exc:
                # 该帧丢弃（与历史"忙时丢帧"语义一致），不阻塞后续帧
                self._last_pipeline_error = f"stage{stage_no}: {type(exc).__name__}: {exc}"
                continue
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                self._pipe_stage_ms[stage_no] += dt_ms
                self._pipe_stage_max[stage_no] = max(self._pipe_stage_max[stage_no], dt_ms)
                self._pipe_stage_count[stage_no] += 1
                self._pipe_stage_samples[stage_no].append(dt_ms)
            if q_out is not None:
                q_out.put(ctx)

    def submit_frame(self, frame, frame_id: int, orig_frame=None) -> bool:
        """B6 管道模式：投递一帧，立即返回。False = 队列满被丢弃。"""
        ctx = self._make_ctx(frame, frame_id)
        if orig_frame is not None:
            ctx["orig_frame"] = orig_frame
        ctx["generation"] = self._pipe_generation
        try:
            self._pipe_q1.put_nowait(ctx)
            return True
        except Exception:
            self._pipe_dropped += 1
            return False

    def drain_results(self) -> list[FrameResult]:
        """B6 管道模式：取走所有已完成 FrameResult（FIFO，按帧序）。"""
        with self._pipe_results_lock:
            out = list(self._pipe_results)
            self._pipe_results.clear()
        return out

    def pipeline_stats(self) -> dict:
        def _qsize(name):
            q = getattr(self, name, None)
            return q.qsize() if q is not None else 0

        def _pct(vals, q):
            s = sorted(vals)
            if not s:
                return 0.0
            return round(s[min(len(s) - 1, int(q * (len(s) - 1)))], 3)

        stage_samples = getattr(self, "_pipe_stage_samples", {1: (), 2: (), 3: ()})
        assemble_samples = getattr(self, "_assemble_samples", ())
        action_samples = getattr(self, "_stage_action_samples", ())
        post_samples = getattr(self, "_stage_post_samples", ())
        return {
            "enabled": self._pipeline_enabled,
            "dropped": self._pipe_dropped,
            "pending_results": len(self._pipe_results),
            "q1": _qsize("_pipe_q1"),
            "q2": _qsize("_pipe_q2"),
            "q3": _qsize("_pipe_q3"),
            "last_error": self._last_pipeline_error,
            "stage_ms": {
                str(k): {
                    "avg_ms": round(self._pipe_stage_ms[k] / max(1, self._pipe_stage_count[k]), 3),
                    "p50_ms": _pct(stage_samples.get(k, ()), 0.50),
                    "p95_ms": _pct(stage_samples.get(k, ()), 0.95),
                    "max_ms": round(self._pipe_stage_max[k], 3),
                    "count": self._pipe_stage_count[k],
                }
                for k in (1, 2, 3)
            },
            "assemble_ms": {
                "p50_ms": _pct(assemble_samples, 0.50),
                "p95_ms": _pct(assemble_samples, 0.95),
                "max_ms": round(max(assemble_samples), 3) if assemble_samples else 0.0,
                "count": len(assemble_samples),
            },
            "action_ms": {
                "p50_ms": _pct(action_samples, 0.50),
                "p95_ms": _pct(action_samples, 0.95),
                "count": len(action_samples),
            },
            "action_pc": {
                "inflight": bool(self._action_pc_inflight),
                "runs": len(self._action_pc_async_samples),
                "async_ms_p50": _pct(self._action_pc_async_samples, 0.50),
                "async_ms_p95": _pct(self._action_pc_async_samples, 0.95),
            },
            "post_ms": {
                "p50_ms": _pct(post_samples, 0.50),
                "p95_ms": _pct(post_samples, 0.95),
                "count": len(post_samples),
            },
            "pose": {
                "stride": self.pose_stride,
                "max_tracks": self.max_pose_tracks,
                "batch_runs": self._pose_batch_runs,
                "cache_size": len(self.pose_cache),
                "track_seen_size": len(self.track_seen_frames),
            },
        }

