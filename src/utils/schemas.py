from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
RoleType = Literal["attacker", "victim", "bystander", "unknown"]


class Behavior(BaseModel):
    type: str = "Normal"
    confidence: float = 0.0
    risk_level: RiskLevel = "LOW"
    role: RoleType = "unknown"


class Detection(BaseModel):
    track_id: str
    bbox: list[int] = Field(default_factory=list)
    behavior: Behavior = Field(default_factory=Behavior)
    pose_keypoints: list[list[float]] = Field(default_factory=list)


class Alert(BaseModel):
    triggered: bool = False
    type: str = "none"
    source: str = "base_model"
    final_decision: str = "pending"
    snapshot_url: str = ""
    clip_path: str = ""


class VLMReview(BaseModel):
    enabled: bool = False
    status: str = "skipped"
    reviewed_behavior: str = ""
    review_confidence: float = 0.0
    reason_tags: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    description: str = ""


class FrameResult(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    frame_id: int
    detections: list[Detection] = Field(default_factory=list)
    alert: Alert = Field(default_factory=Alert)
    vlm_review: VLMReview = Field(default_factory=VLMReview)
    metrics: dict[str, Any] = Field(default_factory=dict)
    scene_narration: dict[str, Any] | None = None
    alert_adjudication: dict[str, Any] | None = None
    rule_alerts: list[dict[str, Any]] = Field(default_factory=list)
    # edge 决策模式（edge_migration.decision_trace）：每目标决策链，"认知视图"数据源
    decision_traces: list[dict[str, Any]] = Field(default_factory=list)


class NarrateRequest(BaseModel):
    """时刻②点谁说谁：对最近处理帧中的目标生成事实旁白"""
    track_id: str


class ChatRequest(BaseModel):
    """时刻③和录像聊天：场景记忆检索问答"""
    question: str


class AnalyzeStreamRequest(BaseModel):
    stream_url: str
    callback_url: str | None = None
    duration: int = 0
    eval_dump: bool = False  # C5: 逐帧标签落盘（事件级评估用）


class AnalyzeStreamResponse(BaseModel):
    task_id: str
    status: str
    endpoint: str


class EvaluateRequest(BaseModel):
    video_path: str
    ground_truth: str | None = None
    profile: str | None = None
    baseline: bool = False
    use_scene_context: bool = True
    frame_stride: int | None = None
    max_frames: int | None = None


class EvaluateResponse(BaseModel):
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    avg_latency_ms: int
    per_class_metrics: dict[str, Any]
    result_file: str


class TrainRequest(BaseModel):
    dataset_path: str
    behavior_types: list[str]
    pretrained_model: str
    epochs: int = 50
    batch_size: int = 8
    device: str = "auto"


class VLMReviewRequest(BaseModel):
    task_id: str
    camera_id: str
    frame_id: int
    timestamp: str
    trigger_reason: str
    candidate_event: dict[str, Any]
    latency_budget_ms: int = 60
    top_k: int = 3


class VLMReviewResponse(BaseModel):
    event_id: str
    review_status: str
    reviewed_behavior: dict[str, Any]
    role_scores: dict[str, dict[str, float]]
    reason_tags: list[str]
    summary: str
    latency_ms: int
    model_name: str
