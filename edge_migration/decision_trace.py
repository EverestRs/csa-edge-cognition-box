from __future__ import annotations

"""决策链路 trace 数据结构（时刻④"让 AI 说出自己的思考"的数据源）。

纪律：trace 只由真实决策路径写入（edge 模式 inference_engine 挂载点，T2.6），
呈现层只渲染、禁止编造。FrameResult 集成时新增字段：
    decision_traces: list[dict[str, Any]] = Field(default_factory=list)
"""

from dataclasses import dataclass, field
from typing import Any

STAGE_LABELS = {
    "pose_classifier": "主路径",
    "crop_arbiter": "Crop复核",
    "rule": "规则",
    "slowfast": "兜底",
    "final": "定论",
    "rule_hit": "布控命中",
    "gate": "门控",
}

VALID_STAGES = set(STAGE_LABELS.keys())


@dataclass
class TraceStep:
    stage: str
    label: str
    conf: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"stage": self.stage, "label": self.label,
                             "conf": round(float(self.conf), 3)}
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass
class DecisionTrace:
    """单目标单帧的决策链。steps 按发生顺序追加，final 步必须存在才算完成。"""

    track_id: str
    frame_id: int
    steps: list[TraceStep] = field(default_factory=list)
    latency_ms: float = 0.0

    def add(self, stage: str, label: str, conf: float = 0.0, **detail: Any) -> "DecisionTrace":
        if stage not in VALID_STAGES:
            raise ValueError(f"unknown trace stage: {stage}")
        self.steps.append(TraceStep(stage=stage, label=str(label), conf=float(conf),
                                    detail=dict(detail)))
        return self

    @property
    def final_label(self) -> str | None:
        for s in reversed(self.steps):
            if s.stage == "final":
                return s.label
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_id": int(self.frame_id),
            "steps": [s.to_dict() for s in self.steps],
            "latency_ms": round(float(self.latency_ms), 2),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DecisionTrace":
        t = cls(track_id=str(d.get("track_id", "")), frame_id=int(d.get("frame_id", 0)),
                latency_ms=float(d.get("latency_ms", 0) or 0))
        for s in d.get("steps", []) or []:
            t.steps.append(TraceStep(stage=str(s.get("stage", "gate")),
                                     label=str(s.get("label", "")),
                                     conf=float(s.get("conf", 0) or 0.0),
                                     detail=dict(s.get("detail", {}) or {})))
        return t


def is_valid_trace(d: dict[str, Any]) -> bool:
    """结构校验（03 测试计划 §3 时刻④的 schema 断言用）。"""
    if not isinstance(d, dict):
        return False
    if not d.get("track_id") or "frame_id" not in d:
        return False
    steps = d.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    stages = set()
    for s in steps:
        if not isinstance(s, dict) or s.get("stage") not in VALID_STAGES:
            return False
        if "label" not in s or "conf" not in s:
            return False
        stages.add(s["stage"])
    return "final" in stages


def render_trace_line(trace: dict[str, Any] | DecisionTrace, *,
                      label_cn: dict[str, str] | None = None) -> str:
    """字幕带一行文本，如：P003 f412: 主路径 M01(0.61) → Crop复核 Normal(0.78) → 定论 Normal"""
    d = trace.to_dict() if isinstance(trace, DecisionTrace) else dict(trace)
    cn = label_cn or {}
    parts = []
    for s in d.get("steps", []):
        name = STAGE_LABELS.get(s["stage"], s["stage"])
        label = cn.get(s["label"], s["label"])
        parts.append(f"{name} {label}({float(s.get('conf', 0)):.2f})")
    head = f"{d.get('track_id', '?')} f{d.get('frame_id', 0)}: "
    return head + " → ".join(parts)
