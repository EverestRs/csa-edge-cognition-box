from __future__ import annotations

"""感知事实编译器：引擎状态 → 结构化事实。

反幻觉红线（01 §2.5 / 03 §3 时刻②）：语言层只允许消费本模块产出的事实字段，
不得触碰图像；旁白合成 prompt 中显式约束"仅复述事实"。
纯 stdlib（不依赖 numpy/pydantic），输入为普通 dict/list，便于单测与引擎解耦。
"""

import time
from typing import Any

# 内置类别中文名（与 nl_rule_engine PARSE_PROMPT 口径一致）
CLASS_CN = {
    "Normal": "正常活动", "H01": "打架斗殴", "H02": "校园霸凌", "H03": "跌倒",
    "H04": "疑似轻生", "H05": "破坏设施", "M01": "吸烟", "M02": "使用手机",
    "M03": "遮挡摄像头", "S01": "徘徊", "S02": "闯入禁区",
    # crop 演示扩展类
    "Wave": "挥手", "Clap": "鼓掌", "Throw": "投掷", "Handshake": "握手",
}


def _bbox_stats(bboxes: list[list[float]]) -> dict[str, Any]:
    """bbox 序列统计：[x,y,w,h] 列表 → 中心位移/均面积/端点。"""
    if not bboxes:
        return {}
    centers = [(b[0] + b[2] / 2.0, b[1] + b[3] / 2.0) for b in bboxes if len(b) >= 4]
    if not centers:
        return {}
    x0, y0 = centers[0]
    x1, y1 = centers[-1]
    shift = round(float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5), 1)
    path = sum(
        float(((centers[i][0] - centers[i - 1][0]) ** 2 +
               (centers[i][1] - centers[i - 1][1]) ** 2) ** 0.5)
        for i in range(1, len(centers))
    )
    areas = [b[2] * b[3] for b in bboxes if len(b) >= 4]
    return {
        "center_shift_px": shift,
        "path_length_px": round(path, 1),
        "mean_area_px2": round(sum(areas) / len(areas)) if areas else 0,
        "start_center": [round(x0), round(y0)],
        "end_center": [round(x1), round(y1)],
    }


def build_scene_facts(*, frame_id: int, timestamp: str = "",
                      tracks: list[dict[str, Any]],
                      crop_verdicts: dict[str, str] | None = None,
                      zone_events: list[dict[str, Any]] | None = None,
                      fps: float = 0.0,
                      extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """场景级事实（scene_narrator facts 模式 / 底牌角标共用）。

    tracks: [{track_id, bbox:[x,y,w,h], label:str, confidence:float}]
    """
    crop_verdicts = crop_verdicts or {}
    behavior_list = [
        {"track_id": t.get("track_id", "?"), "label": t.get("label", "Normal"),
         "confidence": round(float(t.get("confidence", 0.0)), 3)}
        for t in tracks
    ]
    label_hist: dict[str, int] = {}
    for b in behavior_list:
        label_hist[b["label"]] = label_hist.get(b["label"], 0) + 1
    facts: dict[str, Any] = {
        "kind": "scene",
        "frame_id": int(frame_id),
        "timestamp": timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
        "person_count": len(tracks),
        "behavior_distribution": label_hist,
        "abnormal_tracks": [b for b in behavior_list if b["label"] != "Normal"],
        "crop_verified": {tid: cls for tid, cls in crop_verdicts.items() if cls and cls != "Normal"},
        "zone_events": [
            {"track_id": z.get("track_id"), "zone": z.get("zone"), "type": z.get("type", "forbidden")}
            for z in (zone_events or [])
        ],
    }
    if fps:
        facts["fps"] = round(float(fps), 2)
    if extra:
        facts.update(extra)
    return facts


def build_track_facts(*, track_id: str, frame_id: int, window_seconds: float,
                      label_history: list[tuple[str, float]] | list[str],
                      bboxes: list[list[float]],
                      crop_verdicts: list[str] | None = None,
                      zone_events: list[str] | None = None,
                      speed_px_per_s: float = 0.0) -> dict[str, Any]:
    """目标级事实（点谁说谁 track_narrator 输入）。

    label_history: 最近行为标签序列（支持 [(label, conf)] 或 [label]）。
    """
    labels: list[str] = []
    confs: list[float] = []
    for item in label_history or []:
        if isinstance(item, (tuple, list)) and item:
            labels.append(str(item[0]))
            try:
                confs.append(float(item[1]))
            except (TypeError, ValueError, IndexError):
                confs.append(0.0)
        else:
            labels.append(str(item))
            confs.append(0.0)
    hist: dict[str, int] = {}
    for lb in labels:
        hist[lb] = hist.get(lb, 0) + 1
    dominant = max(hist, key=hist.get) if hist else "Normal"
    crop_hits = [c for c in (crop_verdicts or []) if c and c != "Normal"]
    facts: dict[str, Any] = {
        "kind": "track",
        "track_id": track_id,
        "frame_id": int(frame_id),
        "observation_window_s": round(float(window_seconds), 1),
        "label_histogram": hist,
        "dominant_label": dominant,
        "label_changes": sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1]),
        "bbox_summary": _bbox_stats(bboxes),
        "speed_px_per_s": round(float(speed_px_per_s), 1),
        "zone_events": list(zone_events or []),
        "crop_confirmed": sorted(set(crop_hits)),
    }
    if confs:
        facts["mean_confidence"] = round(sum(confs) / len(confs), 3)
    return facts


def build_event_facts(*, alert: dict[str, Any], scene_facts: dict[str, Any],
                      track_facts: dict[str, Any] | None = None,
                      traces: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """告警事件事实（alert_adjudicator edge 档第一趟：事实→描述→裁决）。"""
    facts: dict[str, Any] = {
        "kind": "event",
        "event_type": alert.get("type", "unknown"),
        "trigger_label": alert.get("label") or alert.get("type"),
        "confidence": alert.get("confidence"),
        "frame_id": alert.get("frame_id"),
        "snapshot_url": alert.get("snapshot_url", ""),
        "scene": {
            "person_count": scene_facts.get("person_count", 0),
            "behavior_distribution": scene_facts.get("behavior_distribution", {}),
            "crop_verified": scene_facts.get("crop_verified", {}),
            "zone_events": scene_facts.get("zone_events", []),
        },
        "trace_digest": [
            {"track_id": t.get("track_id"), "steps": len(t.get("steps", [])),
             "final": next((s.get("label") for s in reversed(t.get("steps", []))
                            if s.get("stage") == "final"), None)}
            for t in (traces or [])
        ],
    }
    if track_facts:
        facts["track"] = track_facts
    return facts


def facts_to_text(facts: dict[str, Any]) -> dict[str, str]:
    """事实 → 确定性中文短句（LLM 不可用时的降级旁白，也是反幻觉对照基准）。"""
    kind = facts.get("kind", "scene")
    out: dict[str, str] = {}
    if kind == "track":
        dom = facts.get("dominant_label", "Normal")
        out["summary"] = (
            f"目标{facts.get('track_id', '?')}观察{facts.get('observation_window_s', 0)}秒："
            f"主要行为{CLASS_CN.get(dom, dom)}，"
            f"位移{facts.get('bbox_summary', {}).get('center_shift_px', 0)}px，"
            f"行为变化{facts.get('label_changes', 0)}次"
            + (f"，Crop确认{[' '.join(CLASS_CN.get(c, c) for c in facts['crop_confirmed'])]}"
               if facts.get("crop_confirmed") else "")
            + (f"，触发区域:{','.join(facts['zone_events'])}" if facts.get("zone_events") else "")
            + "。"
        )
    elif kind == "scene":
        dist = facts.get("behavior_distribution", {})
        parts = "、".join(f"{CLASS_CN.get(k, k)}{v}人" for k, v in dist.items())
        out["summary"] = (
            f"画面{facts.get('person_count', 0)}人：{parts or '无明确行为'}"
            + (f"；已确认行为:{','.join(CLASS_CN.get(v, v) for v in facts['crop_verified'].values())}"
               if facts.get("crop_verified") else "")
            + "。"
        )
    elif kind == "event":
        out["description"] = (
            f"事件{facts.get('event_type', '')}于帧{facts.get('frame_id', 0)}触发，"
            f"触发行为{CLASS_CN.get(str(facts.get('trigger_label')), str(facts.get('trigger_label', '')))}，"
            f"当时画面{facts.get('scene', {}).get('person_count', 0)}人。"
        )
    return out
