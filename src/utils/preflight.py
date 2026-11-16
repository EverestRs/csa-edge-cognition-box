from __future__ import annotations

from pathlib import Path
from typing import Any


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def resolve_profile(config: dict[str, Any], profile_name: str) -> tuple[dict[str, Any], dict[str, str]]:
    profiles = config.get("profiles", {})
    profile_cfg = profiles.get(profile_name, {})

    merged = dict(config)
    merged_inference = dict(merged.get("inference", {}))
    for key in (
        "frame_stride",
        "pose_stride",
        "max_pose_tracks",
        "action_stride",
        "max_action_tracks",
        "action_min_track_score",
        "action_warmup_frames",
        "action_min_track_age",
        "action_default_label",
        "action_default_confidence",
        "max_fps",
        "startup_no_drop_seconds",
        "input_size",
        "push_stride",
        "jpeg_quality",
        "ws_sleep_ms",
        "source_resize",
        "send_pose_keypoints",
        "show_pose_overlay",
        "replay_buffer_frames",
        "replay_from_task_start",
        # edge 决策模式（edge_migration/01 §2.4）与检测门控
        "decision_mode",
        "slowfast_enabled",
        "detector_stride",
        "detector_max_skip_frames",
        "enable_detector_gating",
        # 高吞吐档可关异步三阶段管道，走串行检测（避免满帧投递时队列积压失速）
        "pipeline",
    ):
        if key in profile_cfg:
            merged_inference[key] = profile_cfg[key]
    merged["inference"] = merged_inference

    detector_cfg = dict(merged.get("detector", {}))
    pose_cfg = dict(merged.get("pose", {}))
    action_cfg = dict(merged.get("action", {}))
    temporal_cfg = dict(merged.get("temporal_adjudicator", {}))

    if "detector_conf_threshold" in profile_cfg:
        detector_cfg["confidence_threshold"] = profile_cfg["detector_conf_threshold"]

    if "pose_keypoint_keep_indices" in profile_cfg:
        pose_cfg["keypoint_keep_indices"] = profile_cfg["pose_keypoint_keep_indices"]

    resolved_model_paths = {
        "detector_onnx": profile_cfg.get("detector_onnx_path") or detector_cfg.get("onnx_path", ""),
        "pose_onnx": profile_cfg.get("pose_onnx_path") or pose_cfg.get("onnx_path", ""),
        "action_onnx": profile_cfg.get("action_onnx_path") or action_cfg.get("onnx_path", ""),
    }

    detector_cfg["active_onnx_path"] = resolved_model_paths["detector_onnx"]
    pose_cfg["active_onnx_path"] = resolved_model_paths["pose_onnx"]
    action_cfg["active_onnx_path"] = resolved_model_paths["action_onnx"]

    if isinstance(profile_cfg.get("temporal_adjudicator"), dict):
        temporal_cfg.update(dict(profile_cfg.get("temporal_adjudicator", {})))

    merged["detector"] = detector_cfg
    merged["pose"] = pose_cfg
    merged["action"] = action_cfg
    merged["temporal_adjudicator"] = temporal_cfg

    return merged, resolved_model_paths


def run_preflight(base_dir: Path, config: dict[str, Any], profile_name: str) -> dict[str, Any]:
    merged, resolved_model_paths = resolve_profile(config, profile_name)

    checks: list[dict[str, Any]] = []

    model_checks = {
        "detector_onnx": resolved_model_paths.get("detector_onnx", ""),
        "pose_onnx": resolved_model_paths.get("pose_onnx", ""),
        "action_onnx": resolved_model_paths.get("action_onnx", ""),
        "tracker_reid": merged.get("tracker", {}).get("reid_model_path", ""),
    }

    label_checks = {
        "detector_labels": merged.get("detector", {}).get("labels_path", ""),
        "pose_labels": merged.get("pose", {}).get("labels_path", ""),
        "action_labels": merged.get("action", {}).get("labels_path", ""),
    }

    for key, value in {**model_checks, **label_checks}.items():
        resolved = _resolve_path(base_dir, value) if value else None
        exists = bool(resolved and resolved.exists())
        checks.append(
            {
                "name": key,
                "configured_path": value,
                "resolved_path": str(resolved) if resolved else "",
                "exists": exists,
            }
        )

    missing = [item for item in checks if not item["exists"]]
    return {
        "profile": profile_name,
        "summary": {
            "total": len(checks),
            "ok": len(checks) - len(missing),
            "missing": len(missing),
        },
        "checks": checks,
    }
