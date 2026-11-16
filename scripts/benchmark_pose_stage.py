from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.detector import Detector
from src.core.pose_estimator import PoseEstimator
from src.core.tracker import Tracker
from src.utils.config_loader import load_yaml
from src.utils.preflight import resolve_profile


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(round(q * (len(values_sorted) - 1)))
    return float(values_sorted[idx])


def maybe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def clone_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)


def bbox_iou_xywh(a: list[float], b: list[float]) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pose-stage cost on a fixed video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", default="realtime")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--pose-model", default="", help="Override pose ONNX path")
    parser.add_argument("--pose-motion-gate-iou", type=float, default=-1.0)
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_yaml(PROJECT_ROOT / args.model_config)
    merged_cfg, _ = resolve_profile(cfg, args.profile)

    runtime_cfg = dict(merged_cfg.get("runtime", {}))

    detector_cfg = clone_cfg(merged_cfg.get("detector", {}))
    detector_cfg["runtime"] = runtime_cfg

    pose_cfg = clone_cfg(merged_cfg.get("pose", {}))
    pose_cfg["runtime"] = runtime_cfg
    if args.pose_model:
        pose_cfg["active_onnx_path"] = args.pose_model

    tracker_cfg = clone_cfg(merged_cfg.get("tracker", {}))

    inference_cfg = dict(merged_cfg.get("inference", {}))
    frame_stride = int(max(1, inference_cfg.get("frame_stride", 1)))
    pose_stride = int(max(1, inference_cfg.get("pose_stride", 8)))
    max_pose_tracks = int(max(1, inference_cfg.get("max_pose_tracks", 2)))
    pose_motion_gate_iou = float(inference_cfg.get("pose_motion_gate_iou_threshold", 1.01))
    if args.pose_motion_gate_iou >= 0:
        pose_motion_gate_iou = float(args.pose_motion_gate_iou)

    detector = Detector(detector_cfg)
    tracker = Tracker(tracker_cfg)
    pose_estimator = PoseEstimator(pose_cfg)

    if detector.session is None:
        raise RuntimeError(f"detector unavailable: {detector.session_init_error}")
    if pose_estimator.session is None:
        raise RuntimeError(f"pose unavailable: {pose_estimator.session_init_error}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")

    frame_id = 0
    sampled = 0
    detector_stage_ms: list[float] = []
    pose_stage_ms: list[float] = []
    pose_call_ms: list[float] = []
    total_stage_ms: list[float] = []
    pose_calls_per_frame: list[int] = []
    keypoint_count: list[int] = []
    keypoint_conf_mean: list[float] = []
    pose_skip_motion_gate = 0
    pose_cache: dict[str, list[list[float]]] = {}
    pose_last_bbox: dict[str, list[float]] = {}

    while sampled < int(max(1, args.frames)):
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        if frame_id % int(max(1, args.sample_step)) != 0:
            continue
        if frame_id % frame_stride != 0:
            continue

        sampled += 1

        t0 = time.perf_counter()
        det_t0 = time.perf_counter()
        detections_raw = detector.infer(frame)
        tracks = tracker.update(detections_raw)
        det_ms = (time.perf_counter() - det_t0) * 1000.0

        pose_ms_this_frame = 0.0
        pose_calls = 0
        pose_compute_set = set()
        if (frame_id % pose_stride) == 0:
            sorted_tracks = sorted(tracks, key=lambda item: float(item.get("score", 0.0)), reverse=True)
            pose_compute_set = {item.get("track_id") for item in sorted_tracks[:max_pose_tracks]}

        for track in tracks:
            track_id = track.get("track_id")
            should_pose_infer = (track_id in pose_compute_set) or (track_id not in pose_cache)
            if (
                should_pose_infer
                and track_id in pose_cache
                and track_id in pose_compute_set
                and pose_motion_gate_iou <= 1.0
            ):
                prev_bbox = pose_last_bbox.get(track_id)
                if prev_bbox is not None:
                    iou = bbox_iou_xywh(prev_bbox, track.get("bbox", [0, 0, 0, 0]))
                    if iou >= pose_motion_gate_iou:
                        should_pose_infer = False
                        pose_skip_motion_gate += 1

            if should_pose_infer:
                p0 = time.perf_counter()
                keypoints = pose_estimator.infer(frame, track)
                p_ms = (time.perf_counter() - p0) * 1000.0
                pose_ms_this_frame += p_ms
                pose_call_ms.append(float(p_ms))
                pose_calls += 1
                pose_cache[track_id] = keypoints
                pose_last_bbox[track_id] = list(track.get("bbox", [0, 0, 0, 0]))
            else:
                keypoints = pose_cache.get(track_id, [])

            keypoint_count.append(len(keypoints))
            conf_values = [float(pt[2]) for pt in keypoints if len(pt) >= 3]
            if conf_values:
                keypoint_conf_mean.append(sum(conf_values) / len(conf_values))

        track_ids = {track.get("track_id") for track in tracks}
        stale_ids = [tid for tid in pose_cache if tid not in track_ids]
        for tid in stale_ids:
            pose_cache.pop(tid, None)
            pose_last_bbox.pop(tid, None)

        all_ms = (time.perf_counter() - t0) * 1000.0

        detector_stage_ms.append(det_ms)
        pose_stage_ms.append(pose_ms_this_frame)
        total_stage_ms.append(all_ms)
        pose_calls_per_frame.append(pose_calls)

    cap.release()

    pose_model_used = str(pose_cfg.get("active_onnx_path") or pose_cfg.get("onnx_path") or "")

    result = {
        "video": args.video,
        "profile": args.profile,
        "sample_step": int(max(1, args.sample_step)),
        "frames_target": int(max(1, args.frames)),
        "frames_sampled": sampled,
        "frame_stride": frame_stride,
        "pose_stride": pose_stride,
        "max_pose_tracks": max_pose_tracks,
        "pose_motion_gate_iou_threshold": pose_motion_gate_iou,
        "pose_model": pose_model_used,
        "runtime": {
            "execution_provider": runtime_cfg.get("execution_provider", "CPUExecutionProvider"),
            "intra_op_num_threads": runtime_cfg.get("intra_op_num_threads", 0),
            "inter_op_num_threads": runtime_cfg.get("inter_op_num_threads", 0),
            "graph_optimization_level": runtime_cfg.get("graph_optimization_level", "ORT_ENABLE_ALL"),
        },
        "detector_stage_ms": {
            "avg": round(maybe_mean(detector_stage_ms), 3),
            "p50": round(percentile(detector_stage_ms, 0.50), 3),
            "p95": round(percentile(detector_stage_ms, 0.95), 3),
        },
        "pose_stage_ms": {
            "avg": round(maybe_mean(pose_stage_ms), 3),
            "p50": round(percentile(pose_stage_ms, 0.50), 3),
            "p95": round(percentile(pose_stage_ms, 0.95), 3),
        },
        "pose_call_ms": {
            "count": len(pose_call_ms),
            "avg": round(maybe_mean(pose_call_ms), 3),
            "p50": round(percentile(pose_call_ms, 0.50), 3),
            "p95": round(percentile(pose_call_ms, 0.95), 3),
        },
        "pose_calls_per_frame": {
            "avg": round(maybe_mean([float(v) for v in pose_calls_per_frame]), 3),
            "max": int(max(pose_calls_per_frame) if pose_calls_per_frame else 0),
            "skip_motion_gate": int(pose_skip_motion_gate),
        },
        "keypoints": {
            "avg_count": round(maybe_mean([float(v) for v in keypoint_count]), 3),
            "avg_confidence": round(maybe_mean(keypoint_conf_mean), 4),
        },
        "total_stage_ms": {
            "avg": round(maybe_mean(total_stage_ms), 3),
            "p50": round(percentile(total_stage_ms, 0.50), 3),
            "p95": round(percentile(total_stage_ms, 0.95), 3),
        },
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
