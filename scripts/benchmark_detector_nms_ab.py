from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.detector import Detector
from src.utils.config_loader import load_yaml
from src.utils.preflight import resolve_profile


def old_compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def old_nms(self: Detector, boxes_xyxy: np.ndarray, scores: np.ndarray) -> list[int]:
    if len(boxes_xyxy) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while len(order) > 0:
        index = int(order[0])
        keep.append(index)
        if len(order) == 1:
            break

        remaining = order[1:]
        ious = np.array([old_compute_iou(boxes_xyxy[index], boxes_xyxy[j]) for j in remaining], dtype=np.float32)
        order = remaining[ious < self.iou_threshold]

    return keep


def det_signature(dets: list[dict]) -> str:
    normalized = []
    for d in dets:
        bbox = d.get("bbox", [0, 0, 0, 0])
        normalized.append(
            {
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "score": round(float(d.get("score", 0.0)), 6),
                "class": str(d.get("class", "person")),
            }
        )
    normalized.sort(key=lambda x: (x["bbox"][0], x["bbox"][1], x["bbox"][2], x["bbox"][3], -x["score"]))
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round(0.95 * (len(arr) - 1)))
    return arr[idx]


def collect_frames(video: str, max_frames: int, sample_step: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")

    frames: list[np.ndarray] = []
    frame_id = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % sample_step != 0:
            continue
        frames.append(frame)

    cap.release()
    if not frames:
        raise RuntimeError("no frames sampled")
    return frames


def run_detector(detector: Detector, frames: list[np.ndarray]) -> tuple[list[float], list[str]]:
    times_ms: list[float] = []
    signatures: list[str] = []

    # warmup
    _ = detector.infer(frames[0])

    for frame in frames:
        t0 = time.perf_counter()
        dets = detector.infer(frame)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)
        signatures.append(det_signature(dets))

    return times_ms, signatures


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B benchmark detector NMS old vs optimized")
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", default="realtime")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--sample-step", type=int, default=3)
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--output", default="results/detector_benchmark_ab.json")
    args = parser.parse_args()

    model_cfg = load_yaml(PROJECT_ROOT / args.model_config)
    merged_cfg, _ = resolve_profile(model_cfg, args.profile)
    detector_cfg = merged_cfg.get("detector", {})

    frames = collect_frames(args.video, max_frames=max(10, args.max_frames), sample_step=max(1, args.sample_step))

    det_old = Detector(detector_cfg)
    det_old._nms = old_nms.__get__(det_old, Detector)

    det_new = Detector(detector_cfg)

    old_times, old_signatures = run_detector(det_old, frames)
    new_times, new_signatures = run_detector(det_new, frames)

    mismatches = sum(1 for a, b in zip(old_signatures, new_signatures) if a != b)

    old_avg = statistics.mean(old_times)
    new_avg = statistics.mean(new_times)
    speedup = ((old_avg - new_avg) / max(1e-6, old_avg)) * 100.0

    result = {
        "video": args.video,
        "profile": args.profile,
        "sampled_frames": len(frames),
        "output_identical": mismatches == 0,
        "mismatch_frames": mismatches,
        "old_ms": {
            "avg": round(old_avg, 3),
            "p50": round(statistics.median(old_times), 3),
            "p95": round(p95(old_times), 3),
        },
        "new_ms": {
            "avg": round(new_avg, 3),
            "p50": round(statistics.median(new_times), 3),
            "p95": round(p95(new_times), 3),
        },
        "improvement_pct": round(speedup, 2),
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
