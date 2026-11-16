from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.inference_engine import InferenceEngine
from src.utils.config_loader import load_json, load_yaml
from src.utils.preflight import resolve_profile


CONFIG_DIR = BASE_DIR / "configs"


def _build_engine(profile: str) -> tuple[InferenceEngine, dict]:
    model_cfg = load_yaml(CONFIG_DIR / "model_config.yaml")
    vlm_cfg = load_yaml(CONFIG_DIR / "vlm_config.yaml")
    behavior_rules = load_json(CONFIG_DIR / "behavior_rules.json")

    if isinstance(behavior_rules, dict) and isinstance(behavior_rules.get("heuristics"), dict):
        model_cfg["behavior_heuristics"] = dict(behavior_rules["heuristics"])

    merged_model_cfg, _ = resolve_profile(model_cfg, profile)
    engine = InferenceEngine(model_cfg=merged_model_cfg, vlm_cfg=vlm_cfg, output_dir=str(BASE_DIR / "results"))
    return engine, merged_model_cfg


def _frame_signature_update(signature: hashlib._hashlib.HASH, payload: dict) -> None:
    frame_id = int(payload.get("frame_id", 0))
    detections = payload.get("detections", [])
    detections_sorted = sorted(
        detections,
        key=lambda item: (str(item.get("track_id", "")), str(item.get("behavior", {}).get("type", ""))),
    )
    compact = {
        "frame_id": frame_id,
        "detections": [
            {
                "track_id": str(item.get("track_id", "")),
                "bbox": [int(v) for v in item.get("bbox", [0, 0, 0, 0])],
                "behavior": {
                    "type": str(item.get("behavior", {}).get("type", "Normal")),
                    "confidence": round(float(item.get("behavior", {}).get("confidence", 0.0)), 4),
                    "role": str(item.get("behavior", {}).get("role", "unknown")),
                },
            }
            for item in detections_sorted
        ],
    }
    signature.update(json.dumps(compact, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark full-frame inference pipeline on a local video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", default="high_accuracy")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    engine, merged_cfg = _build_engine(args.profile)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0.0:
        source_fps = 25.0

    stage_sums = Counter()
    stage_max = Counter()
    label_counter = Counter()
    signature = hashlib.sha256()
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    total_frames = 0
    source_frame_id = 0
    start_frame = max(0, int(args.start_frame))
    wall_t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            source_frame_id += 1

            if source_frame_id <= start_frame:
                continue

            if args.max_frames > 0 and total_frames >= args.max_frames:
                break
            total_frames += 1
            frame_id = start_frame + total_frames
            result = engine.process(frame, frame_id)
            payload = result.model_dump()

            stage = payload.get("metrics", {}).get("pipeline_stage_ms", {})
            for key in ["detector", "pose", "action", "post", "sum_stages", "pipeline_compute"]:
                value = float(stage.get(key, 0.0))
                stage_sums[key] += value
                stage_max[key] = max(stage_max[key], value)

            for det in payload.get("detections", []):
                label = str(det.get("behavior", {}).get("type", "Normal"))
                label_counter[label] += 1

            _frame_signature_update(signature, payload)

            if total_frames % 200 == 0:
                if total_frame_count > 0:
                    print(f"progress: {frame_id}/{total_frame_count}")
                else:
                    print(f"progress: {frame_id}")
    finally:
        cap.release()

    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
    effective_fps = float(total_frames) / max(1e-6, wall_ms / 1000.0)

    summary = {
        "video": str(video_path),
        "profile": args.profile,
        "start_frame": start_frame,
        "config_inference": merged_cfg.get("inference", {}),
        "config_runtime": merged_cfg.get("runtime", {}),
        "frames": total_frames,
        "source_fps": round(source_fps, 4),
        "wall_time_ms": round(wall_ms, 3),
        "effective_fps": round(effective_fps, 4),
        "stage_avg_ms": {
            key: round(stage_sums[key] / max(1, total_frames), 4)
            for key in ["detector", "pose", "action", "post", "sum_stages", "pipeline_compute"]
        },
        "stage_max_ms": {
            key: round(stage_max[key], 4)
            for key in ["detector", "pose", "action", "post", "sum_stages", "pipeline_compute"]
        },
        "label_counts": dict(label_counter),
        "result_signature_sha256": signature.hexdigest(),
    }

    out_path = Path(args.output) if args.output else (BASE_DIR / "results" / f"perf_benchmark_{args.profile}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
