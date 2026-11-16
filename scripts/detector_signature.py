from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_yaml
from src.utils.preflight import resolve_profile
from src.core.detector import Detector


def normalize_detection(det: dict) -> dict:
    bbox = det.get("bbox", [0, 0, 0, 0])
    return {
        "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
        "score": round(float(det.get("score", 0.0)), 6),
        "class": str(det.get("class", "person")),
    }


def frame_signature(dets: list[dict]) -> str:
    payload = json.dumps(dets, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture detector output signatures on sampled frames")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="realtime")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_cfg = load_yaml(project_root / args.model_config)
    merged_cfg, _ = resolve_profile(model_cfg, args.profile)
    detector_cfg = merged_cfg.get("detector", {})
    detector = Detector(detector_cfg)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")

    rows = []
    frame_id = 0
    kept = 0
    sample_step = max(1, int(args.sample_step))
    max_frames = max(1, int(args.max_frames))

    while kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % sample_step != 0:
            continue

        dets = [normalize_detection(d) for d in detector.infer(frame)]
        dets.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], -d["score"]))
        rows.append(
            {
                "frame_id": frame_id,
                "count": len(dets),
                "signature": frame_signature(dets),
            }
        )
        kept += 1

    cap.release()

    output = {
        "video": args.video,
        "profile": args.profile,
        "sample_step": sample_step,
        "sampled_frames": kept,
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "sampled_frames": kept}, ensure_ascii=False))


if __name__ == "__main__":
    main()
