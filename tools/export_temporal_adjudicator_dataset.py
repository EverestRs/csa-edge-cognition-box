from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections import Counter

import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.inference_engine import InferenceEngine
from src.utils.config_loader import load_json, load_yaml
from src.utils.preflight import resolve_profile


def parse_behavior_to_class(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    text = str(value or "").strip()
    if not text:
        return mapping
    for part in text.split(","):
        item = part.strip()
        if not item or ":" not in item:
            continue
        key, val = item.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key and val:
            mapping[key] = val
    return mapping


def load_target_map_file(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): str(v) for k, v in data.items()}


def load_target_map_from_report(path: Path, behavior_to_class: dict[str, str]) -> dict[int, str]:
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    target_map: dict[int, str] = {}
    behaviors = list(report.get("behaviors", [])) if isinstance(report, dict) else []
    for behavior in behaviors:
        label = str(behavior.get("label", ""))
        target_class = behavior_to_class.get(label, "")
        if not target_class:
            continue
        intervals = list(behavior.get("event_intervals", []))
        for interval in intervals:
            start_frame = int(interval.get("start_frame", -1))
            end_frame = int(interval.get("end_frame", -1))
            if start_frame <= 0 or end_frame <= 0:
                continue
            if end_frame < start_frame:
                start_frame, end_frame = end_frame, start_frame
            for fid in range(start_frame, end_frame + 1):
                if fid not in target_map:
                    target_map[fid] = target_class
    return target_map


def load_target_map_from_intervals(path: Path, video_filter: str = "") -> dict[int, str]:
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    intervals: list[dict[str, object]] = []

    if isinstance(data, list):
        intervals = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict):
        if isinstance(data.get("intervals"), list):
            intervals = [item for item in data.get("intervals", []) if isinstance(item, dict)]
        else:
            # Also support {"direct_fall": [{"start_frame": 10, "end_frame": 20}], ...}
            for cls_name, entries in data.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if isinstance(entry, dict):
                        item = dict(entry)
                        item.setdefault("target_class", str(cls_name))
                        intervals.append(item)
                    elif isinstance(entry, list) and len(entry) >= 2:
                        intervals.append(
                            {
                                "start_frame": int(entry[0]),
                                "end_frame": int(entry[1]),
                                "target_class": str(cls_name),
                            }
                        )

    target_map: dict[int, str] = {}
    video_filter = str(video_filter or "").strip().replace("\\", "/").lower()

    for interval in intervals:
        interval_video = str(interval.get("video", "")).strip().replace("\\", "/").lower()
        if video_filter and interval_video and interval_video != video_filter:
            continue

        target_class = str(interval.get("target_class", "")).strip()
        if not target_class:
            continue

        start_frame = int(interval.get("start_frame", interval.get("start", -1)))
        end_frame = int(interval.get("end_frame", interval.get("end", -1)))
        if start_frame <= 0 or end_frame <= 0:
            continue
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        for fid in range(start_frame, end_frame + 1):
            target_map[fid] = target_class
    return target_map


def is_conflict_supervision(main_label: str, target_class: str) -> bool:
    if not target_class:
        return False
    expected = {"H03": "direct_fall", "H04": "risk_then_fall"}
    if main_label not in expected:
        return False
    return str(target_class) != expected[main_label]


def collect_hard_negative_candidates(rows: list[dict[str, object]], score_threshold: float) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        target_class = str(row.get("target_class", "")).strip()
        if target_class:
            continue
        if str(row.get("main_label", "")) != "Normal":
            continue
        if str(row.get("prediction", "")) != "non_fall":
            continue
        prediction_score = float(row.get("prediction_score", 0.0))
        if prediction_score < score_threshold:
            continue
        out.append(row)
    out.sort(key=lambda item: (-float(item.get("prediction_score", 0.0)), int(item.get("frame_id", 0))))
    return out


def build_engine(profile: str) -> InferenceEngine:
    config = load_yaml(BASE_DIR / "configs" / "model_config.yaml")
    behavior_rules = load_json(BASE_DIR / "configs" / "behavior_rules.json")
    if isinstance(behavior_rules, dict) and isinstance(behavior_rules.get("heuristics"), dict):
        config["behavior_heuristics"] = dict(behavior_rules["heuristics"])
    temporal_cfg = dict(config.get("temporal_adjudicator", {}))
    temporal_cfg["enabled"] = True
    temporal_cfg["apply_decision"] = False
    config["temporal_adjudicator"] = temporal_cfg

    config, _ = resolve_profile(config, profile)
    return InferenceEngine(config, load_yaml(BASE_DIR / "configs" / "vlm_config.yaml"), output_dir=str(BASE_DIR / "results"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export temporal adjudicator training dataset")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="realtime")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--target-map", default="")
    parser.add_argument("--target-report", default="")
    parser.add_argument("--target-intervals", default="", help="manual interval labels json; overrides map/report on overlap")
    parser.add_argument("--behavior-to-class", default="H03:direct_fall,H04:risk_then_fall")
    parser.add_argument("--max-non-fall-samples", type=int, default=0, help="auto-label top-N unlabeled non_fall rows as hard negatives")
    parser.add_argument("--non-fall-score-threshold", type=float, default=0.92)
    parser.add_argument("--conflict-sample-weight", type=int, default=1, help="duplicate conflict-supervision rows by this factor")
    args = parser.parse_args()

    target_map: dict[int, str] = {}
    target_source_map: dict[int, str] = {}
    if args.target_map:
        p = Path(args.target_map)
        for fid, cls in load_target_map_file(p).items():
            if fid not in target_map:
                target_map[fid] = cls
                target_source_map[fid] = "target_map"

    behavior_to_class = parse_behavior_to_class(str(args.behavior_to_class))
    if args.target_report:
        report_path = Path(args.target_report)
        for fid, cls in load_target_map_from_report(report_path, behavior_to_class).items():
            if fid not in target_map:
                target_map[fid] = cls
                target_source_map[fid] = "target_report"

    if args.target_intervals:
        interval_path = Path(args.target_intervals)
        for fid, cls in load_target_map_from_intervals(interval_path, video_filter=str(args.video)).items():
            target_map[fid] = cls
            target_source_map[fid] = "target_intervals"

    engine = build_engine(str(args.profile))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0

    frame_step = max(1, int(args.frame_step))
    max_frames = max(0, int(args.max_frames))

    rows: list[dict[str, object]] = []
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_id += 1
        if max_frames > 0 and frame_id > max_frames:
            break
        if frame_id % frame_step != 0:
            continue

        result = engine.process(frame, frame_id)
        metrics = result.metrics if isinstance(result.metrics, dict) else {}
        temporal_debug = metrics.get("temporal_adjudicator_debug", {}) if isinstance(metrics.get("temporal_adjudicator_debug", {}), dict) else {}
        if not temporal_debug.get("enabled", False):
            continue

        row = {
            "frame_id": frame_id,
            "second": round(frame_id / fps, 3),
            "main_label": str(result.alert.type).replace("_detected", "").upper() if result.alert and result.alert.type != "none" else "Normal",
            "prediction": temporal_debug.get("prediction", ""),
            "prediction_score": temporal_debug.get("prediction_score", 0.0),
            "feature_names": temporal_debug.get("feature_names", []),
            "feature_vector": temporal_debug.get("feature_vector", []),
            "target_class": target_map.get(frame_id, ""),
            "target_source": target_source_map.get(frame_id, ""),
        }
        rows.append(row)

    cap.release()

    hard_negative_added = 0
    max_non_fall = max(0, int(args.max_non_fall_samples))
    if max_non_fall > 0:
        candidates = collect_hard_negative_candidates(rows, float(args.non_fall_score_threshold))
        selected = candidates[:max_non_fall]
        for row in selected:
            row["target_class"] = "non_fall"
            row["target_source"] = "auto_hard_negative"
        hard_negative_added = len(selected)

    conflict_weight = max(1, int(args.conflict_sample_weight))
    conflict_augmented = 0
    if conflict_weight > 1:
        extra_rows: list[dict[str, object]] = []
        for row in rows:
            target_class = str(row.get("target_class", "")).strip()
            main_label = str(row.get("main_label", "")).strip()
            if not is_conflict_supervision(main_label, target_class):
                continue
            for _ in range(conflict_weight - 1):
                extra_rows.append(dict(row))
        conflict_augmented = len(extra_rows)
        rows.extend(extra_rows)

    class_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    for row in rows:
        target_class = str(row.get("target_class", "")).strip() or "unlabeled"
        class_counter[target_class] += 1
        target_source = str(row.get("target_source", "")).strip() or "none"
        source_counter[target_source] += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "video": args.video,
                "rows": len(rows),
                "output": str(output_path),
                "fps": round(fps, 3),
                "class_counts": dict(class_counter),
                "source_counts": dict(source_counter),
                "hard_negative_added": int(hard_negative_added),
                "conflict_augmented": int(conflict_augmented),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
