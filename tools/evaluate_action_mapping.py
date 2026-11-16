from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_LABELS = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "S01"]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def iter_event_files(results_dir: Path, pattern: str, sort_by: str = "name") -> list[Path]:
    files = [item for item in results_dir.glob(pattern) if item.is_file()]
    if sort_by == "mtime":
        files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    else:
        files.sort()
    return files


def parse_one_event(payload: dict[str, Any]) -> dict[str, Any]:
    frame_id = payload.get("frame_id", -1)
    detections = payload.get("detections", []) or []
    metrics = payload.get("metrics", {}) or {}
    action_debug = metrics.get("action_debug", {}) or {}

    entries: list[dict[str, Any]] = []
    for det in detections:
        behavior = det.get("behavior", {}) or {}
        entries.append(
            {
                "frame_id": frame_id,
                "track_id": det.get("track_id", "-"),
                "label": behavior.get("type", "Normal"),
                "confidence": safe_float(behavior.get("confidence", 0.0)),
                "mapping_mode": str(action_debug.get("mapping_mode", "unknown")),
                "external_top1_label": str(action_debug.get("external_top1_label", "")),
                "top1_score": safe_float(action_debug.get("top1_score", behavior.get("confidence", 0.0))),
                "used_fallback": bool(action_debug.get("used_fallback", True)),
                "error": str(action_debug.get("error", "")),
                "source_label_count": int(action_debug.get("source_label_count", 0) or 0),
            }
        )

    return {"entries": entries, "metrics": metrics}


def evaluate(entries: list[dict[str, Any]], low_conf_threshold: float, topk_external: int, topk_low: int) -> dict[str, Any]:
    if not entries:
        return {
            "summary": {
                "total_detections": 0,
                "message": "no detections found",
            }
        }

    label_counter = Counter(item["label"] for item in entries)
    mode_counter = Counter(item["mapping_mode"] for item in entries)
    external_counter = Counter(item["external_top1_label"] for item in entries if item["external_top1_label"])
    fallback_counter = Counter("fallback" if item["used_fallback"] else "model" for item in entries)
    error_counter = Counter(item["error"] for item in entries if item["error"])

    conf_values = [item["confidence"] for item in entries]
    top1_values = [item["top1_score"] for item in entries]

    unknown_labels = [item for item in entries if item["label"] not in PROJECT_LABELS]
    low_conf_items = [item for item in entries if item["confidence"] < low_conf_threshold]

    per_label_conf: dict[str, list[float]] = defaultdict(list)
    for item in entries:
        per_label_conf[item["label"]].append(item["confidence"])

    per_label_conf_mean = {
        key: round(mean(values), 4) for key, values in sorted(per_label_conf.items(), key=lambda kv: kv[0]) if values
    }

    low_conf_samples = sorted(
        low_conf_items,
        key=lambda item: (item["confidence"], item["frame_id"]),
    )[:topk_low]

    report = {
        "summary": {
            "total_detections": len(entries),
            "project_label_coverage": round((len(entries) - len(unknown_labels)) / max(1, len(entries)), 4),
            "unknown_label_count": len(unknown_labels),
            "avg_confidence": round(mean(conf_values), 4),
            "avg_top1_score": round(mean(top1_values), 4),
            "low_confidence_ratio": round(len(low_conf_items) / max(1, len(entries)), 4),
            "low_conf_threshold": low_conf_threshold,
        },
        "distribution": {
            "label_count": dict(label_counter.most_common()),
            "mapping_mode_count": dict(mode_counter.most_common()),
            "inference_source_count": dict(fallback_counter.most_common()),
            "mean_confidence_by_label": per_label_conf_mean,
        },
        "external_top1_topk": [
            {"label": label, "count": count} for label, count in external_counter.most_common(topk_external)
        ],
        "errors_topk": [
            {"error": error, "count": count} for error, count in error_counter.most_common(10)
        ],
        "low_confidence_samples": [
            {
                "frame_id": item["frame_id"],
                "track_id": item["track_id"],
                "label": item["label"],
                "confidence": item["confidence"],
                "mapping_mode": item["mapping_mode"],
                "external_top1_label": item["external_top1_label"],
            }
            for item in low_conf_samples
        ],
        "unknown_label_samples": [
            {
                "frame_id": item["frame_id"],
                "track_id": item["track_id"],
                "label": item["label"],
                "mapping_mode": item["mapping_mode"],
                "external_top1_label": item["external_top1_label"],
            }
            for item in unknown_labels[:topk_low]
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate temporary 400->8 mapping quality from result events")
    parser.add_argument("--results-dir", default="results", help="directory containing event_*.json")
    parser.add_argument("--pattern", default="event_*.json", help="glob pattern for event files")
    parser.add_argument("--max-files", type=int, default=0, help="limit number of files to scan, 0 means all")
    parser.add_argument("--recent-files", type=int, default=0, help="scan only N most recent files by mtime")
    parser.add_argument("--sort-by", choices=["name", "mtime"], default="name", help="file scan order")
    parser.add_argument("--low-conf-threshold", type=float, default=0.55, help="threshold for low-confidence samples")
    parser.add_argument("--topk-external", type=int, default=20, help="top K external labels in report")
    parser.add_argument("--topk-low", type=int, default=20, help="top K low-confidence and unknown samples")
    parser.add_argument("--output", default="results/mapping_eval_report.json", help="output report path")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    results_dir = (base_dir / args.results_dir).resolve()

    if not results_dir.exists() or not results_dir.is_dir():
        raise SystemExit(f"[ERROR] results dir not found: {results_dir}")

    event_files = iter_event_files(results_dir, args.pattern, sort_by=args.sort_by)
    if args.recent_files and args.recent_files > 0:
        event_files = event_files[: args.recent_files]
    if args.max_files and args.max_files > 0:
        event_files = event_files[: args.max_files]

    if not event_files:
        raise SystemExit(f"[ERROR] no files matched: dir={results_dir}, pattern={args.pattern}")

    all_entries: list[dict[str, Any]] = []
    for file_path in event_files:
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            parsed = parse_one_event(payload)
            all_entries.extend(parsed["entries"])
        except Exception as exc:
            print(f"[WARN] skip invalid file: {file_path} ({exc})")

    report = evaluate(
        entries=all_entries,
        low_conf_threshold=args.low_conf_threshold,
        topk_external=args.topk_external,
        topk_low=args.topk_low,
    )

    report["scan"] = {
        "results_dir": str(results_dir),
        "pattern": args.pattern,
        "sort_by": args.sort_by,
        "recent_files": args.recent_files,
        "files_scanned": len(event_files),
        "output": str((base_dir / args.output).resolve()),
    }

    output_path = (base_dir / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("[OK] mapping evaluation finished")
    print(f"[INFO] files_scanned={len(event_files)} detections={report.get('summary', {}).get('total_detections', 0)}")
    print(f"[INFO] output={output_path}")


if __name__ == "__main__":
    main()
