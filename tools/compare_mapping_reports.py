from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pick(report: dict[str, Any], keys: list[str], default: Any = 0) -> Any:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def to_num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two mapping evaluation reports")
    parser.add_argument("--baseline", default="results/mapping_eval_report_recent12.json")
    parser.add_argument("--candidate", default="results/mapping_eval_report_clip16.json")
    parser.add_argument("--sampling", default="results/action_debug_sampling_clip16_stride1.json")
    parser.add_argument("--output", default="results/mapping_eval_comparison.json")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    baseline_path = (base_dir / args.baseline).resolve()
    candidate_path = (base_dir / args.candidate).resolve()
    output_path = (base_dir / args.output).resolve()
    sampling_path = (base_dir / args.sampling).resolve()

    baseline = read_json(baseline_path)
    candidate = read_json(candidate_path)

    summary_keys = {
        "total_detections": ["summary", "total_detections"],
        "avg_confidence": ["summary", "avg_confidence"],
        "avg_top1_score": ["summary", "avg_top1_score"],
        "low_confidence_ratio": ["summary", "low_confidence_ratio"],
    }

    compare_summary: dict[str, Any] = {}
    for name, key_path in summary_keys.items():
        old_val = pick(baseline, key_path, 0)
        new_val = pick(candidate, key_path, 0)
        compare_summary[name] = {
            "baseline": old_val,
            "candidate": new_val,
            "delta": round(to_num(new_val) - to_num(old_val), 6),
        }

    base_mode = pick(baseline, ["distribution", "mapping_mode_count"], {}) or {}
    cand_mode = pick(candidate, ["distribution", "mapping_mode_count"], {}) or {}
    base_infer = pick(baseline, ["distribution", "inference_source_count"], {}) or {}
    cand_infer = pick(candidate, ["distribution", "inference_source_count"], {}) or {}

    comparison = {
        "baseline_report": str(baseline_path),
        "candidate_report": str(candidate_path),
        "summary_compare": compare_summary,
        "mapping_mode_compare": {
            "baseline": base_mode,
            "candidate": cand_mode,
        },
        "inference_source_compare": {
            "baseline": base_infer,
            "candidate": cand_infer,
        },
        "notes": [],
    }

    if to_num(compare_summary["total_detections"]["delta"]) > 0:
        comparison["notes"].append("candidate has more evaluated detections")
    if to_num(compare_summary["avg_top1_score"]["delta"]) > 0:
        comparison["notes"].append("candidate has higher average top1 score")

    if sampling_path.exists():
        sampling = read_json(sampling_path)
        sampling_summary = sampling.get("summary", {}) if isinstance(sampling, dict) else {}
        comparison["sampling_summary"] = sampling_summary
        if to_num(sampling_summary.get("non_fallback_ratio", 0.0)) > 0:
            comparison["notes"].append("A-Plus runtime sampling observed non-fallback action inference")
        else:
            comparison["notes"].append("A-Plus runtime sampling did not capture non-fallback action inference")
    else:
        comparison["sampling_summary"] = {"status": "not_found", "path": str(sampling_path)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2)

    print("[OK] comparison generated")
    print(f"[INFO] output={output_path}")


if __name__ == "__main__":
    main()
