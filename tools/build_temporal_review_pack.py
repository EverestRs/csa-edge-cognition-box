from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"report not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid report format: {path}")
    return data


def get_intervals(report: dict[str, Any], label: str) -> list[dict[str, Any]]:
    behaviors = report.get("behaviors", [])
    if not isinstance(behaviors, list):
        return []
    for item in behaviors:
        if not isinstance(item, dict):
            continue
        if str(item.get("label", "")).upper() != label.upper():
            continue
        intervals = item.get("event_intervals", [])
        if isinstance(intervals, list):
            out: list[dict[str, Any]] = []
            for iv in intervals:
                if not isinstance(iv, dict):
                    continue
                s = int(iv.get("start_frame", -1))
                e = int(iv.get("end_frame", -1))
                if s <= 0 or e <= 0:
                    continue
                if e < s:
                    s, e = e, s
                out.append({"start_frame": s, "end_frame": e})
            return out
    return []


def nearest_distance(interval: dict[str, int], others: list[dict[str, int]]) -> int:
    s1 = int(interval["start_frame"])
    e1 = int(interval["end_frame"])
    best = 10**9
    for iv in others:
        s2 = int(iv["start_frame"])
        e2 = int(iv["end_frame"])
        if e1 < s2:
            d = s2 - e1
        elif e2 < s1:
            d = s1 - e2
        else:
            d = 0
        if d < best:
            best = d
    return best if best < 10**9 else -1


def merged_union(intervals: list[dict[str, int]]) -> list[dict[str, int]]:
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: (x["start_frame"], x["end_frame"]))
    merged: list[dict[str, int]] = [dict(sorted_ivs[0])]
    for iv in sorted_ivs[1:]:
        cur = merged[-1]
        if iv["start_frame"] <= cur["end_frame"] + 1:
            cur["end_frame"] = max(cur["end_frame"], iv["end_frame"])
        else:
            merged.append(dict(iv))
    return merged


def build_non_fall_candidates(
    all_risk: list[dict[str, int]],
    total_frames: int,
    min_gap_frames: int,
    sample_len_frames: int,
    max_samples: int,
) -> list[dict[str, int]]:
    if total_frames <= 0:
        return []
    merged = merged_union(all_risk)
    gaps: list[tuple[int, int]] = []

    cur = 1
    for iv in merged:
        s = int(iv["start_frame"])
        e = int(iv["end_frame"])
        if s > cur:
            gaps.append((cur, s - 1))
        cur = max(cur, e + 1)
    if cur <= total_frames:
        gaps.append((cur, total_frames))

    out: list[dict[str, int]] = []
    for g0, g1 in gaps:
        gap_len = g1 - g0 + 1
        if gap_len < min_gap_frames:
            continue
        mid = (g0 + g1) // 2
        half = max(1, sample_len_frames // 2)
        s = max(g0, mid - half)
        e = min(g1, mid + half)
        out.append({"start_frame": int(s), "end_frame": int(e)})
        if len(out) >= max_samples:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal annotation review pack from verify reports")
    parser.add_argument("--reports", nargs="+", required=True, help="verify report json paths")
    parser.add_argument("--output-review", required=True, help="output review json")
    parser.add_argument("--output-intervals", required=True, help="output interval json template")
    parser.add_argument("--h03-h04-near-gap", type=int, default=180, help="frames to consider H03 near H04 as conflict")
    parser.add_argument("--max-items-per-report", type=int, default=28)
    parser.add_argument("--non-fall-min-gap", type=int, default=180)
    parser.add_argument("--non-fall-len", type=int, default=60)
    parser.add_argument("--non-fall-per-report", type=int, default=8)
    args = parser.parse_args()

    review_items: list[dict[str, Any]] = []
    interval_items: list[dict[str, Any]] = []

    for rp in args.reports:
        report_path = Path(rp)
        report = load_report(report_path)
        video = str(report.get("video", ""))
        total_frames = int(report.get("total_frames", 0))

        h03 = get_intervals(report, "H03")
        h04 = get_intervals(report, "H04")

        local: list[dict[str, Any]] = []

        for idx, iv in enumerate(h04, start=1):
            local.append(
                {
                    "id": f"{report_path.stem}-h04-{idx}",
                    "video": video,
                    "start_frame": int(iv["start_frame"]),
                    "end_frame": int(iv["end_frame"]),
                    "suggested_target_class": "risk_then_fall",
                    "confidence": "high",
                    "reason": "Detected as H04 event interval",
                    "source_report": str(report_path),
                    "review_status": "todo",
                    "review_target_class": "",
                    "review_note": "",
                }
            )

        for idx, iv in enumerate(h03, start=1):
            d = nearest_distance(iv, h04)
            if 0 <= d <= int(args.h03_h04_near_gap):
                suggested = "risk_then_fall"
                confidence = "medium"
                reason = f"H03 interval near H04 within {d} frames"
            else:
                suggested = "direct_fall"
                confidence = "medium"
                reason = "Detected as H03 interval away from H04"
            local.append(
                {
                    "id": f"{report_path.stem}-h03-{idx}",
                    "video": video,
                    "start_frame": int(iv["start_frame"]),
                    "end_frame": int(iv["end_frame"]),
                    "suggested_target_class": suggested,
                    "confidence": confidence,
                    "reason": reason,
                    "source_report": str(report_path),
                    "review_status": "todo",
                    "review_target_class": "",
                    "review_note": "",
                }
            )

        non_fall = build_non_fall_candidates(
            all_risk=[*h03, *h04],
            total_frames=total_frames,
            min_gap_frames=max(1, int(args.non_fall_min_gap)),
            sample_len_frames=max(8, int(args.non_fall_len)),
            max_samples=max(0, int(args.non_fall_per_report)),
        )
        for idx, iv in enumerate(non_fall, start=1):
            local.append(
                {
                    "id": f"{report_path.stem}-nf-{idx}",
                    "video": video,
                    "start_frame": int(iv["start_frame"]),
                    "end_frame": int(iv["end_frame"]),
                    "suggested_target_class": "non_fall",
                    "confidence": "medium",
                    "reason": "Long gap between risk intervals",
                    "source_report": str(report_path),
                    "review_status": "todo",
                    "review_target_class": "",
                    "review_note": "",
                }
            )

        local_sorted = sorted(
            local,
            key=lambda x: (
                0 if x["suggested_target_class"] == "risk_then_fall" else 1,
                int(x["start_frame"]),
            ),
        )

        for item in local_sorted[: max(1, int(args.max_items_per_report))]:
            review_items.append(item)
            interval_items.append(
                {
                    "start_frame": int(item["start_frame"]),
                    "end_frame": int(item["end_frame"]),
                    "target_class": str(item["suggested_target_class"]),
                    "source_id": str(item["id"]),
                    "review_status": "todo",
                    "review_target_class": "",
                    "review_note": "",
                }
            )

    review_payload = {
        "instructions": [
            "Set review_status to approved/rejected/todo.",
            "If approved and class should change, fill review_target_class.",
            "Allowed classes: direct_fall, risk_then_fall, non_fall.",
            "Keep start_frame/end_frame unless interval boundaries need correction.",
        ],
        "items": review_items,
    }

    interval_payload = {
        "intervals": interval_items,
        "instructions": {
            "how_to_use": "After review, keep approved intervals and pass this file to --target-intervals.",
            "allowed_classes": ["direct_fall", "risk_then_fall", "non_fall"],
        },
    }

    out_review = Path(args.output_review)
    out_review.parent.mkdir(parents=True, exist_ok=True)
    out_review.write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_intervals = Path(args.output_intervals)
    out_intervals.parent.mkdir(parents=True, exist_ok=True)
    out_intervals.write_text(json.dumps(interval_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "reports": len(args.reports),
                "review_items": len(review_items),
                "interval_items": len(interval_items),
                "output_review": str(out_review),
                "output_intervals": str(out_intervals),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
