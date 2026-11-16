from __future__ import annotations

import json
import time
import argparse
from collections import Counter
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
STREAM_URL = "D:/Workspace/Project/VideoAnalyst/data/whole_process_forward_1.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short inference and sample action debug")
    parser.add_argument("--duration", type=int, default=70)
    parser.add_argument("--stream-url", default=STREAM_URL)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--max-polls", type=int, default=240)
    args = parser.parse_args()

    requests.post(f"{BASE}/api/v1/profile", params={"profile": "high_accuracy"}, timeout=20)

    payload = {"stream_url": args.stream_url, "duration": int(args.duration), "callback_url": None}
    task_resp = requests.post(f"{BASE}/api/v1/analyze/stream", json=payload, timeout=20)
    task_resp.raise_for_status()
    task_id = task_resp.json()["task_id"]
    print(f"task_id={task_id}")

    last = {}
    sampled_debug: list[dict] = []
    mode_counter: Counter[str] = Counter()
    non_fallback_count = 0

    for _ in range(int(args.max_polls)):
        time.sleep(max(0.2, float(args.poll_seconds)))
        status_resp = requests.get(f"{BASE}/api/v1/tasks/{task_id}", timeout=20)
        status_resp.raise_for_status()
        data = status_resp.json()
        last = data
        health = requests.get(f"{BASE}/health", timeout=20).json()
        action_debug = health.get("action_debug", {})
        mode = str(action_debug.get("mapping_mode", "unknown"))
        used_fallback = bool(action_debug.get("used_fallback", True))
        mode_counter[mode] += 1
        if not used_fallback:
            non_fallback_count += 1

        sampled_debug.append(
            {
                "status": data.get("status", "unknown"),
                "frame": int(data.get("last_frame", -1) or -1),
                "used_fallback": used_fallback,
                "mapping_mode": mode,
                "error": str(action_debug.get("error", "")),
                "buffer_len": int(action_debug.get("buffer_len", 0) or 0),
                "external_top1_label": str(action_debug.get("external_top1_label", "")),
                "top1_label": str(action_debug.get("top1_label", "Normal")),
                "top1_score": float(action_debug.get("top1_score", 0.0) or 0.0),
            }
        )

        print(
            f"status={data.get('status')} frame={data.get('last_frame', -1)} "
            f"fallback={used_fallback} mode={mode} err={action_debug.get('error', '')}"
        )
        if data.get("status") != "running":
            break

    health = requests.get(f"{BASE}/health", timeout=20).json()
    action_debug = health.get("action_debug", {})
    print("action_debug:", json.dumps(action_debug, ensure_ascii=False))

    results_dir = Path(__file__).resolve().parents[1] / "results"
    newest = sorted(results_dir.glob("event_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
    out = {
        "task_id": task_id,
        "config": {
            "duration": args.duration,
            "stream_url": args.stream_url,
            "poll_seconds": args.poll_seconds,
            "max_polls": args.max_polls,
        },
        "final_task": last,
        "action_debug": action_debug,
        "sampling_summary": {
            "total_samples": len(sampled_debug),
            "non_fallback_samples": non_fallback_count,
            "non_fallback_ratio": round(non_fallback_count / max(1, len(sampled_debug)), 4),
            "mapping_mode_counter": dict(mode_counter),
        },
        "samples": sampled_debug,
        "newest_events": [item.name for item in newest],
    }
    out_path = results_dir / "short_eval_summary.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary={out_path}")


if __name__ == "__main__":
    main()
