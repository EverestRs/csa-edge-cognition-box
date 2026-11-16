from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
STREAM_URL = "D:/Workspace/Project/VideoAnalyst/data/whole_process_forward_1.mp4"


def main() -> None:
    requests.post(f"{BASE}/api/v1/profile", params={"profile": "high_accuracy"}, timeout=20)
    resp = requests.post(
        f"{BASE}/api/v1/analyze/stream",
        json={"stream_url": STREAM_URL, "duration": 70, "callback_url": None},
        timeout=20,
    )
    resp.raise_for_status()
    task_id = resp.json().get("task_id", "")

    samples: list[dict] = []
    mode_counter: Counter[str] = Counter()
    non_fallback = 0

    for _ in range(140):
        time.sleep(1)
        try:
            health = requests.get(f"{BASE}/health", timeout=10).json()
        except Exception:
            continue

        action_debug = health.get("action_debug", {})
        used_fallback = bool(action_debug.get("used_fallback", True))
        mode = str(action_debug.get("mapping_mode", "unknown"))
        mode_counter[mode] += 1
        if not used_fallback:
            non_fallback += 1

        samples.append(
            {
                "used_fallback": used_fallback,
                "mapping_mode": mode,
                "buffer_len": int(action_debug.get("buffer_len", 0) or 0),
                "error": str(action_debug.get("error", "")),
                "top1_label": str(action_debug.get("top1_label", "Normal")),
                "top1_score": float(action_debug.get("top1_score", 0.0) or 0.0),
                "external_top1_label": str(action_debug.get("external_top1_label", "")),
            }
        )

    results_dir = Path(__file__).resolve().parents[1] / "results"
    newest = sorted(results_dir.glob("event_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]

    summary = {
        "task_id": task_id,
        "samples": len(samples),
        "non_fallback_samples": non_fallback,
        "non_fallback_ratio": round(non_fallback / max(1, len(samples)), 4),
        "mapping_mode_counter": dict(mode_counter),
        "newest_events": [item.name for item in newest],
        "last_sample": samples[-1] if samples else {},
    }

    out = {
        "summary": summary,
        "samples": samples,
    }

    out_path = results_dir / "action_debug_sampling_clip16_stride1.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
