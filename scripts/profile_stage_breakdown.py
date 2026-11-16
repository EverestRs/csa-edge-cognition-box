from __future__ import annotations

import asyncio
import argparse
import json
import urllib.request
from pathlib import Path

import websockets


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_task(base: str, stream_url: str, duration: int) -> dict:
    payload = {
        "stream_url": stream_url,
        "duration": duration,
        "callback_url": None,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/v1/analyze/stream",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def sample_ws(ws_url: str, max_samples: int = 80, timeout_s: float = 3.0) -> dict:
    stats = {
        "detector": [],
        "pose": [],
        "action": [],
        "post": [],
        "sum": [],
        "pipe": [],
    }
    async with websockets.connect(ws_url, max_size=2**22) as ws:
        await ws.send("start")
        for _ in range(max_samples):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            except Exception:
                break
            obj = json.loads(msg)
            stage = obj.get("metrics", {}).get("pipeline_stage_ms")
            if not stage:
                continue
            stats["detector"].append(float(stage.get("detector", 0.0)))
            stats["pose"].append(float(stage.get("pose", 0.0)))
            stats["action"].append(float(stage.get("action", 0.0)))
            stats["post"].append(float(stage.get("post", 0.0)))
            stats["sum"].append(float(stage.get("sum_stages", 0.0)))
            stats["pipe"].append(float(obj.get("metrics", {}).get("pipeline_latency_ms", 0.0)))
    return stats


def avg(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile per-stage pipeline cost from websocket events")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument(
        "--video",
        default="D:/Workspace/Project/VideoAnalyst/data/laying_backward_1-1.mp4",
        help="Local video path",
    )
    parser.add_argument("--duration", type=int, default=30, help="Task duration seconds")
    parser.add_argument("--max-samples", type=int, default=80, help="Max websocket samples")
    parser.add_argument("--timeout", type=float, default=3.0, help="Per message timeout seconds")
    parser.add_argument("--output", default="", help="Optional output json path")
    args = parser.parse_args()

    base = args.base
    stream_url = args.video

    task = create_task(base, stream_url, duration=int(args.duration))
    ws_url = task["endpoint"]
    if "127.0.0.1:8000" in ws_url and "127.0.0.1:8000" not in base:
        ws_url = ws_url.replace("127.0.0.1:8000", base.split("//", 1)[-1])
    sampled = asyncio.run(sample_ws(ws_url, max_samples=int(args.max_samples), timeout_s=float(args.timeout)))

    sample_count = len(sampled["sum"])
    if sample_count == 0:
        result = {"task_id": task.get("task_id", ""), "samples": 0, "note": "no pipeline_stage_ms sampled"}
        if args.output:
            out = Path(args.output)
            if not out.is_absolute():
                out = PROJECT_ROOT / out
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return

    avg_ms = {
        "detector": round(avg(sampled["detector"]), 3),
        "pose": round(avg(sampled["pose"]), 3),
        "action": round(avg(sampled["action"]), 3),
        "post": round(avg(sampled["post"]), 3),
        "sum_stages": round(avg(sampled["sum"]), 3),
        "pipeline_compute": round(avg(sampled["pipe"]), 3),
    }
    denom = max(0.001, avg_ms["sum_stages"])
    share = {
        "detector": round(avg_ms["detector"] * 100.0 / denom, 2),
        "pose": round(avg_ms["pose"] * 100.0 / denom, 2),
        "action": round(avg_ms["action"] * 100.0 / denom, 2),
        "post": round(avg_ms["post"] * 100.0 / denom, 2),
    }

    result = {
        "task_id": task.get("task_id", ""),
        "samples": sample_count,
        "avg_ms": avg_ms,
        "share_pct_of_stage_sum": share,
        "sum_vs_pipeline_pct": round(avg_ms["sum_stages"] * 100.0 / max(0.001, avg_ms["pipeline_compute"]), 2),
    }
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
