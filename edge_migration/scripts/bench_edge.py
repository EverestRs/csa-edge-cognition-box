#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T2.4 前置：edge 档位 bench300 等效基准（盒上/x86 通用）。

跑 N 帧完整感知链路（指定 profile），输出 fps / P50 / P95 / 各阶段耗时，
供评估回归证据表直接引用（数字必须来自本脚本 JSON，禁止手工填表）。

用法:
  python edge_migration/scripts/bench_edge.py --frames 300
  python edge_migration/scripts/bench_edge.py --profile realtime --video ../test/whole_process_corner_3.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="edge")
    ap.add_argument("--video", default="../test/whole_process_corner_3.mp4")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "results/edge_test" / f"bench_{a.profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    from src.pipeline.inference_engine import InferenceEngine
    from src.utils.preflight import resolve_profile

    model_cfg = yaml.safe_load((ROOT / "configs/model_config.yaml").read_text(encoding="utf-8"))
    vlm_cfg = yaml.safe_load((ROOT / "configs/vlm_config.yaml").read_text(encoding="utf-8")) or {}
    merged, _ = resolve_profile(model_cfg, a.profile)

    engine = InferenceEngine(model_cfg=merged, vlm_cfg=vlm_cfg,
                             output_dir=str(ROOT / "results/bench_tmp"))
    engine.begin_task("bench")
    engine.set_source_fps(25.0)

    cap = cv2.VideoCapture(str(ROOT / a.video))
    if not cap.isOpened():
        print(f"[bench] 视频打不开: {a.video}")
        return 1
    lat: list[float] = []
    stage_sums = {"detector": 0.0, "pose": 0.0, "action": 0.0, "post": 0.0}
    processed = 0
    t0 = time.perf_counter()
    while processed < a.frames:
        ok, frame = cap.read()
        if not ok:
            break
        processed += 1
        s = time.perf_counter()
        result = engine.process(frame, processed)
        lat.append((time.perf_counter() - s) * 1000)
        stages = result.metrics.get("pipeline_stage_ms", {})
        for k in stage_sums:
            stage_sums[k] += float(stages.get(k, 0) or 0)
    cap.release()
    wall = time.perf_counter() - t0
    lat.sort()
    report = {
        "probe": "bench_edge", "ts": time.time(), "profile": a.profile,
        "video": a.video, "frames": processed,
        "wall_s": round(wall, 2),
        "fps": round(processed / max(wall, 1e-6), 2),
        "latency_ms_p50": round(lat[len(lat) // 2], 1),
        "latency_ms_p95": round(lat[int(len(lat) * 0.95)], 1),
        "latency_ms_mean": round(sum(lat) / max(len(lat), 1), 1),
        "stage_ms_mean": {k: round(v / max(processed, 1), 2) for k, v in stage_sums.items()},
        "model_load": result.metrics.get("model_load_status", {}),
    }
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))
    print(f"[bench] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
