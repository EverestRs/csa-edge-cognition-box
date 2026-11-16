#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G6 探针：PoseClassifier GRU 时延（T1.7）。

用法: python edge_migration/scripts/g6_gru_bench.py [--model models/action_v2/pose_classifier.onnx]
x86 即可预跑（对照盒上 A55 数字的比值预期）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge_migration.inference_backend import create_backend  # noqa: E402


def resolve_shape(declared) -> tuple[bool, tuple | None]:
    """ORT 动态维是字符串/'?'/-1：静态形状直接采用，否则交给候选试跑。"""
    def concrete(d):
        try:
            return int(d)
        except (TypeError, ValueError):
            return None
    static = tuple(concrete(d) for d in declared) if declared else None
    if static is not None and len(static) == 4 and all(v is not None and v > 0 for v in static):
        return True, tuple(static)
    return False, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/action_v2/pose_classifier.onnx")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--candidates", default="1,32,133,3;1,32,17,3")
    ap.add_argument("--out", default="results/edge_m0/g6_gru_latency.json")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    if not Path(a.model).exists():
        json.dump({"probe": "g6_gru", "status": "FAIL", "reason": f"missing: {a.model}"},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"[g6] FAIL: {a.model} 不存在")
        return 1

    backend = create_backend(a.model, runtime={"intra_op_num_threads": a.threads})
    if not backend.is_real:
        json.dump({"probe": "g6_gru", "status": "FAIL",
                   "reason": getattr(backend, "last_error", "?")},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print("[g6] FAIL: 模型加载失败")
        return 1

    rng = np.random.default_rng(5)
    used_shape = None
    declared = getattr(backend, "input_shape", ())
    _, static_shape = resolve_shape(declared)
    tried = []
    if static_shape:
        tried.append(static_shape)
    tried += [tuple(int(v) for v in c.split(",")) for c in a.candidates.split(";")]
    for shape in tried:
        if len(shape) != 4:
            continue
        try:
            x = rng.random(shape, dtype=np.float32)
            o = backend.run(x)
            if o and all(np.isfinite(np.asarray(v)).all() for v in o):
                used_shape = shape
                break
        except Exception as exc:
            tried.append(str(exc)[:60])
    if used_shape is None:
        json.dump({"probe": "g6_gru", "status": "FAIL", "reason": "no_runnable_input_shape",
                   "declared": list(declared or [])},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print("[g6] FAIL: 无法确定可运行输入形状")
        return 1

    for _ in range(8):
        backend.run(x)
    times = []
    for _ in range(a.runs):
        s = time.perf_counter()
        backend.run(x)
        times.append((time.perf_counter() - s) * 1000)
    arr = np.array(times)
    rss_mb = None
    try:
        import psutil
        rss_mb = round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 1)
    except Exception:
        pass
    out = {"probe": "g6_gru_latency", "ts": time.time(), "model": a.model,
           "input_shape": list(used_shape), "threads": a.threads,
           "gru_ms_p50": round(float(np.percentile(arr, 50)), 2),
           "gru_ms_p95": round(float(np.percentile(arr, 95)), 2),
           "rss_mb_after_warm": rss_mb, "pass_50ms": bool(np.percentile(arr, 95) <= 50.0)}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[g6] shape={used_shape} p50={out['gru_ms_p50']}ms p95={out['gru_ms_p95']}ms -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
