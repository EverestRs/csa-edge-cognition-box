#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G3 盒上执行体：加载 OM 做 bench，并与同源 ONNX 做同输入数值对拍。

x86 上仅静态检查（pyACL 缺失时 AscendOMBackend 自动降级 → 本脚本报 SKIP）。
用法: python edge_migration/scripts/g3_om_infer.py --om X.om --compare-onnx X.onnx --runs 100 --out JSON
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge_migration.inference_backend import AscendOMBackend, OnnxRuntimeBackend  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--om", required=True)
    ap.add_argument("--compare-onnx", default="")
    ap.add_argument("--shape", default="1,3,640,640")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--out", default="results/edge_m0/g3_om_bench.json")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(v) for v in a.shape.split(","))

    result: dict = {"probe": "g3_om_infer", "ts": time.time(), "om": a.om, "shape": shape}
    om = AscendOMBackend(a.om, input_shape=shape)
    if not om.is_real:
        result.update({"status": "SKIP", "reason": om.last_error})
        json.dump(result, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[g3_om] SKIP: {om.last_error}")
        return 0

    rng = np.random.default_rng(7)
    inp = rng.random(shape, dtype=np.float32)
    for _ in range(5):
        om.run(inp)                    # warmup
    times = []
    outs = []
    for _ in range(a.runs):
        s = time.perf_counter()
        o = om.run(inp)
        times.append((time.perf_counter() - s) * 1000)
        outs = o
    arr = np.array(times)
    result.update({"status": "OK", "om_ms_p50": float(np.percentile(arr, 50)),
                   "om_ms_p95": float(np.percentile(arr, 95)),
                   "output_shapes": [list(np.asarray(x).shape) for x in outs[:4]]})

    if a.compare_onnx and Path(a.compare_onnx).exists() and outs:
        ort_b = OnnxRuntimeBackend(a.compare_onnx)
        ref = ort_b.run(inp) if ort_b.is_real else []
        if ref and len(ref) == len(outs):
            rel = []
            for x, y in zip(outs, ref):
                xn, yn = np.asarray(x, np.float32).ravel(), np.asarray(y, np.float32).ravel()
                m = min(xn.size, yn.size)
                if m:
                    denom = float(np.abs(yn[:m]).max() or 1e-6)
                    rel.append(float(np.abs(xn[:m] - yn[:m]).max() / denom))
            result["max_rel_err_vs_ort"] = round(max(rel), 5) if rel else None
            result["parity_pass"] = bool(rel and max(rel) < 0.05)  # FP16 OM vs FP32 ORT 容差
    json.dump(result, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[g3_om] {json.dumps(result, ensure_ascii=False)[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
