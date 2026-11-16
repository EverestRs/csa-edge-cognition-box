#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G4 探针：crop 多分类器时延基线（T1.5，随机权重玩具模型即可测速）。

x86 可跑（ORT CPU）。真模型（T2.5 训练产物）出来后用 --model 传入复跑。
用法: python edge_migration/scripts/g4_crop_latency_bench.py [--model models/crop/crop_v1.onnx] [--runs 100]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_toy_onnx(path: Path, num_classes: int = 7, input_size: int = 224) -> Path:
    """构造 ~MobileNet 前端量级的随机权重网络（Conv→Relu→GAP→Gemm）用于纯测速。"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(3)
    w1 = rng.standard_normal((32, 3, 3, 3), dtype=np.float32) * 0.1
    b1 = np.zeros(32, dtype=np.float32)
    w2 = rng.standard_normal((num_classes, 32), dtype=np.float32) * 0.1
    b2 = np.zeros(num_classes, dtype=np.float32)
    nodes = [
        helper.make_node("Conv", ["x", "w1", "b1"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("GlobalAveragePool", ["r1"], ["g1"]),
        helper.make_node("Flatten", ["g1"], ["f1"], axis=1),
        helper.make_node("Gemm", ["f1", "w2", "b2"], ["y"], transB=1),
    ]
    graph = helper.make_graph(
        nodes, "toy_crop",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, input_size, input_size])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, num_classes])],
        initializer=[numpy_helper.from_array(w1, "w1"), numpy_helper.from_array(b1, "b1"),
                     numpy_helper.from_array(w2, "w2"), numpy_helper.from_array(b2, "b2")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # 对齐 onnxruntime 支持的 IR 上限，避免新版 onnx 默认 IR 过新
    onnx.save(model, str(path))
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="", help="crop ONNX 路径；缺省自动构建玩具模型")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2, help="A55 弱核：建议 2")
    ap.add_argument("--out", default="results/edge_m0/g4_crop_latency.json")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    model_path = Path(a.model) if a.model else Path("results/edge_m0/toy_crop_v0.onnx")
    toy = False
    if not model_path.exists():
        model_path = build_toy_onnx(model_path, input_size=a.input_size)
        toy = True

    from edge_migration.inference_backend import create_backend
    backend = create_backend(str(model_path),
                             runtime={"intra_op_num_threads": a.threads})
    if not backend.is_real:
        json.dump({"probe": "g4", "status": "FAIL",
                   "reason": getattr(backend, "last_error", "load_failed")},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print("[g4] FAIL: 模型加载失败")
        return 1

    rng = np.random.default_rng(11)
    x = rng.random((1, 3, a.input_size, a.input_size), dtype=np.float32)
    for _ in range(8):
        backend.run(x)
    times = []
    for _ in range(a.runs):
        s = time.perf_counter()
        backend.run(x)
        times.append((time.perf_counter() - s) * 1000)
    arr = np.array(times)
    out = {"probe": "g4_crop_latency", "ts": time.time(), "model": str(model_path),
           "toy_weights": toy, "input_size": a.input_size, "threads": a.threads,
           "crop_ms_p50": round(float(np.percentile(arr, 50)), 2),
           "crop_ms_p95": round(float(np.percentile(arr, 95)), 2),
           "pass_20ms": bool(np.percentile(arr, 95) <= 20.0 * (a.threads / 2)),
           "note": "toy 权重仅测速；per-class P/R 待 T2.5 真模型复跑"}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[g4] {out['crop_ms_p50']}ms p50 / {out['crop_ms_p95']}ms p95 -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
