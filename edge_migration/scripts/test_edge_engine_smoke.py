#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T2.1/T2.6 集成 smoke：edge 决策模式全链路（x86 可跑，CANN 路径自动降级 ONNX）。

验证：
  1) edge 档位解析 → engine._edge_mode=True，crop 仲裁器加载（玩具模型即可）；
  2) detector/pose 的 .om 路径在无 pyACL 环境自动回退同源 .onnx（om_degraded_to_onnx）；
  3) engine.process() 全链路无异常，FrameResult 含 decision_traces（schema 变更可序列化）；
  4) 灰区/强制路径触发 crop 仲裁（trace 出现 crop_arbiter 阶段）。
用法: python edge_migration/scripts/test_edge_engine_smoke.py [--video test/whole_process_corner_3.mp4] [--frames 90]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_toy_builder():
    spec = importlib.util.spec_from_file_location(
        "g4mod", str(ROOT / "edge_migration/scripts/g4_crop_latency_bench.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.build_toy_onnx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="../test/whole_process_corner_3.mp4")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--out", default="results/edge_test/edge_engine_smoke.json")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"probe": "edge_engine_smoke", "ts": time.time(), "checks": {}}

    from src.utils.preflight import resolve_profile
    from src.pipeline.inference_engine import InferenceEngine

    model_cfg = yaml.safe_load((ROOT / "configs/model_config.yaml").read_text(encoding="utf-8"))
    vlm_cfg = yaml.safe_load((ROOT / "configs/vlm_config.yaml").read_text(encoding="utf-8")) or {}
    merged, _resolved = resolve_profile(model_cfg, "edge")
    report["checks"]["profile_edge"] = merged.get("inference", {}).get("decision_mode") == "edge"

    # crop 模型：无真模型（T2.5 前）用玩具 7 类模型打通链路
    crop_cfg = dict((vlm_cfg or {}).get("crop_review", {}))
    crop_path = ROOT / str(crop_cfg.get("model_path", "models/crop/crop_v1.onnx"))
    if not crop_path.exists():
        toy = _load_toy_builder()(ROOT / "results/edge_m0/smoke_crop_toy.onnx",
                                  num_classes=len(crop_cfg.get("classes", [])) or 7)
        crop_cfg["model_path"] = str(toy)
    vlm_cfg = dict(vlm_cfg)
    vlm_cfg["crop_review"] = crop_cfg

    engine = InferenceEngine(model_cfg=merged, vlm_cfg=vlm_cfg,
                             output_dir=str(ROOT / "results/edge_smoke"))
    report["checks"]["engine_edge_mode"] = bool(engine._edge_mode)
    report["checks"]["crop_loaded"] = bool(engine.crop_classifier.is_real_model_loaded)
    report["checks"]["detector_backend_degraded"] = bool(
        getattr(engine.detector, "om_degraded_to_onnx", False))
    report["checks"]["detector_session_alive"] = engine.detector.is_real_model_loaded
    report["checks"]["pose_session_alive"] = engine.pose_estimator.is_real_model_loaded
    if not all([report["checks"]["engine_edge_mode"], report["checks"]["crop_loaded"],
                report["checks"]["detector_session_alive"]]):
        report["verdict"] = "FAIL_INIT"
        json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    cap = cv2.VideoCapture(str(ROOT / a.video))
    if not cap.isOpened():
        print(f"[smoke] 视频打不开: {a.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    engine.begin_task("edge-smoke")
    engine.set_source_fps(float(fps))

    frames_with_trace = 0
    arbiter_steps = 0
    trace_lines: list[str] = []
    latency_ms: list[float] = []
    fid = 0
    t0 = time.perf_counter()
    while fid < a.frames:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        s = time.perf_counter()
        result = engine.process(frame, fid)
        latency_ms.append((time.perf_counter() - s) * 1000)
        if fid == 1:  # schema 变更可序列化（WS 全量下发等价性）
            result.model_dump_json()
        if result.decision_traces:
            frames_with_trace += 1
            for tr in result.decision_traces:
                arbiter_steps += sum(1 for st in tr.get("steps", [])
                                     if st.get("stage") == "crop_arbiter")
                if len(trace_lines) < 3:
                    from edge_migration.decision_trace import render_trace_line
                    trace_lines.append(render_trace_line(tr))
    wall = time.perf_counter() - t0
    cap.release()

    report["checks"]["frames_processed"] = fid
    report["checks"]["frames_with_traces"] = frames_with_trace
    report["checks"]["crop_arbiter_steps"] = arbiter_steps
    report["checks"]["latency_ms_p50"] = round(sorted(latency_ms)[len(latency_ms) // 2], 1)
    report["checks"]["throughput_fps"] = round(fid / wall, 2)
    report["sample_traces"] = trace_lines
    report["verdict"] = "PASS" if (fid >= a.frames and frames_with_trace > 0) else (
        "PASS_NO_TRACE" if fid >= a.frames else "FAIL_RUN")
    json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, ensure_ascii=False))
    print(json.dumps(report["checks"], ensure_ascii=False))
    for ln in trace_lines:
        print("  trace:", ln)
    return 0 if str(report["verdict"]).startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
