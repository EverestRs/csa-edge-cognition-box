#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""edge_migration 包内离线自测（x86 即可，全部无网络/NPU 依赖）。

验证目标：模块可导入、降级路径成立、纯逻辑正确（对齐 ARCH §12 优雅降级约定）。
用法: /home/user/.conda/envs/VdAnalyst/bin/python edge_migration/selftest.py
退出码 0 = 全部通过。CANN 相关路径仅验证"缺 pyACL 时不崩"（静态检查级）。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


def _load_g4_builder():
    spec = importlib.util.spec_from_file_location(
        "g4mod", str(ROOT / "edge_migration/scripts/g4_crop_latency_bench.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.build_toy_onnx


def t1_decision_trace() -> None:
    from edge_migration.decision_trace import DecisionTrace, is_valid_trace, render_trace_line
    t = DecisionTrace("P003", 412)
    t.add("pose_classifier", "M01", 0.61).add("crop_arbiter", "Normal", 0.78).add("final", "Normal")
    d = t.to_dict()
    check("trace 有效结构", is_valid_trace(d))
    check("final_label", t.final_label == "Normal")
    check("渲染含中文阶段名", "主路径" in render_trace_line(d) and "Crop复核" in render_trace_line(d))
    bad = {"track_id": "P1", "frame_id": 1,
           "steps": [{"stage": "pose_classifier", "label": "X", "conf": 0.5}]}
    check("缺 final 判无效", not is_valid_trace(bad))
    try:
        DecisionTrace("P1", 1).add("bogus_stage", "x")
        check("非法 stage 拒绝", False)
    except ValueError:
        check("非法 stage 拒绝", True)


def t2_fact_compiler() -> None:
    from edge_migration.fact_compiler import (build_event_facts, build_scene_facts,
                                              build_track_facts, facts_to_text)
    tf = build_track_facts(track_id="P001", frame_id=480, window_seconds=86.0,
                           label_history=[("S01", 0.61)] * 9 + [("Normal", 0.7)] * 3,
                           bboxes=[[300, 200, 60, 150]] * 12, crop_verdicts=["Normal"],
                           speed_px_per_s=41.2)
    check("track 主导标签", tf["dominant_label"] == "S01")
    check("track 变化次数", tf["label_changes"] == 1)
    check("bbox 位移统计", tf["bbox_summary"]["center_shift_px"] == 0.0)
    sf = build_scene_facts(frame_id=480, fps=4.8, tracks=[
        {"track_id": "P001", "bbox": [0, 0, 5, 5], "label": "S01", "confidence": 0.61},
        {"track_id": "P002", "bbox": [0, 0, 5, 5], "label": "Normal", "confidence": 0.83}],
        crop_verdicts={"P001": "Normal"})
    check("scene 人数", sf["person_count"] == 2)
    ef = build_event_facts(alert={"type": "h03_detected", "label": "H03", "confidence": 0.66,
                                  "frame_id": 480}, scene_facts=sf)
    check("event 摘要含 scene", ef["scene"]["person_count"] == 2)
    txt = facts_to_text(tf)
    check("降级旁白非空且含行为词", bool(txt.get("summary")) and "徘徊" in txt["summary"])
    json.dumps({"tf": tf, "sf": sf, "ef": ef}, ensure_ascii=False)  # 可序列化
    check("facts 全 JSON 可序列化", True)


def t3_backend() -> None:
    from edge_migration.inference_backend import NullBackend, OnnxRuntimeBackend, create_backend
    b = create_backend("不存在的模型.onnx")
    check("缺文件→NullBackend 不抛", isinstance(b, NullBackend) or not b.is_real)
    build_toy = _load_g4_builder()
    toy = build_toy(Path("results/edge_m0/selftest_toy.onnx"))
    r = OnnxRuntimeBackend(str(toy))
    check("toy ONNX 可加载", r.is_real, r.last_error)
    if r.is_real:
        outs = r.run(np.random.rand(1, 3, 224, 224).astype(np.float32))
        check("toy 输出形状 (1,7)", bool(outs) and tuple(np.asarray(outs[0]).shape) == (1, 7))
    om = create_backend("models/detector/yolov8n_310B.om", fallback_onnx=str(toy))
    check("OM 缺失→fallback 生效", getattr(om, "kind", "") == "onnxruntime" and om.is_real)


def t4_crop_classifier() -> None:
    from edge_migration.crop_classifier import CropClassifier
    cfg = {"model_path": "results/edge_m0/selftest_toy.onnx",
           "classes": ["Normal", "M01", "M02", "H03", "Wave", "Clap", "Throw"],
           "accept_conf": 0.6}
    c = CropClassifier(cfg)
    check("crop 真实模型加载", c.is_real_model_loaded, str(c.last_debug))
    frame = np.random.default_rng(1).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    vs = c.classify(frame, [[10, 10, 60, 120], [200, 30, 50, 50]],
                    track_ids=["P1", "P2"], frame_id=100)
    names = {"Normal", "M01", "M02", "H03", "Wave", "Clap", "Throw"}
    check("verdict 类名合法", len(vs) == 2 and all(v.cls in names for v in vs))
    check("缓存节流命中", c.cached("P1", 105) is not None and c.cached("P1", 999) is None)
    bad = CropClassifier({"model_path": "缺失.onnx"})
    check("缺模型→Unknown 不误伤", bad.classify(frame, [[0, 0, 10, 10]])[0].cls == "Unknown")


def t5_telemetry() -> None:
    from edge_migration import npu_telemetry as t
    check("x86 read()=None", t.read() is None)
    m = t.read_or_mock(allow_mock=True)
    check("mock 分支", bool(m.get("mock")) and m.get("power_w") is not None)
    real = t.read_or_mock()
    check("无 mock 时显式不可用", real.get("available") is False or not real.get("mock"))


def t6_mqtt_dry() -> None:
    from edge_migration.mqtt_bridge import MqttBridge
    b = MqttBridge({"enabled": True, "box_id": "selftest"})
    b.start()  # paho 缺失/无 broker：必须不抛
    b.publish_event({"event_id": "e1", "label": "H03"})
    b.publish_status({"fps": 3.1})
    check("dry_run 缓冲事件", len(b._event_buffer) == (1 if not b.available else 0))
    check("start 不抛且记录 debug", isinstance(b.last_debug, dict))
    b.stop()


def t7_narrator_fallback() -> None:
    from edge_migration.fact_compiler import build_track_facts, facts_to_text
    from edge_migration.track_narrator import TrackNarrator, anti_hallucination_check
    tf = build_track_facts(track_id="P001", frame_id=10, window_seconds=30.0,
                          label_history=[("M01", 0.6)] * 5, bboxes=[[0, 0, 40, 100]] * 5)
    n = TrackNarrator({"vllm_url": "http://127.0.0.1:1/v1/chat/completions",
                       "narrate_timeout_s": 2.0})
    out = n.narrate_track(tf)
    check("端点不可达→fallback", out["source"] == "fallback" and bool(out["text"]))
    check("fallback 文本可过反幻觉", anti_hallucination_check(out["text"], tf)["ok"])
    bad = anti_hallucination_check("刚才47人吸烟并翻越围栏", tf)
    check("编造数字/行为被拦截", not bad["ok"] and len(bad["problems"]) >= 1)


def main() -> int:
    print(f"[selftest] python={sys.version.split()[0]} root={ROOT}")
    t0 = time.time()
    for fn in (t1_decision_trace, t2_fact_compiler, t3_backend, t4_crop_classifier,
               t5_telemetry, t6_mqtt_dry, t7_narrator_fallback):
        print(f"-- {fn.__name__}")
        try:
            fn()
        except Exception as exc:
            check(f"{fn.__name__} 未抛异常", False, f"{type(exc).__name__}: {exc}")
    print(f"[selftest] PASS={PASS} FAIL={FAIL} ({time.time()-t0:.1f}s)")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
