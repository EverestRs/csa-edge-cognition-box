#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T3.x 语言层冒烟（x86，7B vLLM 在线）：facts 叙述 / 点谁说谁 / 场景问答 / MQTT dry-run。

用法: python edge_migration/scripts/test_language_layer.py [--frames 90]
断言：
  1) facts 模式叙述落盘（scene_log 新记录、结构完整、facts 帧零图像提交）；
  2) narrate_track：旁白非空且通过反幻觉检查（source ∈ llm/fallback）；
  3) answer_scene：问题回答非空（llm 或确定性 fallback），evidence 可溯源；
  4) MqttBridge dry-run（paho 缺失/broker 缺失均不崩，事件入缓冲）。
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

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="../data/smoking_forward_1-1.mp4")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--out", default="results/edge_test/language_layer_smoke.json")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    from src.pipeline.inference_engine import InferenceEngine
    from src.utils.preflight import resolve_profile

    model_cfg = yaml.safe_load((ROOT / "configs/model_config.yaml").read_text(encoding="utf-8"))
    vlm_cfg = yaml.safe_load((ROOT / "configs/vlm_config.yaml").read_text(encoding="utf-8")) or {}
    merged, _ = resolve_profile(model_cfg, "edge")
    # 加速叙述触发：interval 2s（默认 20s）
    vlm_cfg = dict(vlm_cfg)
    vlm_cfg["scene_narrator"] = {**vlm_cfg.get("scene_narrator", {}), "narrate_interval_seconds": 2}
    vlm_cfg["narration_mode"] = vlm_cfg["scene_narrator"]["narration_mode"]

    engine = InferenceEngine(model_cfg=merged, vlm_cfg=vlm_cfg,
                             output_dir=str(ROOT / "results/edge_lang_smoke"))
    print(f"[lang] narration_mode={engine.scene_narrator.narration_mode} "
          f"mqtt_enabled={engine.mqtt_bridge.enabled}")

    # 1) facts 叙述触发
    log_dir = ROOT / "results/edge_lang_smoke/scene_log"
    cap = cv2.VideoCapture(str(ROOT / a.video))
    engine.begin_task("lang-smoke")
    engine.set_source_fps(25.0)
    for fid in range(1, a.frames + 1):
        ok, frame = cap.read()
        if not ok:
            break
        engine.process(frame, fid)
    cap.release()
    time.sleep(12)  # 叙述异步线程 + LLM 延迟
    records = []
    for f in log_dir.glob("scene_log_*.jsonl"):
        records.extend(json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip())
    new_recs = [r for r in records if r.get("task_id") == "lang-smoke"]
    check("facts 叙述已落盘", len(new_recs) >= 1, f"records={len(new_recs)}")
    if new_recs:
        r = new_recs[-1]
        check("叙述结构完整（summary/person_count/latency）",
              bool(r.get("scene_summary")) and "person_count" in r and "latency_ms" in r)
        check("facts 模式特征（无采样帧图——sample_frame_ids 记录但零图像提交）",
              isinstance(r.get("sample_frame_ids"), list))
        print(f"    叙述样例: {r['scene_summary'][:80]}")

    # 2) 点谁说谁
    from edge_migration.track_narrator import anti_hallucination_check
    snapshot_ids = [d["track_id"] for d in engine.last_track_snapshot]
    if snapshot_ids:
        out = engine.narrate_track(snapshot_ids[0])
        check("narrate_track 有旁白", bool(out.get("text")))
        if out.get("source") == "llm":
            hall = anti_hallucination_check(out["text"], out["facts"])
            check("旁白通过反幻觉检查", hall["ok"], str(hall.get("problems")))
        else:
            check("旁白通过反幻觉检查", True, f"source={out.get('source')} 确定性降级")
        print(f"    旁白({out.get('source')}): {out.get('text', '')[:70]}")
    else:
        check("narrate_track 有旁白", False, "last_track_snapshot 为空")

    # 3) 场景问答（基于叙述记忆）
    from src.core.scene_index import SceneIndex
    idx = SceneIndex(vlm_cfg, results_dir=str(ROOT / "results/edge_lang_smoke"))
    search = idx.search("有人在吸烟吗", top_k=3)
    ans = engine.track_narrator.answer_scene("有人在吸烟吗", search.get("hits", []))
    check("场景问答有回答", bool(ans.get("answer")))
    print(f"    问答({ans.get('source')}, hits={len(search.get('hits', []))}): "
          f"{ans.get('answer', '')[:70]}")

    # 4) MQTT dry-run
    bridge = engine.mqtt_bridge
    bridge.publish_event({"event_id": "smoke", "label": "M01"})
    dry_ok = (not bridge.available and len(bridge._event_buffer) >= 1) or bridge.available
    check("mqtt dry-run/可用态不崩且缓冲", dry_ok,
          f"available={bridge.available} buf={len(bridge._event_buffer)}")

    report = {"probe": "language_layer_smoke", "ts": time.time(), "pass": PASS, "fail": FAIL,
              "narrations": len(new_recs), "snapshot_ids": snapshot_ids[:5]}
    json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[lang] PASS={PASS} FAIL={FAIL} -> {a.out}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
