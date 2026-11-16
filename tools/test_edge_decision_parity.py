#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T2.6 出口指标：edge 决策 vs server 决策 方向一致率（03 §2）。

设计（控制变量）：
  - 两遍共用 edge 档位框架（同感知步长、同 SlowFast 禁用），唯一差异 = 仲裁器
    （server: 7B VLM person-crop 复核 / edge: crop 多分类仲裁）；
  - server 遍把引擎的异步复核 submit patch 成同步执行——消除回写时序噪声，
    两遍轨迹逐帧可比（VLM 失败仍走内部 try/except 降级，不阻塞）；
  - ByteTrack 同输入确定性 → track_id 跨遍对齐；逐帧逐 track 比对 final 标签；
  - 指标：标签级一致率 + 方向级一致率（Normal vs 非Normal，验收线 ≥90%）+ trace schema 校验率。

用法: python tools/test_edge_decision_parity.py --video ../data/smoking_forward_1-1.mp4 --frames 200
前置: 7B vLLM 在线（curl localhost:8001/v1/models）；models/crop/crop_v1.onnx 已产出（T2.5）。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_migration.decision_trace import is_valid_trace  # noqa: E402
from src.pipeline.inference_engine import InferenceEngine  # noqa: E402
from src.utils.preflight import resolve_profile  # noqa: E402


def _sync_submit(fn, *args, **kwargs):
    """同步执行异步复核任务（parity 专用：消除回写时序噪声）。"""
    fn(*args, **kwargs)
    return None


def run_pass(model_cfg: dict, vlm_cfg: dict, frames: list, *, sync_vlm: bool,
             decision_mode: str) -> tuple[dict, list, dict]:
    mc = copy.deepcopy(model_cfg)
    mc.setdefault("inference", {})["decision_mode"] = decision_mode  # 两遍唯一差异点
    engine = InferenceEngine(model_cfg=mc,
                             vlm_cfg=copy.deepcopy(vlm_cfg),
                             output_dir=str(ROOT / "results/edge_parity"))
    if sync_vlm:
        engine._vlm_async_executor.submit = _sync_submit  # type: ignore[method-assign]
    calls = {"vlm": 0}
    orig_clip = engine.vlm_reviewer.review_clip

    def counting_clip(*a, **k):
        calls["vlm"] += 1
        return orig_clip(*a, **k)

    engine.vlm_reviewer.review_clip = counting_clip  # type: ignore[method-assign]
    engine.begin_task("parity")
    engine.set_source_fps(25.0)
    labels: dict[tuple[int, str], tuple[str, float]] = {}
    traces = []
    for fid, frame in enumerate(frames, start=1):
        result = engine.process(frame, fid)
        for det in result.detections:
            labels[(fid, str(det.track_id))] = (det.behavior.type, float(det.behavior.confidence))
        traces.extend(result.decision_traces)
    return labels, traces, calls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="../data/smoking_forward_1-1.mp4")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--require", type=float, default=0.90, help="方向一致率验收线")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "results/edge_test" / \
        f"decision_parity_{Path(a.video).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    model_cfg = yaml.safe_load((ROOT / "configs/model_config.yaml").read_text(encoding="utf-8"))
    vlm_cfg = yaml.safe_load((ROOT / "configs/vlm_config.yaml").read_text(encoding="utf-8")) or {}
    merged, _ = resolve_profile(model_cfg, "edge")  # 两遍共用：感知步长+SlowFast禁用

    cap = cv2.VideoCapture(str(ROOT / a.video))
    if not cap.isOpened():
        print(f"[parity] 视频打不开: {a.video}")
        return 1
    frames = []
    while len(frames) < a.frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"[parity] 帧 {len(frames)}，开始 server 遍（同步 VLM 复核，7B 在线）…")
    t0 = time.time()
    server_labels, _, s_calls = run_pass(merged, vlm_cfg, frames, sync_vlm=True,
                                         decision_mode="server")
    print(f"[parity] server 遍完成 {time.time()-t0:.0f}s（{len(server_labels)} 个判定，"
          f"VLM 复核 {s_calls['vlm']} 次）")
    print("[parity] 开始 edge 遍（crop 仲裁）…")
    t0 = time.time()
    edge_labels, traces, e_calls = run_pass(merged, vlm_cfg, frames, sync_vlm=False,
                                            decision_mode="edge")
    print(f"[parity] edge 遍完成 {time.time()-t0:.0f}s（{len(edge_labels)} 个判定，"
          f"VLM 复核 {e_calls['vlm']} 次）")

    keys = sorted(set(server_labels) & set(edge_labels))
    same_label = sum(1 for k in keys if server_labels[k][0] == edge_labels[k][0])

    def direction(l: str) -> str:
        return "Normal" if l == "Normal" else "Abnormal"

    same_dir = sum(1 for k in keys
                   if direction(server_labels[k][0]) == direction(edge_labels[k][0]))
    diffs = [{"frame_id": k[0], "track_id": k[1],
              "server": server_labels[k][0], "edge": edge_labels[k][0]}
             for k in keys if server_labels[k][0] != edge_labels[k][0]]
    trace_valid = sum(1 for t in traces if is_valid_trace(t))
    only_s = len(set(server_labels) - set(edge_labels))
    only_e = len(set(edge_labels) - set(server_labels))

    dir_rate = same_dir / max(len(keys), 1)
    from collections import Counter
    report = {
        "probe": "decision_parity", "ts": time.time(), "video": a.video, "frames": len(frames),
        "compared": len(keys), "label_match_rate": round(same_label / max(len(keys), 1), 4),
        "direction_match_rate": round(dir_rate, 4), "require": a.require,
        "verdict": "PASS" if dir_rate >= a.require else "FAIL",
        "vlm_calls_server": s_calls["vlm"], "vlm_calls_edge": e_calls["vlm"],
        "only_in_server": only_s, "only_in_edge": only_e,
        "trace_schema_valid_rate": round(trace_valid / max(len(traces), 1), 4),
        "diff_samples": diffs[:20],
        # 差异方向汇总（server标签 -> edge标签 计数）
        "diff_summary": {f"{s}->{e}": n for (s, e), n in
                         Counter((d["server"], d["edge"]) for d in diffs).most_common(10)},
    }
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "diff_samples"},
                     ensure_ascii=False, indent=2))
    for d in diffs[:6]:
        print(f"  diff: f{d['frame_id']} {d['track_id']}: server={d['server']} edge={d['edge']}")
    print(f"[parity] -> {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
