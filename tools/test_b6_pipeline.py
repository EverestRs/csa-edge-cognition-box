"""B6 离线验证：同步 vs 管道两种模式的输出一致性 + 吞吐对比。

用法: python tools/test_b6_pipeline.py <video_path> [n_frames]
三组运行：
  1) sync 全量串行（正确性基线 + 吞吐基线）
  2) pipe 慢速提交（350ms/帧，低于瓶颈阶段耗时 → 几乎不丢帧，全量比对一致性）
  3) pipe 快速提交（40ms/帧 ≈ 25fps 实时输入 → 测真实吞吐与丢帧率）
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import cv2
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.inference_engine import InferenceEngine  # noqa: E402


def load_cfgs():
    model_cfg = yaml.safe_load(open(BASE_DIR / "configs" / "model_config.yaml", encoding="utf-8"))
    vlm_cfg = yaml.safe_load(open(BASE_DIR / "configs" / "vlm_config.yaml", encoding="utf-8"))
    return model_cfg, vlm_cfg


def make_result_record(r):
    return {
        "n_det": len(r.detections),
        "labels": tuple(d.behavior.type for d in r.detections),
        "alert": r.alert.triggered,
        "stage_ms": dict(r.metrics.get("pipeline_stage_ms", {})),
    }


def run_sync(model_cfg, vlm_cfg, frames, frame_ids):
    engine = InferenceEngine(model_cfg=model_cfg, vlm_cfg=vlm_cfg,
                             output_dir=str(BASE_DIR / "results" / "b6_test_sync"))
    out = {}
    t0 = time.perf_counter()
    for frame, fid in zip(frames, frame_ids):
        out[fid] = make_result_record(engine.process(frame, fid))
    wall = time.perf_counter() - t0
    engine.close()
    return out, wall


def run_pipeline(model_cfg, vlm_cfg, frames, frame_ids, pace_s: float, tag: str):
    engine = InferenceEngine(model_cfg=model_cfg, vlm_cfg=vlm_cfg,
                             output_dir=str(BASE_DIR / "results" / f"b6_test_{tag}"))
    assert engine._pipeline_enabled
    engine.begin_task(f"b6_{tag}")
    out = {}
    t0 = time.perf_counter()
    submitted = 0
    for frame, fid in zip(frames, frame_ids):
        engine.submit_frame(frame, fid)
        submitted += 1
        for r in engine.drain_results():
            out[r.frame_id] = make_result_record(r)
        time.sleep(pace_s)
    # 等待在途帧排空（completed + dropped == submitted 即收敛）
    deadline = time.time() + 120
    while time.time() < deadline:
        for r in engine.drain_results():
            out[r.frame_id] = make_result_record(r)
        stats = engine.pipeline_stats()
        if len(out) + stats["dropped"] >= submitted and all(
                getattr(engine, q) .qsize() == 0 for q in ("_pipe_q1", "_pipe_q2", "_pipe_q3")):
            break
        time.sleep(0.05)
    wall = time.perf_counter() - t0
    stats = engine.pipeline_stats()
    engine.begin_task(f"{tag}_end")  # 隔离验证：残留应为空
    assert engine.drain_results() == [], "begin_task 后残留结果应为空"
    engine.close()
    return out, wall, stats


def main():
    video_path = sys.argv[1]
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    cap = cv2.VideoCapture(video_path)
    frames, frame_ids = [], []
    fid = 0
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        if fid % 4 == 0:  # 对齐 realtime profile frame_stride=4
            frames.append(frame)
            frame_ids.append(fid)
    cap.release()
    n = len(frames)
    print(f"video={Path(video_path).name} processed_frames={n}")

    model_cfg, vlm_cfg = load_cfgs()
    vlm_cfg = copy.deepcopy(vlm_cfg)
    vlm_cfg["use_vllm"] = False  # 确定性：测试不触发 VLM
    vlm_cfg.setdefault("scene_narrator", {})["enabled"] = False
    vlm_cfg.setdefault("alert_adjudicator", {})["enabled"] = False

    pipe_cfg = copy.deepcopy(model_cfg)
    pipe_cfg.setdefault("inference", {})["pipeline"] = {"enabled": True, "queue_size": 4}

    print("[1/3] 同步基线...")
    sync_out, sync_wall = run_sync(model_cfg, vlm_cfg, frames, frame_ids)
    print(f"  sync: {len(sync_out)} 帧, wall={sync_wall:.1f}s, 吞吐={n/sync_wall:.2f} 帧/s")

    print("[2/3] 管道·慢速提交（350ms/帧，全量一致性）...")
    pipe_out, pipe_wall, pstats = run_pipeline(pipe_cfg, vlm_cfg, frames, frame_ids, 0.35, "slow")
    print(f"  pipe-slow: {len(pipe_out)} 帧, dropped={pstats['dropped']}, wall={pipe_wall:.1f}s")

    print("[3/3] 管道·快速提交（40ms/帧≈25fps 实时）...")
    fast_out, fast_wall, fstats = run_pipeline(pipe_cfg, vlm_cfg, frames, frame_ids, 0.04, "fast")
    print(f"  pipe-fast: {len(fast_out)} 帧, dropped={fstats['dropped']}, wall={fast_wall:.1f}s, "
          f"吞吐={len(fast_out)/fast_wall:.2f} 帧/s")

    # 一致性（慢速跑基本无丢帧 → 与 sync 全量比）
    common = sorted(set(sync_out) & set(pipe_out))
    mismatch = sum(1 for f in common
                   if (sync_out[f]["n_det"], sync_out[f]["labels"], sync_out[f]["alert"])
                   != (pipe_out[f]["n_det"], pipe_out[f]["labels"], pipe_out[f]["alert"]))
    print("\n===== 结论 =====")
    print(f"一致性: {len(common)-mismatch}/{len(common)} 帧完全一致"
          + (f"（管道丢帧 {pstats['dropped']} 帧未比对）" if pstats["dropped"] else "（全量）"))
    # 阶段耗时（来自 sync 结果的 metrics）
    if sync_out:
        first = next(iter(sync_out.values()))["stage_ms"]
        print(f"阶段耗时样例: {first}")
    print(f"吞吐: sync {n/sync_wall:.2f} 帧/s → pipe-fast {len(fast_out)/fast_wall:.2f} 帧/s "
          f"(完成率 {100*len(fast_out)/n:.0f}% vs 串行100%按帧均摊 {100*(n/sync_wall)/(n/sync_wall+0)*1:.0f}%)")


if __name__ == "__main__":
    main()
