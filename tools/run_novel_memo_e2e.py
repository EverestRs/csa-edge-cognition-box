from __future__ import annotations

"""unknown_events 真机 E2E（ROADMAP 阶段 1/2 验收，§6）。

驱动真实 InferenceEngine（检测/跟踪/姿态/主路径 + vLLM 旁路）跑三段 HMDB51
片段（训练体系外）：wave_1 / wave_2（同类）/ throw_1（异类）。

验收点（路线图 §6）:
  1. 首次遇到即产出描述且关键属性正确（Stage 1）：novel memo 落盘、is_novel、
     motion_pattern/body_parts 等字段非空；
  2. 第二段同类视频认出是同一原型（Stage 2 再识别）：wave_1 与 wave_2 的备忘
     归入同一 prototype，throw 归入另一 prototype。

用法:
  python tools/run_novel_memo_e2e.py \
    --clips data_external/unknown_events/wave/A.txt wave_1.avi \
    ...
简化: 内置默认三段，--list 查看可选片段。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.inference_engine import InferenceEngine

ROOT = Path(__file__).resolve().parents[1]
UE = ROOT / "data_external" / "unknown_events"


def pick(clause: str, idx: int) -> Path:
    files = sorted(UE.glob(f"{clause}/*.avi"))
    return files[idx % len(files)]


def run_clip(engine: InferenceEngine, clip: Path, task_id: str, hint_frame_ratio: float = 0.6) -> dict:
    engine.begin_task(task_id)
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    engine.set_source_fps(fps)
    fid = 0
    hinted = False
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        engine.process(frame, fid)
        if not hinted and fid >= int(hint_frame_ratio * max(1, cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            # 模拟叙述器报告一次"未归类活动"（触发器 A；文本刻意不含闭集关键词）
            engine.novel_behavior_memo.note_narrator_activities(
                ["画面中有人员做出尚未归类的手势动作"], fid)
            hinted = True
        fid += 1
    cap.release()
    return {"clip": clip.name, "frames": fid, "fps": round(fps, 1), "seconds": round(time.time() - t0, 1)}


def drain(engine: InferenceEngine, timeout_s: float = 180.0) -> None:
    """等 memo 队列排空（主动驱动：失败重排队后由 poke 在退避到期时重派发）。"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        engine.novel_behavior_memo.poke()
        st = engine.novel_behavior_memo.status()
        if not st["inflight"] and st["queue_length"] == 0:
            time.sleep(2.0)
            return
        time.sleep(2.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clause", nargs=3, default=["wave", "wave", "throw"],
                    help="三段片段的类别（默认 wave/wave/throw）")
    ap.add_argument("--index", nargs=3, type=int, default=[0, 1, 0])
    ap.add_argument("--output", default="results/e2e_novel")
    args = ap.parse_args()

    model_cfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    vlm_cfg = yaml.safe_load((ROOT / "configs" / "vlm_config.yaml").read_text(encoding="utf-8"))
    engine = InferenceEngine(model_cfg, vlm_cfg, output_dir=args.output)

    clips = []
    for clause, idx in zip(args.clause, args.index):
        clips.append(pick(clause, idx))
    print("E2E 片段:", [c.name for c in clips])

    stats = []
    for i, clip in enumerate(clips):
        print(f"[{i + 1}/3] 处理 {clip.name} ...", flush=True)
        stats.append(run_clip(engine, clip, f"e2e_{i + 1}_{clip.stem}"))
        drain(engine, timeout_s=120.0)  # 片段间排空：begin_task 会清队列，未派发的备忘会丢
    drain(engine)
    for s in stats:
        print(" ", s)

    # ── 验收 1：备忘描述 ──
    memo = engine.novel_behavior_memo
    memos = []
    for i in range(3):
        memos.extend(memo.iter_memos(f"e2e_{i + 1}_{clips[i].stem}"))
    print(f"\n=== 验收1: 共 {len(memos)} 条备忘 ===")
    for m in memos:
        mm = m["memo"]
        print(f"  [{m['task_id']}] is_novel={m['is_novel']} conf={m['confidence']:.2f} "
              f"模式={mm.get('motion_pattern')} 部位={mm.get('body_parts')} | {mm.get('summary', '')[:60]}")

    # ── 验收 2：原型归簇（wave 同簇、throw 异簇）──
    protos = engine.behavior_prototypes
    wave_ids, throw_ids = set(), set()
    for m in memos:
        pid = m.get("prototype_id")
        if not pid:
            continue
        if m["task_id"].startswith("e2e_1") or m["task_id"].startswith("e2e_2"):
            wave_ids.add(pid)
        elif m["task_id"].startswith("e2e_3"):
            throw_ids.add(pid)
    print("\n=== 验收2: 原型归簇 ===")
    print(f"  wave 备忘归入: {sorted(wave_ids)}  throw 备忘归入: {sorted(throw_ids)}")
    same = len(wave_ids) == 1 and len(throw_ids) == 1 and wave_ids != throw_ids
    print(f"  wave 同簇且 throw 异簇: {'✓ 通过' if same else '✗ 未通过'}")
    for p in protos.list_prototypes():
        print(f"  {p['proto_id']} [{p['status']}] {p.get('name') or p.get('proposed_name') or '—'} "
              f"occ={p['occurrences']} members={len(p['member_memo_keys'])}")

    out = Path(args.output) / "e2e_report.json"
    out.write_text(json.dumps({
        "clips": stats, "memos": len(memos),
        "wave_proto_ids": sorted(wave_ids), "throw_proto_ids": sorted(throw_ids),
        "stage2_same_cluster_and_throw_different": bool(same),
        "prototype_status": protos.status(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
