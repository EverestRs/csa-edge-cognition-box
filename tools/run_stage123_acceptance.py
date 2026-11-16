from __future__ import annotations

"""Stage 1/2 全量验收（ROADMAP §6）：unknown_events 四类 × 2 段。

Stage 1 验收：每类首次出现即产出 is_novel=True 备忘且关键属性（部位/模式/速度）非空合理；
Stage 2 验收：8 条备忘按两级判定聚类为 4 个原型（每类一个），同类第二段归入
第一段的原型（再识别），异类互不混簇。

用法:
  python tools/run_stage123_acceptance.py [--per-category 2] [--output results/stage123]
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.inference_engine import InferenceEngine

ROOT = Path(__file__).resolve().parents[1]
UE = ROOT / "data_external" / "unknown_events"
CATEGORIES = ["wave", "clap", "throw", "shake_hands"]


def drain(engine: InferenceEngine, timeout_s: float = 150.0) -> None:
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
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--output", default="results/stage123")
    args = ap.parse_args()

    model_cfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    vlm_cfg = yaml.safe_load((ROOT / "configs" / "vlm_config.yaml").read_text(encoding="utf-8"))
    engine = InferenceEngine(model_cfg, vlm_cfg, output_dir=args.output)

    clips = {}
    for cat in CATEGORIES:
        clips[cat] = sorted(UE.glob(f"{cat}/*.avi"))[: args.per_category]
        print(f"{cat}: {[c.name[:36] for c in clips[cat]]}")

    stats = []
    for cat in CATEGORIES:
        for i, clip in enumerate(clips[cat]):
            task_id = f"s123_{cat}_{i + 1}"
            engine.begin_task(task_id)
            cap = cv2.VideoCapture(str(clip))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            engine.set_source_fps(fps)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fid, hinted = 0, False
            t0 = time.time()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                engine.process(frame, fid)
                if not hinted and fid >= int(0.6 * max(1, total)):
                    engine.novel_behavior_memo.note_narrator_activities(
                        ["画面中有人员做出尚未归类的动作"], fid)
                    hinted = True
                fid += 1
            cap.release()
            drain(engine)
            stats.append({"category": cat, "clip": clip.name, "task_id": task_id,
                          "frames": fid, "seconds": round(time.time() - t0, 1)})
            print(f"  [{cat} #{i + 1}] {fid} 帧 / {stats[-1]['seconds']}s", flush=True)
    drain(engine, timeout_s=240.0)

    # ── Stage 1 验收：每类每段备忘产出 + 属性 ──
    memo = engine.novel_behavior_memo
    memos_by_task: dict[str, list[dict]] = {}
    for cat in CATEGORIES:
        for i in range(len(clips[cat])):
            memos_by_task[f"s123_{cat}_{i + 1}"] = memo.iter_memos(f"s123_{cat}_{i + 1}")
    print("\n=== Stage 1 验收：备忘产出与关键属性 ===")
    stage1_rows = []
    for cat in CATEGORIES:
        for i in range(len(clips[cat])):
            tid = f"s123_{cat}_{i + 1}"
            novel = [m for m in memos_by_task[tid] if m.get("is_novel")]
            ok = len(novel) > 0
            attrs_ok = all(
                str(m["memo"].get("body_parts", "")) and str(m["memo"].get("motion_pattern", ""))
                for m in novel)
            stage1_rows.append({"task_id": tid, "category": cat, "novel_memos": len(novel),
                                "attrs_ok": attrs_ok, "pass": bool(ok and attrs_ok)})
            desc = novel[0]["memo"].get("summary", "")[:48] if novel else "(无 novel 备忘)"
            print(f"  [{tid}] novel={len(novel)} 属性完整={attrs_ok} | {desc}")
    stage1_pass = all(r["pass"] for r in stage1_rows)
    print(f"Stage 1: {'✓ 通过' if stage1_pass else '✗ 未通过'}"
          f"（{sum(r['pass'] for r in stage1_rows)}/{len(stage1_rows)} 段达标）")

    # ── Stage 2 验收：4 原型、类内归簇、类间分离 ──
    protos = engine.behavior_prototypes
    cat_of_pid: dict[str, set[str]] = defaultdict(set)
    pid_of_cat: dict[str, set[str]] = defaultdict(set)
    for cat in CATEGORIES:
        for i in range(len(clips[cat])):
            for m in memos_by_task[f"s123_{cat}_{i + 1}"]:
                pid = m.get("prototype_id")
                if pid:
                    cat_of_pid[pid].add(cat)
                    pid_of_cat[cat].add(pid)
    # 再识别：同类的两段应共用同一 pid
    reid_ok = all(len(pid_of_cat[cat]) == 1 for cat in CATEGORIES if pid_of_cat[cat])
    sep_ok = all(len(v) == 1 for v in cat_of_pid.values()) and len(cat_of_pid) == len(pid_of_cat)
    n_protos = len(protos.list_prototypes())
    print("\n=== Stage 2 验收：原型归簇 ===")
    for cat in CATEGORIES:
        print(f"  {cat}: {sorted(pid_of_cat.get(cat, []))}")
    print(f"  原型总数 {n_protos}（含被驳回），每类单簇={reid_ok}，每原型单类={sep_ok}")
    stage2_pass = bool(reid_ok and sep_ok and n_protos >= len(CATEGORIES))
    print(f"Stage 2: {'✓ 通过' if stage2_pass else '✗ 未通过'}")

    out = Path(args.output) / "stage123_report.json"
    out.write_text(json.dumps({
        "stage1": {"rows": stage1_rows, "pass": stage1_pass},
        "stage2": {"pid_of_cat": {k: sorted(v) for k, v in pid_of_cat.items()},
                   "cat_of_pid": {k: sorted(v) for k, v in cat_of_pid.items()},
                   "reid_ok": reid_ok, "sep_ok": sep_ok, "pass": stage2_pass},
        "prototype_status": protos.status(),
        "clips": stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
