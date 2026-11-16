from __future__ import annotations

"""双口径评估（WEAKNESS_AUDIT 复查#一-2）：同一批系统预测分别对
人工金标（gt.json）与 VLM 伪真值（manifest）匹配，输出两套指标对照。

消除"教师=考官"循环的关键一步：伪真值口径看趋势（与历史评估可比），
人工金标口径看真实水平。两口径差异 = 标注质量差（教师系统性漏标/误标），
不是泛化差——corner_1 同时出现在伪真值训练集内（in-domain，须披露）。

用法:
  python tools/eval_dual_caliber.py --video ../data/whole_process_corner_1.mp4 \
      --gt results/frame_gt/whole_process_corner_1/gt.json --frames 1200
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.inference_engine import InferenceEngine

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"]


def iou(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match(preds_by_frame: dict[int, list[dict]], gt_entries: list[dict],
          frame_tol: int = 0) -> tuple[Counter, Counter, Counter, int]:
    """GT 条目 → 同帧(±tol) IoU≥0.4 的预测，贪心取最优。

    不依赖 track_id 字符串——两套运行（金标包生成 vs 本次评估）的 tracker ID
    分配时序不同，ID 不可比（TECHNICAL_REPORT §9.1 同款结论）。
    返回 (tp, fp, fn, matched_total)。
    """
    tp, fp, fn = Counter(), Counter(), Counter()
    matched = 0
    used: set[tuple[int, int]] = set()
    for g in gt_entries:
        gf = int(g["frame_id"])
        best, best_iou, best_key = None, 0.0, None
        for df in range(-frame_tol, frame_tol + 1):
            for p_idx, p in enumerate(preds_by_frame.get(gf + df, [])):
                if (gf + df, p_idx) in used:
                    continue
                v = iou(g["bbox"], p["bbox"])
                if v >= 0.4 and v > best_iou:
                    best, best_iou, best_key = p, v, (gf + df, p_idx)
        if best is not None:
            used.add(best_key)
            matched += 1
            if best["label"] == g["label"]:
                tp[g["label"]] += 1
            else:
                fn[g["label"]] += 1
                fp[best["label"]] += 1
        else:
            fn[g["label"]] += 1
    return tp, fp, fn, matched


def prf_table(gt_entries: list[dict], tp: Counter, fp: Counter, fn: Counter) -> dict:
    per = {}
    for c in CLASSES:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else None
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else None
        f = 2 * p * r / (p + r) if p and r else None
        per[c] = {"support": tp[c] + fn[c], "precision": round(p, 3) if p is not None else None,
                  "recall": round(r, 3) if r is not None else None, "f1": round(f, 3) if f is not None else None}
    correct = sum(tp.values())
    macro_f = [v["f1"] for v in per.values() if v["f1"] is not None]
    return {"per_class": per, "accuracy": round(correct / max(1, len(gt_entries)), 3),
            "macro_f1": round(sum(macro_f) / len(macro_f), 3) if macro_f else None,
            "support_total": len(gt_entries)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="../data/whole_process_corner_1.mp4")
    ap.add_argument("--gt", default="results/frame_gt/whole_process_corner_1/gt.json")
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--pseudo-tol", type=int, default=8, help="伪真值帧容差（与历史评估口径一致）")
    ap.add_argument("--gt-frame-tol", type=int, default=8, help="人工金标帧容差（GT每30处理帧采样，人在周期内移动）")
    ap.add_argument("--out", default="results/eval/dual_caliber_report.json")
    args = ap.parse_args()

    # ── 1. 离线跑引擎收预测（VLM 关：纯主路径，与历史伪真值评估口径一致） ──
    model_cfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    vlm_cfg = yaml.safe_load((ROOT / "configs" / "vlm_config.yaml").read_text(encoding="utf-8"))
    vlm_cfg["use_vllm"] = False  # 离线评估不依赖 VLM（灰区复核退回 SlowFast，与 --baseline 口径一致）
    engine = InferenceEngine(model_cfg, vlm_cfg, output_dir="results/eval_dual_tmp")
    engine.begin_task("dual_caliber")
    cap = cv2.VideoCapture(args.video)
    preds_by_frame: dict[int, list[dict]] = defaultdict(list)
    fid = 0
    while fid < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        res = engine.process(frame, fid)
        for d in res.detections:
            preds_by_frame[fid].append({
                "track_id": d.track_id, "bbox": list(d.bbox),
                "label": d.behavior.type, "confidence": d.behavior.confidence,
            })
        fid += 1
    cap.release()
    n_pred = sum(len(v) for v in preds_by_frame.values())
    print(f"引擎离线推理: {fid} 帧, {n_pred} 条 (frame,track) 预测")

    # ── 2. 人工金标口径（帧精确匹配：GT 即处理帧采样） ──
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    tp, fp, fn, matched = match(preds_by_frame, gt, frame_tol=args.gt_frame_tol)
    human = prf_table(gt, tp, fp, fn)
    human["matched"] = matched
    human["caliber"] = "human_gt"
    print(f"\n人工金标口径: accuracy={human['accuracy']} Macro-F1={human['macro_f1']} "
          f"匹配率={matched}/{len(gt)}")
    for c, v in human["per_class"].items():
        if v["support"]:
            print(f"  {c}: support={v['support']} P={v['precision']} R={v['recall']} F1={v['f1']}")

    # ── 3. 伪真值口径（参照）：历史评估指标引用 + corner_1 教师标签分布对照 ──
    # 伪真值实例匹配不可行：manifest 无 bbox，且两套运行 track_id 不可比——
    # 故伪真值侧引用历史评估报告（corner_3/entrance_3，教师 bbox IoU 匹配口径）的指标，
    # 并给 corner_1 的"教师标签分布 vs 系统预测分布"帧级分布对照（弱参照，无实例对齐）。
    manifest = json.loads((ROOT / "results/training_data/manifest.json").read_text(encoding="utf-8"))
    pseudo_c1 = [s for s in manifest if "corner_1" in s.get("video", "")]
    teacher_dist = Counter(s["refined_label"] for s in pseudo_c1)
    sys_dist = Counter(p["label"] for plist in preds_by_frame.values() for p in plist)
    pseudo_rep = {
        "caliber": "pseudo_gt(teacher) — 参照口径：历史报告指标引用 + corner_1 标签分布对照（无实例对齐）",
        "historical_reports": {},
        "corner_1_teacher_dist": dict(sorted(teacher_dist.items(), key=lambda kv: -kv[1])),
        "corner_1_system_dist": dict(sorted(sys_dist.items(), key=lambda kv: -kv[1])),
    }
    for rp in sorted((ROOT / "results/eval").glob("eval_report_*_main.json")):
        try:
            d = json.loads(rp.read_text(encoding="utf-8"))
            pseudo_rep["historical_reports"][rp.stem] = {
                "video": d.get("video"), "gt_source": d.get("gt_source"),
                "accuracy": d.get("metrics", {}).get("accuracy"),
                "macro_f1": d.get("metrics", {}).get("macro_f1"),
            }
        except Exception:
            continue
    print(f"\n伪真值口径（参照）: 历史报告 {list(pseudo_rep['historical_reports'].keys())}")
    print(f"  corner_1 教师分布: {pseudo_rep['corner_1_teacher_dist']}")
    print(f"  corner_1 系统分布: {pseudo_rep['corner_1_system_dist']}")

    report = {
        "video": Path(args.video).name,
        "disclosure": "corner_1 同时在伪真值训练集内（in-domain）；双口径差异反映标注质量差（教师系统性漏标 vs 人工逐帧核看），不是泛化差",
        "n_predictions": n_pred,
        "human_gt": human,
        "pseudo_gt": pseudo_rep,
        "delta_note": "伪真值侧为参照口径（历史报告引用+分布对照），不作逐数差值",
        "human_gt_metrics": {"accuracy": human["accuracy"], "macro_f1": human["macro_f1"]},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n双口径报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
