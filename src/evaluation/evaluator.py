from __future__ import annotations

"""评估核心：系统输出 vs VLM 伪真值的帧级对比评估。

口径说明（伪真值与被评模型同源，存在乐观偏差，需在材料中声明；
326 帧人工抽检保留为独立锚点）:
  - 匹配单元: (frame_id, track_id) 样本，GT 来自 VLM 教师标注（conf ≥ 0.6）
  - 漏报率: GT 为异常但系统判为 Normal 的比例
  - 误报率: 系统判为异常但 GT 为 Normal 的比例
  - 异常检出率(any): GT 为异常且系统判为任意异常（不限类别）的比例
"""

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import cv2

from src.pipeline.inference_engine import InferenceEngine
from src.utils.config_loader import load_json, load_yaml
from src.utils.preflight import resolve_profile

BASE_DIR = Path(__file__).resolve().parents[2]
ABNORMAL_LABELS = {"H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"}


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def run_evaluation(
    video_path: str | Path,
    ground_truth_path: str | Path,
    profile: str | None = None,
    baseline: bool = False,
    use_scene_context: bool = True,
    frame_stride: int | None = None,
    match_tolerance_frames: int | None = None,
    max_frames: int | None = None,
    output_dir: str | Path = "results/eval",
    report_tag: str | None = None,
    progress_cb: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """跑一遍推理流水线并与伪真值对比，返回并落盘评估报告。"""
    video_path = Path(video_path)
    if not video_path.is_absolute() and not video_path.exists():
        candidate = BASE_DIR.parent / video_path
        if candidate.exists():
            video_path = candidate
    video_path = video_path.resolve()
    ground_truth_path = Path(ground_truth_path)

    gt_doc = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    gt_samples: list[dict] = gt_doc.get("samples", [])
    if not gt_samples:
        raise ValueError(f"ground truth 无样本: {ground_truth_path}")

    # ── 配置组装 ─────────────────────────────────────────────
    raw_model_cfg = load_yaml(str(BASE_DIR / "configs" / "model_config.yaml"))
    behavior_rules = load_json(str(BASE_DIR / "configs" / "behavior_rules.json"))
    raw_model_cfg["behavior_heuristics"] = dict(behavior_rules.get("heuristics", {}))
    raw_model_cfg["alert_policy"] = dict(behavior_rules.get("alert_policy", {}))
    profile = profile or "realtime"
    model_cfg, resolved_paths = resolve_profile(raw_model_cfg, profile)
    vlm_cfg = load_yaml(str(BASE_DIR / "configs" / "vlm_config.yaml"))

    if baseline:
        # 对照组：禁用 PoseClassifier 主路径，强制 SlowFast 关键词映射
        model_cfg.setdefault("action", {})["pose_classifier_onnx_path"] = ""
    if not use_scene_context:
        vlm_cfg = dict(vlm_cfg)
        vlm_cfg["scene_narrator"] = dict(vlm_cfg.get("scene_narrator", {}), enabled=False)

    engine = InferenceEngine(model_cfg=model_cfg, vlm_cfg=vlm_cfg, output_dir=str(BASE_DIR / "results"))
    engine.review_context_enabled = bool(use_scene_context)
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        engine.set_source_fps(source_fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        stride = int(frame_stride or max(1, int(model_cfg.get("inference", {}).get("frame_stride", 2))))
        tolerance = int(match_tolerance_frames if match_tolerance_frames is not None else stride * 2)
        last_gt_frame = max(s["frame_id"] for s in gt_samples)
        limit = last_gt_frame + stride * 4
        if max_frames:
            limit = min(limit, int(max_frames))
        stop_frame = min(total_frames, limit)

        predictions: dict[int, list[dict[str, Any]]] = {}  # frame_id -> [{track_id, bbox, label, conf}]
        latencies: list[float] = []

        fid = 0
        while fid < stop_frame:
            ok, frame = cap.read()
            if not ok:
                break
            fid += 1
            if fid % stride != 0 and fid != 1:
                continue
            result = engine.process(frame, fid)
            record = result.model_dump()
            latencies.append(float(record.get("metrics", {}).get("pipeline_latency_ms", 0)))
            frame_pred: list[dict[str, Any]] = []
            for det in record.get("detections", []):
                frame_pred.append({
                    "track_id": str(det["track_id"]),
                    "bbox": [float(v) for v in det.get("bbox", [])],
                    "label": str(det["behavior"]["type"]),
                    "confidence": float(det["behavior"]["confidence"]),
                })
            predictions[fid] = frame_pred
            if progress_cb is not None and fid % (stride * 100) < stride:
                progress_cb({
                    "stage": "processing",
                    "frame_id": fid,
                    "stop_frame": stop_frame,
                    "progress": round(fid / max(1, stop_frame), 4),
                })
        cap.release()

        # ── GT 与系统预测匹配（bbox IoU，track ID 在两套 tracker 间不可比） ──
        matched_frames = sorted(predictions.keys())
        records: list[dict] = []
        match_counter: Counter = Counter()

        def _bbox_iou(a: list[float], b: list[float]) -> float:
            if len(a) != 4 or len(b) != 4:
                return 0.0
            ax1, ay1, aw, ah = a
            bx1, by1, bw, bh = b
            ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
            bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
            return inter / union if union > 0 else 0.0

        for sample in gt_samples:
            gt_frame = int(sample["frame_id"])
            gt_bbox = [float(v) for v in sample.get("bbox", [])]
            gt_label = str(sample["label"])

            best_iou = 0.0
            best = None
            for pf in matched_frames:
                if abs(pf - gt_frame) > tolerance:
                    continue
                for det in predictions[pf]:
                    iou = _bbox_iou(det["bbox"], gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best = (pf, det)
            if best is None or best_iou < 0.4:
                # 容差内无 bbox 匹配 → 视为系统输出 Normal(0)
                sys_label, sys_conf, match_type = "Normal", 0.0, "missing"
            else:
                pf, det = best
                sys_label, sys_conf = det["label"], det["confidence"]
                match_type = "exact" if pf == gt_frame else "nearest"
            match_counter[match_type] += 1
            records.append({
                "frame_id": gt_frame,
                "track_id": str(sample.get("track_id", "")),
                "gt_label": gt_label,
                "pred_label": sys_label,
                "pred_confidence": sys_conf,
                "matched_frame": best[0] if best else None,
                "match_iou": round(best_iou, 3) if best else 0.0,
                "match_type": match_type,
            })

        # ── 指标计算 ─────────────────────────────────────────
        total = len(records)
        correct = sum(1 for r in records if r["gt_label"] == r["pred_label"])
        gt_abnormal = [r for r in records if r["gt_label"] in ABNORMAL_LABELS]
        gt_normal = [r for r in records if r["gt_label"] == "Normal"]
        sys_abnormal = [r for r in records if r["pred_label"] in ABNORMAL_LABELS]

        miss = sum(1 for r in gt_abnormal if r["pred_label"] == "Normal")
        false_alarm = sum(1 for r in sys_abnormal if r["gt_label"] == "Normal")
        any_hit = sum(1 for r in gt_abnormal if r["pred_label"] in ABNORMAL_LABELS)

        per_class: dict[str, dict] = {}
        for label in ["Normal", *ABNORMAL_LABELS]:
            tp = sum(1 for r in records if r["gt_label"] == label and r["pred_label"] == label)
            fp = sum(1 for r in records if r["gt_label"] != label and r["pred_label"] == label)
            fn = sum(1 for r in records if r["gt_label"] == label and r["pred_label"] != label)
            p, rc, f1 = _prf(tp, fp, fn)
            support = tp + fn
            if support > 0 or fp > 0:
                per_class[label] = {"precision": p, "recall": rc, "f1": f1, "support": support}

        f1_values = [m["f1"] for m in per_class.values() if m["support"] > 0]
        macro_f1 = round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0

        confusion = Counter(
            (r["gt_label"], r["pred_label"]) for r in records if r["gt_label"] != r["pred_label"]
        )
        lat_sorted = sorted(latencies)
        p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0
        p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if lat_sorted else 0

        report = {
            "video": video_path.name,
            "ground_truth": str(ground_truth_path),
            "gt_source": gt_doc.get("source", "vlm_teacher"),
            "config": {
                "profile": profile,
                "profile_model_paths": resolved_paths,
                "baseline": baseline,
                "use_scene_context": use_scene_context,
                "frame_stride": stride,
                "match_tolerance_frames": tolerance,
                "match_method": "bbox_iou_0.4",
                "pose_classifier_loaded": engine.action_recognizer.pose_classifier_loaded,
            },
            "counts": {
                "total": total,
                "matched": match_counter["exact"] + match_counter["nearest"],
                "missing": match_counter["missing"],
                "gt_abnormal": len(gt_abnormal),
                "gt_normal": len(gt_normal),
            },
            "metrics": {
                "accuracy": round(correct / max(1, total), 4),
                "macro_f1": macro_f1,
                "abnormal_recall_any": round(any_hit / max(1, len(gt_abnormal)), 4),
                "false_negative_rate": round(miss / max(1, len(gt_abnormal)), 4),
                "false_positive_rate": round(false_alarm / max(1, len(sys_abnormal)), 4) if sys_abnormal else 0.0,
            },
            "per_class": per_class,
            "top_confusions": [
                {"gt": gt_label, "pred": pred_label, "count": count}
                for (gt_label, pred_label), count in confusion.most_common(10)
            ],
            "latency_ms": {
                "mean": round(sum(latencies) / max(1, len(latencies)), 1),
                "p50": round(p50, 1),
                "p95": round(p95, 1),
                "processed_frames": len(latencies),
            },
            "records": records,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = BASE_DIR / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = report_tag or ("baseline" if baseline else ("no_context" if not use_scene_context else "main"))
        report_path = out_dir / f"eval_report_{video_path.stem}_{tag}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_file"] = str(report_path)
        progress_cb and progress_cb({"stage": "done", "report_file": str(report_path)})
        return report
    finally:
        engine.close()
