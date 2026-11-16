"""C5：事件级评估（告警系统语义）——以人工终审的事件区间 GT 为基准。

指标定义（容差 ±3s）：
  - 事件召回（检测口径）：GT 事件区间内出现 ≥3 个处理帧的同类标签 → 命中；
  - 事件召回（告警口径）：GT 事件区间内触发过 ≥1 次同类告警（alert.triggered）→ 命中；
  - 误报：告警落在所有同类 GT 事件区间（±3s）之外 → 计 1 次假告警；汇总假告警/小时。

用法：
  python tools/eval_event_level.py [--mode live|offline] [--tolerance 3]
  live（默认）：通过 API 跑真实任务（部署路径），采集逐帧标签落盘后算指标；
  offline：本地构建引擎离线重放（供对照，帧数调度语义与线上有差异，仅参考）。
输出：
  results/eval/event_level/event_level_report.json + 控制台摘要
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.inference_engine import InferenceEngine  # noqa: E402
from src.utils.config_loader import load_yaml  # noqa: E402
from src.utils.preflight import resolve_profile  # noqa: E402

GT_DIR = BASE_DIR / "results" / "auto_annotation"
VIDEO_ROOT = Path("/home/user/VSCodeProjects/VdAnalyst/test")
MIN_DET_FRAMES = 3  # 检测口径：窗口内同类标签至少出现的处理帧数


def run_system_on_video(engine: InferenceEngine, video_path: Path, stride: int) -> tuple[list, list, float, float]:
    """离线跑系统，返回 (逐帧标签[(sec,label)], 告警[(sec,label,conf)], fps, duration_s)。"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    engine.set_source_fps(fps)
    engine.begin_task(f"eventeval_{video_path.stem}")

    frame_labels: list[tuple[float, str]] = []
    alerts: list[tuple[float, str, float]] = []
    frame_id = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % stride != 0 and frame_id != 1:
            continue
        result = engine.process(frame, frame_id)
        sec = frame_id / fps
        for det in result.detections:
            frame_labels.append((sec, det.behavior.type))
        # 告警双通道：alert_system（H类，type=h01_detected）+ NL规则命中（M/S类为主）
        if result.alert.triggered:
            label = str(result.alert.type).replace("_detected", "").upper()
            alerts.append((sec, label, 0.0))
        for ra in (result.rule_alerts or []):
            if isinstance(ra, dict) and ra.get("matched_class"):
                alerts.append((int(ra.get("frame_id", frame_id)) / fps,
                               str(ra["matched_class"]), 0.0))
        if frame_id % 400 == 0:
            rate = frame_id / max(time.time() - t0, 1e-6)
            eta = (total - frame_id) / max(rate, 1e-6) / 60
            print(f"    {frame_id}/{total} ({rate:.1f}fps) ETA {eta:.0f}min", flush=True)
    cap.release()
    return frame_labels, alerts, fps, total / fps


API_BASE = "http://localhost:8000"


def run_live_task(video_path: Path, duration_s: float) -> str:
    """通过 API 跑真实任务（eval_dump 落盘），返回 task_id。"""
    import requests
    resp = requests.post(f"{API_BASE}/api/v1/analyze/stream", json={
        "stream_url": str(video_path), "duration": int(duration_s) + 15, "eval_dump": True,
    }, timeout=30)
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    deadline = time.time() + duration_s + 180
    while time.time() < deadline:
        st = requests.get(f"{API_BASE}/api/v1/tasks/{task_id}", timeout=15).json()
        if st.get("status") in ("finished", "failed", "stopped"):
            if st.get("status") != "finished":
                print(f"  [warn] 任务状态 {st.get('status')}: {st.get('error', '')}")
            return task_id
        time.sleep(10)
    raise TimeoutError(f"任务 {task_id} 超时未完成")


def load_dump(task_id: str, fps: float) -> tuple[list, list]:
    """读取落盘的逐帧标签与告警。"""
    dump_path = BASE_DIR / "results" / "eval_dumps" / f"{task_id}.jsonl"
    frame_labels: list[tuple[float, str]] = []
    alerts: list[tuple[float, str, float]] = []
    for line in dump_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        sec = float(rec.get("frame_id", 0)) / fps
        for _, label in rec.get("labels", []):
            if label:
                frame_labels.append((sec, str(label)))
        if rec.get("alert_triggered"):
            alerts.append((sec, str(rec.get("alert_type", "")).replace("_detected", "").upper(), 0.0))
        for label in rec.get("rule_alerts", []):
            alerts.append((sec, str(label), 0.0))
    return frame_labels, alerts


def evaluate_video_live(video_path: Path, gt_events: list[dict], tol: float) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = total / fps
    print(f"  [live] 启动任务（{duration_s/60:.1f}分钟视频，实时速度）...", flush=True)
    task_id = run_live_task(video_path, duration_s)
    print(f"  [live] 任务完成: {task_id}", flush=True)
    frame_labels, alerts = load_dump(task_id, fps)
    return compute_metrics(video_path.name, gt_events, frame_labels, alerts, duration_s, tol)


def compute_metrics(video_name: str, gt_events: list[dict], frame_labels: list,
                    alerts: list, duration_s: float, tol: float) -> dict:
    by_label: dict[str, list[float]] = defaultdict(list)
    for sec, label in frame_labels:
        by_label[label].append(sec)

    per_event = []
    hits_det = hits_alert = 0
    for ev in gt_events:
        label, s0, s1 = ev["label"], float(ev["start_sec"]), float(ev["end_sec"])
        det_frames = sum(1 for sec in by_label.get(label, []) if s0 - tol <= sec <= s1 + tol)
        alert_hit = any(lb == label and s0 - tol <= sec <= s1 + tol for sec, lb, _ in alerts)
        hits_det += det_frames >= MIN_DET_FRAMES
        hits_alert += alert_hit
        per_event.append({
            "label": label, "start_sec": s0, "end_sec": s1,
            "det_frames_in_window": det_frames, "hit_det": det_frames >= MIN_DET_FRAMES,
            "hit_alert": bool(alert_hit),
        })

    false_alerts = []
    for sec, label, conf in alerts:
        if not any(ev["label"] == label and float(ev["start_sec"]) - tol <= sec <= float(ev["end_sec"]) + tol
                   for ev in gt_events):
            false_alerts.append({"sec": round(sec, 1), "label": label, "conf": round(conf, 2)})

    n = len(gt_events)
    return {
        "video": video_name,
        "duration_s": round(duration_s, 1),
        "gt_events": n,
        "recall_detection": round(hits_det / max(1, n), 3),
        "recall_alert": round(hits_alert / max(1, n), 3),
        "hits_det": hits_det,
        "hits_alert": hits_alert,
        "total_alerts": len(alerts),
        "false_alerts": len(false_alerts),
        "false_alerts_per_hour": round(len(false_alerts) / max(duration_s / 3600.0, 1e-6), 2),
        "per_event": per_event,
        "false_alert_detail": false_alerts,
    }


def evaluate_video(engine: InferenceEngine, video_path: Path, gt_events: list[dict],
                   stride: int, tol: float) -> dict:
    frame_labels, alerts, fps, duration_s = run_system_on_video(engine, video_path, stride)
    by_label: dict[str, list[float]] = defaultdict(list)
    for sec, label in frame_labels:
        by_label[label].append(sec)

    per_event = []
    hits_det = hits_alert = 0
    for ev in gt_events:
        label, s0, s1 = ev["label"], float(ev["start_sec"]), float(ev["end_sec"])
        det_frames = sum(1 for sec in by_label.get(label, []) if s0 - tol <= sec <= s1 + tol)
        alert_hit = any(lb == label and s0 - tol <= sec <= s1 + tol for sec, lb, _ in alerts)
        hits_det += det_frames >= MIN_DET_FRAMES
        hits_alert += alert_hit
        per_event.append({
            "label": label, "start_sec": s0, "end_sec": s1,
            "det_frames_in_window": det_frames, "hit_det": det_frames >= MIN_DET_FRAMES,
            "hit_alert": bool(alert_hit),
        })

    # 误报：告警 ±tol 内无同类 GT 事件
    false_alerts = []
    for sec, label, conf in alerts:
        if not any(ev["label"] == label and float(ev["start_sec"]) - tol <= sec <= float(ev["end_sec"]) + tol
                   for ev in gt_events):
            false_alerts.append({"sec": round(sec, 1), "label": label, "conf": round(conf, 2)})

    n = len(gt_events)
    return {
        "video": video_path.name,
        "duration_s": round(duration_s, 1),
        "gt_events": n,
        "recall_detection": round(hits_det / max(1, n), 3),
        "recall_alert": round(hits_alert / max(1, n), 3),
        "hits_det": hits_det,
        "hits_alert": hits_alert,
        "total_alerts": len(alerts),
        "false_alerts": len(false_alerts),
        "false_alerts_per_hour": round(len(false_alerts) / max(duration_s / 3600.0, 1e-6), 2),
        "per_event": per_event,
        "false_alert_detail": false_alerts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "offline"], default="live")
    ap.add_argument("--profile", default="realtime")
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--videos", nargs="*", default=None, help="只评指定视频 stem")
    args = ap.parse_args()

    engine = None
    stride = 4
    if args.mode == "offline":
        raw_model_cfg = load_yaml(str(BASE_DIR / "configs" / "model_config.yaml"))
        model_cfg, _ = resolve_profile(raw_model_cfg, args.profile)
        vlm_cfg = load_yaml(str(BASE_DIR / "configs" / "vlm_config.yaml"))
        engine = InferenceEngine(model_cfg=model_cfg, vlm_cfg=vlm_cfg, output_dir=str(BASE_DIR / "results"))
        stride = int(model_cfg.get("inference", {}).get("frame_stride", 4))

    reports = []
    for gt_path in sorted(GT_DIR.glob("*/gt_events_*.json")):
        stem = gt_path.parent.name
        if args.videos and stem not in args.videos:
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        video_path = Path(gt["video"]) if Path(gt["video"]).is_absolute() else VIDEO_ROOT / gt["video"]
        if not video_path.exists():
            print(f"[skip] 视频不存在: {video_path}")
            continue
        print(f"[eval] {stem}: {len(gt['events'])} 条 GT 事件", flush=True)
        if args.mode == "live":
            rep = evaluate_video_live(video_path, gt["events"], args.tolerance)
        else:
            rep = evaluate_video(engine, video_path, gt["events"], stride, args.tolerance)
        reports.append(rep)
        print(f"  召回(检测)={rep['recall_detection']:.1%} 召回(告警)={rep['recall_alert']:.1%} "
              f"误报={rep['false_alerts']}/{rep['total_alerts']} ({rep['false_alerts_per_hour']}/h)", flush=True)
    if engine is not None:
        engine.close()

    # 汇总
    total_events = sum(r["gt_events"] for r in reports)
    total_hits_det = sum(r["hits_det"] for r in reports)
    total_hits_alert = sum(r["hits_alert"] for r in reports)
    total_fa = sum(r["false_alerts"] for r in reports)
    total_alerts = sum(r["total_alerts"] for r in reports)
    total_hours = sum(r["duration_s"] for r in reports) / 3600.0
    summary = {
        "mode": args.mode, "profile": args.profile, "tolerance_s": args.tolerance,
        "videos": len(reports),
        "total_events": total_events,
        "pooled_recall_detection": round(total_hits_det / max(1, total_events), 3),
        "pooled_recall_alert": round(total_hits_alert / max(1, total_events), 3),
        "total_alerts": total_alerts,
        "total_false_alerts": total_fa,
        "false_alerts_per_hour": round(total_fa / max(total_hours, 1e-6), 2),
    }
    out_dir = BASE_DIR / "results" / "eval" / "event_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "event_level_report.json"
    out_path.write_text(json.dumps({"summary": summary, "per_video": reports},
                                   ensure_ascii=False, indent=2))
    print("\n===== 汇总 =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"报告: {out_path}")


if __name__ == "__main__":
    main()
