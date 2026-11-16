from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import queue
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
import yaml as _yaml
from pydantic import BaseModel

from src.api.websocket_server import ConnectionManager
from src.core.scene_index import SceneIndex
from src.pipeline.inference_engine import InferenceEngine
from src.pipeline.video_stream import VideoStream
from src.utils.config_loader import load_json, load_yaml
from src.utils.preflight import resolve_profile, run_preflight
from src.utils.schemas import (
    AnalyzeStreamRequest,
    AnalyzeStreamResponse,
    ChatRequest,
    EvaluateRequest,
    NarrateRequest,
    TrainRequest,
    VLMReviewRequest,
    VLMReviewResponse,
)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "configs"

raw_model_cfg = load_yaml(CONFIG_DIR / "model_config.yaml")
vlm_cfg = load_yaml(CONFIG_DIR / "vlm_config.yaml")
behavior_rules = load_json(CONFIG_DIR / "behavior_rules.json")

if isinstance(behavior_rules, dict) and isinstance(behavior_rules.get("heuristics"), dict):
    raw_model_cfg["behavior_heuristics"] = dict(behavior_rules["heuristics"])
if isinstance(behavior_rules, dict) and isinstance(behavior_rules.get("alert_policy"), dict):
    raw_model_cfg["alert_policy"] = dict(behavior_rules["alert_policy"])


def apply_inference_profile(config: dict, profile_name: str | None = None) -> tuple[dict, str, dict]:
    selected_profile = profile_name or os.getenv("INFERENCE_PROFILE", "realtime")
    merged, resolved_model_paths = resolve_profile(config, selected_profile)
    return merged, selected_profile, resolved_model_paths


model_cfg, active_profile, active_model_paths = apply_inference_profile(raw_model_cfg)


def _resolve_replay_buffer_frames(config: dict) -> int:
    inference_cfg = config.get("inference", {}) if isinstance(config, dict) else {}
    return int(max(0, inference_cfg.get("replay_buffer_frames", 24)))


def _estimate_task_replay_history_size(config: dict, duration_seconds: int) -> int:
    inference_cfg = config.get("inference", {}) if isinstance(config, dict) else {}
    replay_from_start = bool(inference_cfg.get("replay_from_task_start", True))
    base_buffer = int(max(0, inference_cfg.get("replay_buffer_frames", 24)))
    if not replay_from_start:
        return base_buffer

    max_fps = float(max(0.0, inference_cfg.get("max_fps", 0)))
    push_stride = int(max(1, inference_cfg.get("push_stride", 1)))
    if duration_seconds <= 0 or max_fps <= 0:
        return max(base_buffer, 300)

    estimated = int(duration_seconds * max_fps / push_stride) + 16
    return int(max(base_buffer, min(5000, estimated)))

app = FastAPI(title="Campus Security Agent API", version="0.1.0")

# 盒上适配: 看板(:8501)与 API(:8000)不同源，浏览器 fetch 事件/指标需 CORS。
# 本地演示场景放开来源（不带凭据），避免"看板能开但视频/遥测连不上"。
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root_index(request: Request):
    """根路径导航页：API 本身无业务根路由，避免访客只看到 {"detail":"Not Found"}。"""
    host = str(request.headers.get("host") or "127.0.0.1:8000").split(":", 1)[0]
    base = str(request.base_url).rstrip("/")
    ui = f"http://{host}:8501"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>慧眼智警·边缘认知盒 — 入口</title>
<style>
 body{{margin:0;padding:40px 24px;font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
      background:#0f1720;color:#e6edf3}}
 h1{{font-size:22px;margin:0 0 6px}} p.sub{{color:#8b949e;margin:0 0 28px}}
 .grid{{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));max-width:900px}}
 a.card{{display:block;padding:18px 20px;border-radius:10px;background:#161b22;border:1px solid #30363d;
        text-decoration:none;color:#e6edf3;transition:.15s}}
 a.card:hover{{border-color:#2f81f7;transform:translateY(-2px)}}
 a.card b{{display:block;font-size:16px;margin-bottom:4px}} a.card span{{color:#8b949e;font-size:13px}}
 code{{background:#21262d;padding:2px 6px;border-radius:5px;font-size:13px}}
 .tip{{margin-top:28px;color:#8b949e;font-size:13px;max-width:900px}}
</style></head><body>
<h1>慧眼智警·边缘认知盒</h1>
<p class="sub">Campus Security Agent · 边缘 API 入口</p>
<div class="grid">
  <a class="card" href="{ui}" target="_blank"><b>边缘操作台（全功能）→</b><span>{ui}（视频/检测框/认知视图/问答）</span></a>
  <a class="card" href="{base}/docs" target="_blank"><b>接口文档（Swagger）→</b><span>{base}/docs</span></a>
  <a class="card" href="{base}/health" target="_blank"><b>健康检查 →</b><span>{base}/health</span></a>
  <a class="card" href="{base}/api/v1/metrics" target="_blank"><b>底牌遥测 →</b><span>{base}/api/v1/metrics</span></a>
  <a class="card" href="{base}/remote" target="_blank"><b>远程 MQTT 监测端（任意设备）→</b><span>{base}/remote（浏览器直连 :9001）</span></a>
</div>
<p class="tip">分工：<b>边缘操作台</b>跑在盒上，提供视频与五时刻交互；<b>远程 MQTT 监测端</b>是纯静态页，
任意浏览器直连盒上 MQTT（WebSocket），证明端云联动与协议运用。</p>
<p class="tip">看板需在侧栏把 <code>API Base</code> 指到 <code>{base}</code>；
起流示例：<code>POST /api/v1/analyze/stream</code>，MQTT topic：<code>csa/edge1/status|event</code>。</p>
</body></html>"""


@app.get("/remote", response_class=HTMLResponse, include_in_schema=False)
def remote_dashboard():
    """远程 MQTT 看板：静态页，浏览器直连盒上 mosquitto 的 WebSocket(9001)。"""
    path = BASE_DIR / "demo" / "remote_dashboard.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="remote_dashboard.html not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


manager = ConnectionManager(history_size=_resolve_replay_buffer_frames(model_cfg))
engine = InferenceEngine(model_cfg=model_cfg, vlm_cfg=vlm_cfg, output_dir=str(BASE_DIR / "results"))
TASKS: dict[str, dict] = {}
EVAL_JOBS: dict[str, dict] = {}
TRAIN_JOBS: dict[str, dict] = {"_active": {"running": False, "train_id": ""}}
scene_index = SceneIndex(vlm_cfg, results_dir=str(BASE_DIR / "results"))


def _backend_health(engine) -> dict:
    """切档后校验 OM 是否真的加载成功（CANN 不支持同进程二次 acl.init）。"""
    detector = getattr(engine, "detector", None)
    pose = getattr(engine, "pose_estimator", None)
    detector_degraded = bool(getattr(detector, "om_degraded_to_onnx", False))
    pose_degraded = bool(getattr(pose, "om_degraded_to_onnx", False))
    return {
        "degraded": detector_degraded or pose_degraded,
        "detector_degraded": detector_degraded,
        "detector_model": getattr(detector, "onnx_path_resolved", ""),
        "pose_degraded": pose_degraded,
        "pose_model": getattr(pose, "onnx_path_resolved", ""),
    }


def _profile_needs_restart(profile_name: str) -> bool:
    """目标档检测模型为 .om 时需换新进程：CANN 不支持同进程二次 acl.init，
    进程内重建 OM 必回退 CPU ONNX；exec 自替换后新进程冷启动可正常加载 OM。"""
    _, _, paths = apply_inference_profile(raw_model_cfg, profile_name=profile_name)
    return str(paths.get("detector_onnx", "")).lower().endswith(".om")


async def _restart_into_profile(profile_name: str) -> None:
    """响应发出后：释放 NPU → exec 自替换为带目标档环境变量的新进程（PID 不变）。"""
    await asyncio.sleep(1.0)  # 留时间把响应 flush 到客户端
    try:
        engine.close()  # OM unload→finalize，避免设备句柄带入新进程
    except Exception as exc:
        print(f"[profile-restart] NPU 释放异常（继续重启）: {exc}", flush=True)
    env = dict(os.environ)
    env["INFERENCE_PROFILE"] = profile_name
    print(f"[profile-restart] exec 自重启进入 {profile_name} 档: {sys.executable} {sys.argv}", flush=True)
    try:
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    except OSError as exc:
        print(f"[profile-restart] exec 失败，引擎需手动停再起: {exc}", flush=True)


def reconfigure_profile(profile_name: str) -> dict:
    global model_cfg
    global active_profile
    global active_model_paths
    global engine

    available_profiles = raw_model_cfg.get("profiles", {})
    if profile_name not in available_profiles:
        raise HTTPException(status_code=400, detail=f"invalid profile: {profile_name}")

    model_cfg, active_profile, active_model_paths = apply_inference_profile(raw_model_cfg, profile_name=profile_name)
    manager.set_history_size(_resolve_replay_buffer_frames(model_cfg))
    old_engine = engine
    old_engine.close()  # 先释放 NPU（OM unload→finalize），再重建；否则新 OM 加载冲突回退 CPU ONNX
    engine = InferenceEngine(model_cfg=model_cfg, vlm_cfg=vlm_cfg, output_dir=str(BASE_DIR / "results"))
    health = _backend_health(engine)
    result = {
        "profile": active_profile,
        "runtime": model_cfg.get("runtime", {}),
        "inference": model_cfg.get("inference", {}),
        "models": active_model_paths,
        "backend": health,
    }
    if health["degraded"]:
        result["status"] = "degraded"
        result["degraded"] = True
        result["warning"] = (
            f"档位已切到 {active_profile}，但 OM 在进程内重建失败并回退 CPU ONNX"
            f"（CANN 不支持同进程二次 acl.init）。请停再起引擎才真正生效："
            f"./csa.sh stop engine && INFERENCE_PROFILE={active_profile} ./csa.sh start engine"
        )
    return result


def encode_frame_to_base64(frame, jpeg_quality: int = 75) -> str:
    quality = int(max(30, min(95, jpeg_quality)))
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def annotate_frame(frame, payload: dict, show_pose_overlay: bool = True) -> any:
    detections = payload.get("detections", [])
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        x, y, w, h = [int(value) for value in bbox]
        behavior = det.get("behavior", {})
        behavior_type = behavior.get("type", "Normal")
        confidence = float(behavior.get("confidence", 0.0))
        role = behavior.get("role", "unknown")
        risk = behavior.get("risk_level", "LOW")
        track_id = det.get("track_id", "P000")

        if risk == "HIGH":
            color = (0, 0, 255)
        elif risk == "MEDIUM":
            color = (0, 165, 255)
        else:
            color = (0, 200, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = f"{track_id} {behavior_type} {confidence:.2f} {role}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_bg_top = max(0, y - text_h - 8)
        cv2.rectangle(frame, (x, text_bg_top), (x + text_w + 6, y), color, -1)
        cv2.putText(
            frame,
            label,
            (x + 3, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if show_pose_overlay:
            for keypoint in det.get("pose_keypoints", []):
                if len(keypoint) < 2:
                    continue
                kp_x = int(keypoint[0])
                kp_y = int(keypoint[1])
                conf = float(keypoint[2]) if len(keypoint) > 2 else 1.0
                if conf < 0.2:
                    continue
                cv2.circle(frame, (kp_x, kp_y), 2, (255, 255, 0), -1)

    alert = payload.get("alert", {})
    if alert.get("triggered"):
        alert_text = f"ALERT: {alert.get('type', 'unknown')}"
        cv2.rectangle(frame, (10, 10), (420, 48), (0, 0, 255), -1)
        cv2.putText(frame, alert_text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def read_alert_logs(results_dir: Path, limit: int = 50) -> list[dict]:
    files = sorted(results_dir.glob("event_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    alerts: list[dict] = []
    for file in files[:limit]:
        try:
            alerts.append(json.loads(file.read_text(encoding="utf-8")))
        except Exception:
            continue
    return alerts


def run_render_task(task_id: str, stream_url: str) -> None:
    inference_cfg = model_cfg.get("inference", {})
    show_pose_overlay = bool(inference_cfg.get("show_pose_overlay", True))
    output_dir = BASE_DIR / "results" / "rendered"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task_id}.mp4"

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = "cannot_open_stream"
        return

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 1.0:
        source_fps = 25.0
    engine.begin_task(task_id)
    engine.set_source_fps(source_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = "invalid_video_size"
        cap.release()
        return

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
    if not writer.isOpened():
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = "cannot_open_video_writer"
        cap.release()
        return

    TASKS[task_id]["status"] = "running"
    TASKS[task_id]["mode"] = "render"
    TASKS[task_id]["render"] = {
        "frames_total": total_frames,
        "frames_processed": 0,
        "progress": 0.0,
        "fps": round(source_fps, 3),
        "output_video_path": str(output_path),
    }

    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1
            result = engine.process(frame, frame_id)
            payload = result.model_dump()
            vis_frame = annotate_frame(frame.copy(), payload, show_pose_overlay=show_pose_overlay)
            writer.write(vis_frame)

            progress = 0.0
            if total_frames > 0:
                progress = min(1.0, frame_id / total_frames)

            TASKS[task_id]["last_frame"] = frame_id
            TASKS[task_id]["last_alert"] = payload.get("alert", {})
            TASKS[task_id]["render"] = {
                "frames_total": total_frames,
                "frames_processed": frame_id,
                "progress": round(progress, 4),
                "fps": round(source_fps, 3),
                "output_video_path": str(output_path),
            }

        TASKS[task_id]["status"] = "finished"
    except Exception as exc:
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = f"render_failed: {exc}"
    finally:
        cap.release()
        writer.release()


async def run_task(task_id: str, stream_url: str, duration: int):
    inference_cfg = model_cfg.get("inference", {})
    # T2.2: 解码三档（cv/ffmpeg/auto）——g2 探针实测选档后写入档位；失败回退默认 cv
    try:
        stream = VideoStream(
            stream_url,
            source_backend=str(inference_cfg.get("source_backend", "auto")),
            resolution=inference_cfg.get("source_resolution") or None,
        )
    except Exception:
        stream = VideoStream(stream_url)
    frame_stride = int(max(1, inference_cfg.get("frame_stride", 1)))
    push_stride = int(max(1, inference_cfg.get("push_stride", 1)))
    send_frame_image_b64 = bool(inference_cfg.get("send_frame_image_b64", False))
    max_fps = float(max(0.0, inference_cfg.get("max_fps", 0)))
    startup_no_drop_seconds = float(max(0.0, min(5.0, inference_cfg.get("startup_no_drop_seconds", 1.5))))
    jpeg_quality = int(inference_cfg.get("jpeg_quality", 75))
    ws_sleep_ms = int(max(0, inference_cfg.get("ws_sleep_ms", 1)))
    source_resize = bool(inference_cfg.get("source_resize", False))
    send_pose_keypoints = bool(inference_cfg.get("send_pose_keypoints", True))
    show_pose_overlay = bool(inference_cfg.get("show_pose_overlay", True))
    target_size = inference_cfg.get("input_size", [960, 540])
    target_width = int(target_size[0]) if isinstance(target_size, list) and len(target_size) > 0 else 960
    target_height = int(target_size[1]) if isinstance(target_size, list) and len(target_size) > 1 else 540

    source_fps = float(stream.fps if stream.fps and stream.fps > 0 else 25.0)
    task_started_at = time.time()
    TASKS[task_id]["source_fps"] = round(source_fps, 3)
    engine.begin_task(task_id)
    engine.set_source_fps(source_fps)
    max_frames = int(duration * source_fps) if duration > 0 else None
    min_interval = (1.0 / max_fps) if max_fps > 0 else 0.0
    is_local_file = Path(stream_url).exists()
    read_min_interval = (1.0 / source_fps) if is_local_file and source_fps > 0 else 0.0
    stop_event = threading.Event()
    task_start_ts = time.perf_counter()

    decode_queue: queue.Queue = queue.Queue(maxsize=2)
    result_queue: queue.Queue = queue.Queue(maxsize=2)
    packet_queue: queue.Queue = queue.Queue(maxsize=2)

    stats_lock = threading.Lock()
    read_counter = 0
    infer_counter = 0
    engine_process_counter = 0
    replay_counter = 0
    encoded_counter = 0
    pushed_count = 0
    dropped_frames = 0
    dropped_results = 0
    dropped_packets = 0
    reconnect_attempts = 0
    last_processed_frame_id = 0
    action_runs_total = 0
    action_skip_warmup_total = 0
    action_skip_track_age_total = 0
    action_skip_gate_total = 0
    action_skip_cached_total = 0
    action_skip_default_total = 0

    stage_a_ms_total = 0.0
    stage_a_ms_max = 0.0
    stage_a_count = 0

    stage_b_ms_total = 0.0
    stage_b_ms_max = 0.0
    stage_b_count = 0

    stage_c_ms_total = 0.0
    stage_c_ms_max = 0.0
    stage_c_count = 0

    stage_d_ms_total = 0.0
    stage_d_ms_max = 0.0
    stage_d_count = 0
    stage_d_sleep_ms_total = 0.0

    thread_error = ""

    def enqueue_latest(q: queue.Queue, item: object, drop_counter_name: str) -> None:
        nonlocal dropped_frames
        nonlocal dropped_results
        nonlocal dropped_packets

        try:
            q.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            q.get_nowait()
        except Exception:
            pass

        if drop_counter_name == "frames":
            dropped_frames += 1
        elif drop_counter_name == "results":
            dropped_results += 1
        elif drop_counter_name == "packets":
            dropped_packets += 1

        try:
            q.put_nowait(item)
        except Exception:
            pass

    # B6: 管道模式开关（send_frame_image_b64=True 时帧与结果必须同步，回退串行）
    pipeline_mode = bool(getattr(engine, "_pipeline_enabled", False)) and not send_frame_image_b64

    # C5: 事件级评估——逐帧标签落盘（eval_dump 任务才开启）
    _eval_fh = None
    if TASKS.get(task_id, {}).get("eval_dump"):
        _dump_dir = BASE_DIR / "results" / "eval_dumps"
        _dump_dir.mkdir(parents=True, exist_ok=True)
        _eval_fh = open(_dump_dir / f"{task_id}.jsonl", "w", encoding="utf-8")

    def build_payload(result) -> dict:
        """FrameResult → WS payload（B6: 同步/管道两路共用）。"""
        payload = result.model_dump()
        if _eval_fh is not None:
            try:
                _eval_fh.write(json.dumps({
                    "frame_id": payload.get("frame_id"),
                    "labels": [[d.get("track_id"), (d.get("behavior") or {}).get("type")]
                               for d in payload.get("detections", [])],
                    "alert_triggered": bool(payload.get("alert", {}).get("triggered")),
                    "alert_type": payload.get("alert", {}).get("type", "none"),
                    "rule_alerts": [ra.get("matched_class") for ra in payload.get("rule_alerts", [])
                                    if isinstance(ra, dict) and ra.get("matched_class")],
                    "sources": payload.get("metrics", {}).get("label_sources", {}),
                }, ensure_ascii=False) + "\n")
                _eval_fh.flush()
            except Exception:
                pass
        action_debug = payload.get("metrics", {}).get("action_debug", {})
        if isinstance(action_debug, dict):
            TASKS[task_id]["action_perf"] = {
                "action_runs": int(action_debug.get("action_runs", 0)),
                "action_skip_warmup": int(action_debug.get("action_skip_warmup", 0)),
                "action_skip_track_age": int(action_debug.get("action_skip_track_age", 0)),
                "action_skip_gate": int(action_debug.get("action_skip_gate", 0)),
                "action_skip_cached": int(action_debug.get("action_skip_cached", 0)),
                "action_skip_default": int(action_debug.get("action_skip_default", 0)),
                "action_stride": int(action_debug.get("action_stride", 0)),
                "max_action_tracks": int(action_debug.get("max_action_tracks", 0)),
                "action_warmup_frames": int(action_debug.get("action_warmup_frames", 0)),
                "action_min_track_age": int(action_debug.get("action_min_track_age", 0)),
            }
        if not send_pose_keypoints:
            for item in payload.get("detections", []):
                item["pose_keypoints"] = []
        payload.setdefault("metrics", {})["streaming"] = {
            "frame_stride": frame_stride,
            "push_stride": push_stride,
            "source_fps": round(source_fps, 3),
            "jpeg_quality": jpeg_quality,
            "ws_sleep_ms": ws_sleep_ms,
            "source_resize": source_resize,
            "send_pose_keypoints": send_pose_keypoints,
            "show_pose_overlay": show_pose_overlay,
            "input_size": [target_width, target_height],
        }
        return payload

    def reader_loop() -> None:
        nonlocal thread_error
        nonlocal read_counter
        nonlocal reconnect_attempts
        nonlocal stage_a_ms_total
        nonlocal stage_a_ms_max
        nonlocal stage_a_count
        last_read_ts = 0.0
        is_rtsp_source = bool(getattr(stream, "is_rtsp", False))

        def should_stop() -> bool:
            return stop_event.is_set() or bool(TASKS.get(task_id, {}).get("stop_requested", False))

        try:
            while (
                not stop_event.is_set()
                and not TASKS.get(task_id, {}).get("stop_requested", False)
                and (max_frames is None or read_counter < max_frames)
            ):
                if read_min_interval > 0:
                    now = time.perf_counter()
                    elapsed = now - last_read_ts
                    if elapsed < read_min_interval:
                        time.sleep(read_min_interval - elapsed)
                    last_read_ts = time.perf_counter()

                stage_a_t0 = time.perf_counter()
                ok, frame, frame_id = stream.read()
                if not ok:
                    # 盒上适配: RTSP 断流无限重试——VideoStream.read() 内建 2s→30s 指数退避，
                    # 持续重开到源恢复；stop_event / stop_requested 时退出。
                    # 本地文件仍按 EOF 结束任务。
                    if is_rtsp_source and not should_stop():
                        reconnect_attempts += 1
                        TASKS[task_id]["reconnect_attempts"] = reconnect_attempts
                        TASKS[task_id]["reconnecting"] = True
                        continue
                    break

                TASKS[task_id]["reconnecting"] = False
                read_counter += 1
                orig_frame = frame  # 问题1：原帧复核通道——降采样前的原始帧
                if source_resize and target_width > 0 and target_height > 0:
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

                while not stop_event.is_set():
                    in_startup_protection = (time.perf_counter() - task_start_ts) < startup_no_drop_seconds
                    if in_startup_protection and decode_queue.full():
                        time.sleep(0.001)
                        continue

                    with stats_lock:
                        enqueue_latest(decode_queue, (orig_frame, frame, frame_id), "frames")
                        stage_a_ms = (time.perf_counter() - stage_a_t0) * 1000.0
                        stage_a_ms_total += stage_a_ms
                        stage_a_ms_max = max(stage_a_ms_max, stage_a_ms)
                        stage_a_count += 1
                    break
        except Exception as exc:
            thread_error = f"reader_failed: {exc}"
        finally:
            stop_event.set()

    def infer_loop() -> None:
        nonlocal thread_error
        nonlocal infer_counter
        nonlocal engine_process_counter
        nonlocal replay_counter
        nonlocal last_processed_frame_id
        nonlocal stage_b_ms_total
        nonlocal stage_b_ms_max
        nonlocal stage_b_count
        last_payload: dict | None = None
        last_infer_ts = 0.0
        # 框逐帧外推：detector_stride 缓存复用与回放都会产生"bbox 与上次完全相同"的事件，
        # 导致框冻结后突跳。历史只在 bbox 真正变化时更新（否则速度被同值观测清零）；
        # 同值事件按最近两次真实位移线性外推（最多 4 个观测步长），且不改动 last_payload。
        track_bbox_cur: dict[str, tuple[int, list]] = {}
        track_bbox_prev: dict[str, tuple[int, list]] = {}

        def remember_real_tracks(payload: dict) -> None:
            fid = payload.get("frame_id")
            if not isinstance(fid, int):
                return
            for det in payload.get("detections") or []:
                tid = det.get("track_id")
                bbox = det.get("bbox")
                if not tid or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                cur = track_bbox_cur.get(tid)
                if cur is None:
                    track_bbox_cur[tid] = (fid, list(bbox))
                elif not all(abs(bbox[j] - cur[1][j]) < 0.5 for j in range(4)):
                    track_bbox_prev[tid] = cur
                    track_bbox_cur[tid] = (fid, list(bbox))

        def advance_payload(payload: dict, frame_id: int) -> dict:
            dets = payload.get("detections")
            if not isinstance(dets, list) or not dets:
                return payload
            new_dets = list(dets)
            changed = False
            for i, det in enumerate(dets):
                tid = det.get("track_id")
                bbox = det.get("bbox")
                if not tid or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                cur = track_bbox_cur.get(tid)
                if cur is None or frame_id <= cur[0]:
                    continue
                if not all(abs(bbox[j] - cur[1][j]) < 0.5 for j in range(4)):
                    continue
                prev = track_bbox_prev.get(tid)
                if prev is not None and cur[0] > prev[0]:
                    # 外推最多 4 个观测步长（管道真实结果滞后 30+ 帧到达，2 步会顶格冻结）
                    k = min(4.0, (frame_id - cur[0]) / (cur[0] - prev[0]))
                    nd = dict(det)
                    nd["bbox"] = [cur[1][j] + (cur[1][j] - prev[1][j]) * k for j in range(4)]
                    new_dets[i] = nd
                    changed = True
            return dict(payload, detections=new_dets) if changed else payload
        try:
            while not stop_event.is_set() or not decode_queue.empty():
                if TASKS.get(task_id, {}).get("stop_requested", False):
                    stop_event.set()
                    break
                try:
                    orig_frame, frame, frame_id = decode_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if min_interval > 0:
                    now = time.perf_counter()
                    elapsed = now - last_infer_ts
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                    last_infer_ts = time.perf_counter()

                stage_b_t0 = time.perf_counter()
                should_infer = frame_id == 1 or (frame_id % frame_stride) == 0
                should_push = frame_id == 1 or (frame_id % push_stride) == 0

                if should_infer and pipeline_mode:
                    # B6: 投递本帧 → 收割已完成结果（结果滞后数帧，payload 带各自 frame_id）
                    engine.submit_frame(frame, frame_id, orig_frame=orig_frame)
                    drained = engine.drain_results()
                    for r in drained:
                        p = build_payload(r)
                        remember_real_tracks(p)
                        last_payload = p
                        engine_process_counter += 1
                        if (r.frame_id % push_stride) == 0 or r.frame_id == 1:
                            enqueue_latest(result_queue, (None, r.frame_id, advance_payload(p, r.frame_id)), "results")
                    if drained:
                        infer_counter += 1
                        last_processed_frame_id = frame_id
                        continue
                    # 结果未就绪：复用最近 payload（同重放语义），无历史则跳过本帧
                    if last_payload is None:
                        continue
                    payload = dict(last_payload)
                    payload["frame_id"] = frame_id
                    payload = advance_payload(payload, frame_id)
                    replay_counter += 1
                elif should_infer:
                    result = engine.process(frame, frame_id, orig_frame=orig_frame)
                    payload = build_payload(result)
                    remember_real_tracks(payload)
                    engine_process_counter += 1
                    last_payload = payload
                    payload = advance_payload(payload, frame_id)
                elif last_payload is not None:
                    payload = dict(last_payload)
                    payload["frame_id"] = frame_id
                    payload["timestamp"] = payload.get("timestamp")
                    payload = advance_payload(payload, frame_id)
                    replay_counter += 1
                else:
                    continue

                infer_counter += 1
                last_processed_frame_id = frame_id

                if not should_push:
                    with stats_lock:
                        stage_b_ms = (time.perf_counter() - stage_b_t0) * 1000.0
                        stage_b_ms_total += stage_b_ms
                        stage_b_ms_max = max(stage_b_ms_max, stage_b_ms)
                        stage_b_count += 1
                    continue

                with stats_lock:
                    enqueue_latest(result_queue, (frame, frame_id, payload), "results")
                    stage_b_ms = (time.perf_counter() - stage_b_t0) * 1000.0
                    stage_b_ms_total += stage_b_ms
                    stage_b_ms_max = max(stage_b_ms_max, stage_b_ms)
                    stage_b_count += 1
        except Exception as exc:
            thread_error = f"inference_failed: {exc}"
        finally:
            stop_event.set()

    def encode_loop() -> None:
        nonlocal thread_error
        nonlocal encoded_counter
        nonlocal stage_c_ms_total
        nonlocal stage_c_ms_max
        nonlocal stage_c_count
        try:
            while not stop_event.is_set() or not result_queue.empty():
                if TASKS.get(task_id, {}).get("stop_requested", False):
                    stop_event.set()
                    break
                try:
                    frame, frame_id, payload = result_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                stage_c_t0 = time.perf_counter()
                if send_frame_image_b64:
                    vis_frame = annotate_frame(frame.copy(), payload, show_pose_overlay=show_pose_overlay)
                    payload["frame_image_b64"] = encode_frame_to_base64(vis_frame, jpeg_quality=jpeg_quality)
                else:
                    payload.pop("frame_image_b64", None)
                encoded_counter += 1

                with stats_lock:
                    enqueue_latest(packet_queue, (frame_id, payload), "packets")
                    stage_c_ms = (time.perf_counter() - stage_c_t0) * 1000.0
                    stage_c_ms_total += stage_c_ms
                    stage_c_ms_max = max(stage_c_ms_max, stage_c_ms)
                    stage_c_count += 1
        except Exception as exc:
            thread_error = f"encode_failed: {exc}"
        finally:
            stop_event.set()

    reader_thread = threading.Thread(target=reader_loop, name=f"reader-{task_id}", daemon=True)
    infer_thread = threading.Thread(target=infer_loop, name=f"infer-{task_id}", daemon=True)
    encode_thread = threading.Thread(target=encode_loop, name=f"encode-{task_id}", daemon=True)

    TASKS[task_id]["status"] = "running"
    try:
        reader_thread.start()
        infer_thread.start()
        encode_thread.start()

        while True:
            if TASKS.get(task_id, {}).get("stop_requested", False):
                TASKS[task_id]["status"] = "stopped"
                stop_event.set()
                break

            if thread_error:
                raise RuntimeError(thread_error)

            has_live_worker = reader_thread.is_alive() or infer_thread.is_alive() or encode_thread.is_alive()
            if not has_live_worker and packet_queue.empty():
                break

            try:
                frame_id, payload = packet_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue

            pushed_count += 1
            stage_d_t0 = time.perf_counter()
            TASKS[task_id]["last_frame"] = frame_id
            TASKS[task_id]["last_alert"] = payload.get("alert", {})
            action_debug = payload.get("metrics", {}).get("action_debug", {})
            action_runs_total += int(action_debug.get("action_runs", 0))
            action_skip_warmup_total += int(action_debug.get("action_skip_warmup", 0))
            action_skip_track_age_total += int(action_debug.get("action_skip_track_age", 0))
            action_skip_gate_total += int(action_debug.get("action_skip_gate", 0))
            action_skip_cached_total += int(action_debug.get("action_skip_cached", 0))
            action_skip_default_total += int(action_debug.get("action_skip_default", 0))
            with stats_lock:
                stage_a_avg = stage_a_ms_total / stage_a_count if stage_a_count > 0 else 0.0
                stage_b_avg = stage_b_ms_total / stage_b_count if stage_b_count > 0 else 0.0
                stage_c_avg = stage_c_ms_total / stage_c_count if stage_c_count > 0 else 0.0
                stage_d_avg = stage_d_ms_total / stage_d_count if stage_d_count > 0 else 0.0
            TASKS[task_id]["stream_stats"] = {
                "source_fps": round(source_fps, 3),
                "engine_fps": round(engine_process_counter / max(1e-6, time.time() - task_started_at), 2),
                "pushed_fps": round(pushed_count / max(1e-6, time.time() - task_started_at), 2),
                "read_frames": read_counter,
                "inference_frames": infer_counter,
                "engine_process_frames": engine_process_counter,
                "replayed_frames": replay_counter,
                "encoded_frames": encoded_counter,
                "pushed_frames": pushed_count,
                "last_processed_frame_id": last_processed_frame_id,
                "dropped_frames_decode": dropped_frames,
                "dropped_frames_result": dropped_results,
                "dropped_frames_packet": dropped_packets,
                "reconnect_attempts": TASKS.get(task_id, {}).get("reconnect_attempts", 0),
                "reconnecting": bool(TASKS.get(task_id, {}).get("reconnecting", False)),
                "max_fps": max_fps,
                "startup_no_drop_seconds": startup_no_drop_seconds,
                "frame_stride": frame_stride,
                "push_stride": push_stride,
                "replay_buffer_frames": manager.get_task_history_size(task_id),
                "action_perf": {
                    "action_runs_total": action_runs_total,
                    "action_skip_warmup_total": action_skip_warmup_total,
                    "action_skip_track_age_total": action_skip_track_age_total,
                    "action_skip_gate_total": action_skip_gate_total,
                    "action_skip_cached_total": action_skip_cached_total,
                    "action_skip_default_total": action_skip_default_total,
                },
                "stage_ms": {
                    "stage_a_decode": {
                        "avg_ms": round(stage_a_avg, 3),
                        "max_ms": round(stage_a_ms_max, 3),
                        "count": stage_a_count,
                    },
                    "stage_b_inference": {
                        "avg_ms": round(stage_b_avg, 3),
                        "max_ms": round(stage_b_ms_max, 3),
                        "count": stage_b_count,
                    },
                    "stage_c_encode": {
                        "avg_ms": round(stage_c_avg, 3),
                        "max_ms": round(stage_c_ms_max, 3),
                        "count": stage_c_count,
                    },
                    "stage_d_push": {
                        "avg_ms": round(stage_d_avg, 3),
                        "max_ms": round(stage_d_ms_max, 3),
                        "count": stage_d_count,
                        "sleep_ms_total": round(stage_d_sleep_ms_total, 3),
                    },
                },
            }

            payload_metrics = payload.setdefault("metrics", {})
            payload_metrics["stream_runtime"] = {
                "read_frames": read_counter,
                "inference_frames": infer_counter,
                "engine_process_frames": engine_process_counter,
                "replayed_frames": replay_counter,
                "encoded_frames": encoded_counter,
                "pushed_frames": pushed_count,
                "last_processed_frame_id": last_processed_frame_id,
                "task_status": TASKS[task_id].get("status", "running"),
            }

            manager.record_message(task_id, payload)
            # 快照已由 AlertSystem.build_alert 在引擎侧落盘（frame 在触发点可用）；
            # 此处旧实现引用读帧线程局部变量 frame，NameError 被吞——快照链路断裂（A-1 修复）

            await manager.broadcast(task_id, payload)
            stage_d_ms = (time.perf_counter() - stage_d_t0) * 1000.0
            with stats_lock:
                stage_d_ms_total += stage_d_ms
                stage_d_ms_max = max(stage_d_ms_max, stage_d_ms)
                stage_d_count += 1
            if ws_sleep_ms > 0:
                await asyncio.sleep(ws_sleep_ms / 1000.0)
                with stats_lock:
                    stage_d_sleep_ms_total += float(ws_sleep_ms)

        if TASKS.get(task_id, {}).get("status") != "stopped":
            TASKS[task_id]["status"] = "finished"
    except Exception as exc:
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(exc)
    finally:
        stop_event.set()
        if _eval_fh is not None:
            try:
                _eval_fh.close()
            except Exception:
                pass
        if reader_thread.is_alive():
            reader_thread.join(timeout=1.0)
        if infer_thread.is_alive():
            infer_thread.join(timeout=1.0)
        if encode_thread.is_alive():
            encode_thread.join(timeout=1.0)
        stream.release()
        manager.connections.pop(task_id, None)


@app.get("/health")
def health_check():
    runtime = model_cfg.get("runtime", {})
    inference = model_cfg.get("inference", {})
    preflight = run_preflight(BASE_DIR, raw_model_cfg, active_profile)
    return {
        "status": "ok",
        "profile": active_profile,
        "available_profiles": list(raw_model_cfg.get("profiles", {}).keys()),
        "runtime": runtime,
        "inference": inference,
        "models": active_model_paths,
        "preflight_summary": preflight.get("summary", {}),
        "detector_debug": engine.detector.get_debug_info(),
        "pose_debug": engine.pose_estimator.get_debug_info(),
        "action_debug": engine.action_recognizer.get_debug_info(),
    }


@app.get("/api/v1/preflight")
def preflight(profile: str | None = None):
    target_profile = profile or active_profile
    return run_preflight(BASE_DIR, raw_model_cfg, target_profile)


@app.get("/api/v1/debug/detector")
def detector_debug():
    return {
        "profile": active_profile,
        "runtime": model_cfg.get("runtime", {}),
        "detector": engine.detector.get_debug_info(),
    }


@app.get("/api/v1/debug/pose")
def pose_debug():
    return {
        "profile": active_profile,
        "runtime": model_cfg.get("runtime", {}),
        "pose": engine.pose_estimator.get_debug_info(),
    }


@app.get("/api/v1/debug/action")
def action_debug():
    return {
        "profile": active_profile,
        "runtime": model_cfg.get("runtime", {}),
        "action": engine.action_recognizer.get_debug_info(),
        "vlm_cache": engine._vlm_cache.stats(),
        "vlm_router": engine.vlm_router.stats(),
        "pipeline": engine.pipeline_stats(),
        "smooth_window": engine._smooth_window,
    }


@app.get("/api/v1/profile")
def get_profile():
    """当前生效档位与后端健康（OM 是否真加载），供切档后轮询确认。"""
    return {"profile": active_profile, "models": active_model_paths, "backend": _backend_health(engine)}


@app.post("/api/v1/profile")
def set_profile(profile: str, background_tasks: BackgroundTasks):
    if profile not in raw_model_cfg.get("profiles", {}):
        raise HTTPException(status_code=400, detail=f"invalid profile: {profile}")
    if any(task.get("status") == "running" for task in TASKS.values()):
        raise HTTPException(status_code=409, detail="Cannot switch profile while a task is running")
    if profile != active_profile and _profile_needs_restart(profile):
        # 目标档需 NPU OM：进程内重建必回退 CPU，改为 exec 自重启（PID 不变，日志续写）
        background_tasks.add_task(_restart_into_profile, profile)
        return {
            "status": "restarting",
            "profile": profile,
            "note": (
                f"引擎正在自重启进入 {profile} 档（CANN 需新进程加载 OM），"
                f"约 15-60s 后生效；轮询 GET /api/v1/profile 确认 profile 与 backend。"
            ),
        }
    return {"status": "ok", **reconfigure_profile(profile)}


@app.post("/api/v1/analyze/stream", response_model=AnalyzeStreamResponse)
async def analyze_stream(request: AnalyzeStreamRequest):
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    replay_history_size = _estimate_task_replay_history_size(model_cfg, int(request.duration))
    manager.prepare_task(task_id, history_size=replay_history_size)
    TASKS[task_id] = {
        "status": "queued",
        "stream_url": request.stream_url,
        "last_frame": 0,
        "stop_requested": False,
        "eval_dump": bool(getattr(request, "eval_dump", False)),
    }
    asyncio.create_task(run_task(task_id, request.stream_url, request.duration))
    return AnalyzeStreamResponse(
        task_id=task_id,
        status="running",
        endpoint=f"ws://127.0.0.1:8000/ws/task/{task_id}",
    )


@app.post("/api/v1/analyze/render")
async def analyze_render(request: AnalyzeStreamRequest):
    task_id = f"render_{uuid.uuid4().hex[:8]}"
    TASKS[task_id] = {
        "status": "queued",
        "mode": "render",
        "stream_url": request.stream_url,
        "last_frame": 0,
        "render": {
            "frames_total": 0,
            "frames_processed": 0,
            "progress": 0.0,
            "fps": 0.0,
            "output_video_path": "",
        },
    }
    thread = threading.Thread(target=run_render_task, args=(task_id, request.stream_url), daemon=True)
    thread.start()
    return {
        "task_id": task_id,
        "status": "running",
        "endpoint": f"/api/v1/tasks/{task_id}",
    }


@app.websocket("/ws/task/{task_id}")
async def ws_task(websocket: WebSocket, task_id: str):
    await manager.connect(task_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(task_id, websocket)


@app.get("/api/v1/tasks/{task_id}")
def get_task(task_id: str):
    return TASKS.get(task_id, {"status": "not_found"})


@app.post("/api/v1/tasks/{task_id}/stop")
def stop_task(task_id: str):
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")

    status = task.get("status", "unknown")
    if status in {"finished", "failed", "stopped"}:
        task["status"] = "stopped"
        task["stop_requested"] = True
        return {"task_id": task_id, "status": "stopped", "detail": "task already ended"}

    task["stop_requested"] = True
    task["status"] = "stopped"
    return {"task_id": task_id, "status": "stopped", "detail": "stop requested"}


@app.get("/api/v1/events")
def get_events(limit: int = 50):
    """查询历史预警事件（与 /api/v1/alerts 同源）"""
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return {"items": read_alert_logs(results_dir, limit=limit)}


@app.get("/api/v1/scene/log")
def get_scene_log(limit: int = 50, task_id: str | None = None):
    """查询场景叙述日志（默认当前任务；可指定 task_id 查询历史任务）"""
    return {"items": engine.scene_narrator.get_log(limit=limit, task_id=task_id),
            "current_task_id": engine.scene_narrator.current_task_id}


@app.get("/api/v1/scene/status")
def get_scene_status():
    """场景叙述器 / 告警裁决器运行状态（用于前端健康面板）"""
    return {
        "narrator": engine.scene_narrator.status(),
        "adjudicator": engine.alert_adjudicator.status(),
        "frame_buffer_frames": len(engine.frame_buffer),
    }


class SceneConfigRequest(BaseModel):
    narrate_interval_seconds: float | None = None


@app.post("/api/v1/scene/config")
def update_scene_config(request: SceneConfigRequest):
    """运行时调整场景叙述间隔：改内存立即生效，并写回 configs/vlm_config.yaml。"""
    current = float(engine.scene_narrator.interval_seconds)
    if request.narrate_interval_seconds is None:
        return {"status": "ok", "narrate_interval_seconds": current}

    value = float(request.narrate_interval_seconds)
    if not (5.0 <= value <= 600.0):
        raise HTTPException(status_code=400, detail="narrate_interval_seconds 需在 5–600 之间")
    engine.scene_narrator.interval_seconds = value

    cfg_path = CONFIG_DIR / "vlm_config.yaml"
    try:
        text = cfg_path.read_text(encoding="utf-8")
        new_text, count = re.subn(
            r"(?m)^(\s*narrate_interval_seconds:\s*)[0-9.]+",
            lambda m: f"{m.group(1)}{value:g}",
            text,
            count=1,
        )
        if count:
            cfg_path.write_text(new_text, encoding="utf-8")
    except Exception as exc:
        return {"status": "partial", "narrate_interval_seconds": value, "persist_error": str(exc)}
    return {"status": "ok", "narrate_interval_seconds": value}


@app.get("/api/v1/search")
def search_scene_index(
    q: str = Query(..., min_length=1, description="自然语言检索词"),
    top_k: int = Query(10, ge=1, le=50),
    doc_type: str | None = Query(None, description="scene_narration | alert_event | behavior_prototype"),
    task_id: str | None = Query(None, description="按任务过滤"),
):
    """对场景叙述与告警事件做自然语言检索（BM25 + 可选向量重排）"""
    return scene_index.search(q, top_k=top_k, doc_type=doc_type, task_id=task_id)


@app.get("/api/v1/search/status")
def search_scene_status():
    return scene_index.status()


class AlertRuleCreateRequest(BaseModel):
    text: str


class AlertRuleUpdateRequest(BaseModel):
    text: str | None = None
    enabled: bool | None = None
    min_confidence: float | None = None


@app.post("/api/v1/alert-rules")
def create_alert_rule(request: AlertRuleCreateRequest):
    """创建自然语言告警规则（同步调用 VLM 解析，约 2-4 秒）"""
    try:
        rule = engine.nl_rule_engine.add_rule(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"规则解析失败（VLM 服务不可用？）: {exc}")
    return rule


@app.get("/api/v1/alert-rules")
def list_alert_rules():
    return {"items": engine.nl_rule_engine.list_rules(), "status": engine.nl_rule_engine.status()}


@app.get("/api/v1/metrics")
def api_metrics():
    """时刻⑤底牌角标数据源：NPU 遥测 + 引擎摘要（演示彩排允许 mock，验收用真值）"""
    try:
        from edge_migration import npu_telemetry
        allow_mock = bool(model_cfg.get("mqtt", {}).get("allow_mock", False))
        tel = npu_telemetry.read_or_mock(allow_mock=allow_mock)
    except Exception:
        tel = {"mock": False, "available": False}
    snap = getattr(engine, "last_track_snapshot", []) or []

    def _rss_mb():
        try:
            with open("/proc/self/status", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return round(int(line.split()[1]) / 1024.0, 1)
        except Exception:
            pass
        return None

    task_fps = None
    for _tid, _state in TASKS.items():
        if _state.get("status") != "running":
            continue
        _stats = _state.get("stream_stats") or {}
        if _stats.get("engine_fps") is not None:
            task_fps = _stats.get("engine_fps")
            break

    return {
        "decision_mode": engine.decision_mode,
        "persons": len(snap),
        "fps": task_fps,
        "mem_mb": _rss_mb(),
        "power_w": tel.get("power_w"),
        "temp_c": tel.get("temp_c"),
        "ai_core_util_pct": tel.get("ai_core_util_pct"),
        "npu_mock": bool(tel.get("mock", False)),
        "npu_available": bool(tel.get("available", True)),
        "llm_online": bool(engine.llm_online()),
        "timestamp": time.time(),
    }


@app.post("/api/v1/narrate")
def api_narrate(request: NarrateRequest):
    """时刻②点谁说谁：最近处理帧该目标 → 结构化事实 → 文本旁白（LLM 不可用时确定性降级）"""
    try:
        out = engine.narrate_track(request.track_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"narrate_failed: {exc}")
    if not out.get("ok") and out.get("source") == "none":
        raise HTTPException(status_code=404, detail=out.get("reason", "track_not_in_last_frame"))
    return out


@app.post("/api/v1/chat")
def api_chat(request: ChatRequest):
    """时刻③和录像聊天：场景记忆检索（scene_index）→ 文本 LLM 合成回答 + 证据"""
    # T6.5：演示模式预缓存命中直接返回（source=demo_cache，仍是真实链路的降级入口）
    if engine.demo_pack.enabled:
        cached = engine.demo_pack.cached_answer(request.question)
        if cached:
            return {"ok": True, "answer": cached, "source": "demo_cache",
                    "evidence_times": [], "evidence": [], "search_mode": "demo_cache"}
    try:
        search = scene_index.search(request.question, top_k=5)
        hits = search.get("hits", [])[:5]
        out = engine.track_narrator.answer_scene(request.question, hits)
        # 演示模式：LLM 不可用（fallback）时回退预缓存答案
        if engine.demo_pack.enabled and not out.get("ok"):
            cached = engine.demo_pack.cached_answer(request.question)
            if cached:
                out = {**out, "ok": True, "answer": cached, "source": "demo_cache"}
        out["search_mode"] = search.get("mode")
        out["evidence"] = [
            {"video_time_sec": h.get("video_time_sec"), "task_id": h.get("task_id"),
             "doc_type": h.get("doc_type"), "text": str(h.get("text", ""))[:120]}
            for h in hits]
        return out
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"chat_failed: {exc}")


@app.post("/api/v1/demo/prime")
def api_demo_prime():
    """T6.5 演示预热：走真实 narrate/chat 链路生成剧本包写回 configs/demo_pack.json"""
    try:
        narrations = []
        ids = [d["track_id"] for d in (engine.last_track_snapshot or [])][:10] or ["P001"]
        for tid in ids:
            out = engine.narrate_track(tid)
            if out.get("text"):
                narrations.append(out["text"])
        chat = []
        for q in ("刚才门口发生了什么？", "有人吸烟吗？", "有人跌倒吗？"):
            search = scene_index.search(q, top_k=5)
            ans = engine.track_narrator.answer_scene(q, search.get("hits", []))
            chat.append({"q": q, "a": ans.get("answer", "")})
        counts = engine.demo_pack.prime(narrations=narrations, chat=chat)
        return {"status": "ok", **counts,
                "hint": "演示机开机后、任务跑过一段后调用一次；拍摄时 demo.enabled 置 true"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"demo_prime_failed: {exc}")


@app.get("/api/v1/demo/status")
def api_demo_status():
    return engine.demo_pack.status()


@app.get("/api/v1/alert-rules/{rule_id}")
def get_alert_rule(rule_id: str):
    rule = engine.nl_rule_engine.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"rule not found: {rule_id}")
    return rule


@app.put("/api/v1/alert-rules/{rule_id}")
def update_alert_rule(rule_id: str, request: AlertRuleUpdateRequest):
    try:
        return engine.nl_rule_engine.update_rule(
            rule_id, text=request.text, enabled=request.enabled, min_confidence=request.min_confidence
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"rule not found: {rule_id}")


@app.delete("/api/v1/alert-rules/{rule_id}")
def delete_alert_rule(rule_id: str):
    if not engine.nl_rule_engine.remove_rule(rule_id):
        raise HTTPException(status_code=404, detail=f"rule not found: {rule_id}")
    return {"status": "deleted", "rule_id": rule_id}


# ── 布控区域 CRUD ─────────────────────────────────────────────

class ZoneCreateRequest(BaseModel):
    name: str
    zone_type: str = "forbidden"  # "forbidden" | "danger"
    polygon: list[list[float]]  # [[x1,y1],[x2,y2],...]


def _read_camera_config() -> dict:
    return load_yaml(str(CONFIG_DIR / "camera_config.yaml")) or {}


def _save_camera_config(cfg: dict) -> None:
    with open(CONFIG_DIR / "camera_config.yaml", "w", encoding="utf-8") as f:
        _yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


def _hot_reload_zones() -> None:
    """保存后热更新引擎内的布控区域，无需重启。"""
    cfg = _read_camera_config()
    zones = cfg.get("zones", {})
    engine.behavior_reasoner.update_zones(
        forbidden_zones=zones.get("forbidden_zones", []),
        danger_zones=zones.get("danger_zones", []),
    )


@app.get("/api/v1/zones")
def get_zones():
    cfg = _read_camera_config()
    zones = cfg.get("zones", {})
    return {
        "forbidden_zones": zones.get("forbidden_zones", []),
        "danger_zones": zones.get("danger_zones", []),
    }


@app.post("/api/v1/zones")
def create_zone(request: ZoneCreateRequest):
    if request.zone_type not in ("forbidden", "danger"):
        raise HTTPException(status_code=400, detail="zone_type 必须是 forbidden 或 danger")
    if len(request.polygon) < 3:
        raise HTTPException(status_code=400, detail="多边形至少需要 3 个顶点")
    cfg = _read_camera_config()
    if "zones" not in cfg:
        cfg["zones"] = {}
    key = f"{request.zone_type}_zones"
    if key not in cfg["zones"]:
        cfg["zones"][key] = []
    zone_id = f"zone_{request.zone_type}_{int(time.time())}"
    cfg["zones"][key].append({"id": zone_id, "polygon": request.polygon})
    _save_camera_config(cfg)
    _hot_reload_zones()
    return {"status": "created", "zone_id": zone_id, "zone_type": request.zone_type, "polygon": request.polygon}


@app.delete("/api/v1/zones/{zone_id}")
def delete_zone(zone_id: str):
    cfg = _read_camera_config()
    zones = cfg.get("zones", {})
    for key in ("forbidden_zones", "danger_zones"):
        zone_list = zones.get(key, [])
        before = len(zone_list)
        zones[key] = [z for z in zone_list if z.get("id") != zone_id]
        if len(zones[key]) < before:
            cfg["zones"] = zones
            _save_camera_config(cfg)
            _hot_reload_zones()
            return {"status": "deleted", "zone_id": zone_id}
    raise HTTPException(status_code=404, detail=f"zone not found: {zone_id}")


@app.get("/api/v1/alerts")
def get_alerts(limit: int = 50):
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return {"items": read_alert_logs(results_dir, limit=limit)}


# 自动标注终审页静态服务（干净 URL，避免 media 接口查询参数被复制破坏）
from fastapi.staticfiles import StaticFiles
_annotation_dir = BASE_DIR / "results" / "auto_annotation"
_annotation_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/annotation", StaticFiles(directory=str(_annotation_dir)), name="annotation_static")


@app.get("/api/v1/media")
def get_media(path: str = Query(..., description="Absolute or project-relative local video path")):
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (BASE_DIR / candidate).resolve()
        # 盒上适配: data/ 等资源在项目根而非 campus_security_agent/ 下，补一次回退。
        if not resolved.exists():
            alt = (BASE_DIR.parent / candidate).resolve()
            if alt.exists():
                resolved = alt

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="media file not found")

    media_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
    return FileResponse(path=str(resolved), media_type=media_type, filename=resolved.name)


@app.get("/api/v1/videos", include_in_schema=False)
def list_videos(dir: str = Query("", description="盒子上的视频目录；缺省为项目 data/")):
    """供远程看板选择视频源：文件枚举在盒子侧完成，PC 端无需挂载目录。"""
    root = Path(dir).expanduser() if dir.strip() else (BASE_DIR.parent / "data")
    if not root.is_absolute():
        root = BASE_DIR / root
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=404, detail=f"directory not found: {root}")
    exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}
    files = sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    return {"dir": str(root), "count": len(files), "items": files[:200], "truncated": len(files) > 200}


@app.get("/api/v1/events/recent", include_in_schema=False)
def recent_events(limit: int = Query(500, ge=1, le=5000)):
    """供远程看板做态势统计：只读最近的 event_*.json，避免一次扫全量。"""
    event_dir = BASE_DIR / "results"
    files = sorted(event_dir.glob("event_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    items = []
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        alert = payload.get("alert", {}) or {}
        if not alert.get("triggered"):
            continue
        items.append({
            "type": alert.get("type", "unknown"),
            "decision": alert.get("final_decision", "pending"),
            "timestamp": payload.get("timestamp", ""),
            "frame_id": payload.get("frame_id", 0),
            "source": payload.get("source", "pipeline"),
        })
    return {"count": len(items), "items": items}


@app.post("/api/v1/evaluate")
def evaluate(request: EvaluateRequest):
    """启动后台评估任务（系统输出 vs VLM 伪真值）。返回 eval_id，用 GET 查询结果。"""
    if not request.ground_truth:
        raise HTTPException(status_code=400, detail="ground_truth JSON 路径必填（可用 tools/build_eval_ground_truth.py 生成）")
    video_path = Path(request.video_path)
    if not video_path.exists():
        candidate = BASE_DIR.parent / request.video_path
        if candidate.exists():
            video_path = candidate
        else:
            raise HTTPException(status_code=404, detail=f"video not found: {request.video_path}")
    gt_path = Path(request.ground_truth)
    if not gt_path.exists():
        raise HTTPException(status_code=404, detail=f"ground_truth not found: {request.ground_truth}")

    eval_id = f"eval_{int(time.time() * 1000)}"
    EVAL_JOBS[eval_id] = {"status": "running", "progress": 0.0, "report": None, "error": ""}

    def job():
        def on_progress(payload: dict) -> None:
            if payload.get("stage") == "processing":
                EVAL_JOBS[eval_id]["progress"] = payload.get("progress", 0.0)
        try:
            from src.evaluation.evaluator import run_evaluation

            report = run_evaluation(
                video_path=str(video_path),
                ground_truth_path=str(gt_path),
                profile=request.profile,
                baseline=request.baseline,
                use_scene_context=request.use_scene_context,
                frame_stride=request.frame_stride,
                max_frames=request.max_frames,
                progress_cb=on_progress,
            )
            EVAL_JOBS[eval_id]["report"] = report
            EVAL_JOBS[eval_id]["status"] = "done"
            EVAL_JOBS[eval_id]["progress"] = 1.0
        except Exception as exc:
            EVAL_JOBS[eval_id]["status"] = "failed"
            EVAL_JOBS[eval_id]["error"] = str(exc)

    threading.Thread(target=job, daemon=True).start()
    return {"eval_id": eval_id, "status": "running", "endpoint": f"/api/v1/evaluate/{eval_id}"}


@app.get("/api/v1/evaluate/{eval_id}")
def get_evaluate(eval_id: str):
    job = EVAL_JOBS.get(eval_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"eval job not found: {eval_id}")
    payload = dict(job)
    if job.get("status") == "done" and job.get("report"):
        report = dict(job["report"])
        report.pop("records", None)  # 列表过大，详情看报告文件
        payload["report"] = report
    return payload


@app.post("/api/v1/train")
def train(request: TrainRequest):
    """启动后台 PoseClassifier 训练任务。返回 train_id，用 GET 查询进度。"""
    if TRAIN_JOBS.get("_active", {}).get("running"):
        raise HTTPException(status_code=409, detail="已有训练任务在运行")

    train_id = f"train_{int(time.time() * 1000)}"
    TRAIN_JOBS[train_id] = {"status": "running", "stage": "starting", "history": [],
                            "summary": None, "error": "", "request": request.model_dump()}
    TRAIN_JOBS["_active"] = {"running": True, "train_id": train_id}

    def job():
        try:
            from src.training.pose_classifier import train_pose_classifier

            def on_progress(payload: dict) -> None:
                state = TRAIN_JOBS[train_id]
                state["stage"] = payload.get("stage", "")
                if payload.get("stage") == "epoch":
                    state["history"].append({k: v for k, v in payload.items() if k != "stage"})
                elif payload.get("stage") == "done":
                    state["summary"] = payload

            summary = train_pose_classifier(
                data_dir=request.dataset_path if Path(request.dataset_path).is_absolute()
                else str(BASE_DIR / request.dataset_path),
                output_dir=str(BASE_DIR / "models" / "action_v2"),
                epochs=request.epochs,
                batch_size=request.batch_size,
                device=request.device,
                progress_cb=on_progress,
            )
            TRAIN_JOBS[train_id]["status"] = "done"
            TRAIN_JOBS[train_id]["summary"] = {
                "best_val_acc": summary["best_val_acc"],
                "onnx_path": summary["onnx_path"],
                "epochs": summary["epochs"],
            }
        except Exception as exc:
            TRAIN_JOBS[train_id]["status"] = "failed"
            TRAIN_JOBS[train_id]["error"] = str(exc)
        finally:
            TRAIN_JOBS["_active"] = {"running": False, "train_id": ""}

    threading.Thread(target=job, daemon=True).start()
    return {
        "status": "submitted",
        "train_id": train_id,
        "endpoint": f"/api/v1/train/{train_id}",
        "request": request.model_dump(),
    }


@app.get("/api/v1/train/{train_id}")
def get_train(train_id: str):
    job = TRAIN_JOBS.get(train_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"train job not found: {train_id}")
    return job


@app.post("/api/v1/vlm/review", response_model=VLMReviewResponse)
def vlm_review(request: VLMReviewRequest):
    """对单个候选事件做真实 VLM 复核（与推理流水线共用同一 VLMReviewer 实例）。"""
    event = request.candidate_event

    frames: list = []
    frame_b64 = event.get("frame_b64", "")
    if frame_b64:
        nparr = np.frombuffer(base64.b64decode(frame_b64), np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if decoded is not None:
            frames.append(decoded)
    track_crops = event.get("track_crops", [])
    for crop in track_crops:
        if isinstance(crop, (bytes, bytearray)):
            nparr = np.frombuffer(bytes(crop), np.uint8)
            decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if decoded is not None:
                frames.append(decoded)
        elif isinstance(crop, np.ndarray):
            frames.append(crop)

    result = engine.vlm_reviewer.review_clip(frames=frames, track_id=event.get("track_id", ""))

    if result.get("status") not in {"success"}:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "VLM 复核不可用，未产生复核结论",
                "status": result.get("status", "error"),
                "reason": result.get("reasoning", ""),
            },
        )

    return VLMReviewResponse(
        event_id=event.get("event_id", f"evt_{request.frame_id}"),
        review_status="accepted",
        reviewed_behavior={
            "type": result.get("label", "Normal"),
            "confidence": float(result.get("confidence", 0.0)),
            "risk_level": event.get("risk_level", "LOW"),
        },
        role_scores=event.get("role_scores", {}),
        reason_tags=result.get("reason_tags", []),
        summary=result.get("reasoning", ""),
        latency_ms=int(result.get("latency_ms", 0)),
        model_name=engine.vlm_reviewer.model_name,
    )
