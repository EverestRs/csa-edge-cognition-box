#!/usr/bin/env python3
"""高吞吐档基准（参赛性能指标）。

切到 throughput 档（全帧检测、关姿态/动作旁路、不限频），起任务后以管道计数器差分
测**推理 fps**（detector 吞吐），并采样 npu-smi AICore%/功耗/温度、detector 单帧延迟、
CPU/内存。测毕停任务并切回 edge 档（恢复演示态）。落盘证据 JSON。

用法: ~/.conda/envs/csa/bin/python edge_migration/scripts/throughput_bench.py \
        --duration 40 --out results/edge_test/throughput_20261111.json
"""
import argparse
import asyncio
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import psutil
import requests
import websockets

BASE = Path(__file__).resolve().parents[2]
API = "http://127.0.0.1:8000"


def pctl(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return round(s[k], 2)


def find_engine():
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "run.py" in cmd and "python" in cmd and "grep" not in cmd:
            return p
    return None


def npu_smi():
    """解析 npu-smi info：AICore%、功耗W、温度C、显存MB。"""
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {}
    res = {}
    m = re.search(r"310B\d\s*\|\s*\w+\s*\|\s*([\d.]+)\s+(\d+)", out)
    if m:
        res["power_w"] = float(m.group(1))
        res["temp_c"] = int(m.group(2))
    d = re.search(r"\|\s*0\s+0\s*\|\s*NA\s*\|\s*(\d+)\s+(\d+)\s*/\s*(\d+)", out)
    if d:
        res["ai_core_pct"] = int(d.group(1))
        res["npu_mem_used_mb"] = int(d.group(2))
        res["npu_mem_total_mb"] = int(d.group(3))
    return res


def switch_profile(name):
    r = requests.post(f"{API}/api/v1/profile", params={"profile": name}, timeout=60)
    r.raise_for_status()
    prof = r.json()
    if prof.get("status") == "restarting":
        # OM 档切换触发引擎 exec 自重启（CANN 需新进程），轮询直至新档就绪
        deadline = time.time() + 150
        while time.time() < deadline:
            time.sleep(3)
            try:
                cur = requests.get(f"{API}/api/v1/profile", timeout=5).json()
            except Exception:
                continue
            if cur.get("profile") == name:
                if (cur.get("backend") or {}).get("degraded"):
                    raise RuntimeError(f"重启后 OM 仍回退 CPU ONNX，请查 CANN 环境（§8.3）")
                print(f"[thr] 引擎自重启完成，profile={name}")
                return {"profile": name}
        raise RuntimeError(f"等待引擎重启进入 {name} 档超时（150s）")
    return prof


async def ws_collect(task_id, duration):
    """WS 客户端收流：逐帧 detector 延迟 + 有框覆盖（与看板同源 payload）。"""
    url = f"ws://127.0.0.1:8000/ws/task/{task_id}"
    events = []
    async with websockets.connect(url, max_size=16 << 20) as ws:
        await ws.send("start")
        t0 = time.monotonic()
        last_fid = 0
        while time.monotonic() - t0 < duration:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            t = time.monotonic()
            msg = json.loads(raw)
            fid = msg.get("frame_id")
            if not isinstance(fid, int) or fid <= last_fid:
                continue  # 回放/乱序去重
            last_fid = fid
            psm = (msg.get("metrics") or {}).get("pipeline_stage_ms") or {}
            events.append({
                "t": round(t - t0, 4), "frame_id": fid,
                "dets": len(msg.get("detections") or []),
                "detector_ms": psm.get("detector"),
                "pipeline_compute_ms": psm.get("pipeline_compute"),
                "alert": bool((msg.get("alert") or {}).get("triggered")),
            })
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="rtsp://127.0.0.1:8554/live")
    ap.add_argument("--duration", type=int, default=40)
    ap.add_argument("--profile", default="throughput")
    ap.add_argument("--restore", default="edge")
    ap.add_argument("--no-switch", action="store_true",
                    help="不切档（引擎已以目标档冷启动时用；CANN 进程内重初始化会回退 CPU ONNX）")
    ap.add_argument("--out", default="results/edge_test/throughput.json")
    args = ap.parse_args()

    eng = find_engine()
    if eng:
        try:
            eng.cpu_percent(None)
        except psutil.NoSuchProcess:
            pass
    psutil.cpu_percent(None)

    if args.no_switch:
        print(f"[thr] --no-switch：沿用引擎当前档（冷启动档）")
    else:
        prof = switch_profile(args.profile)
        print(f"[thr] 切到 {args.profile} 档: inference={json.dumps(prof.get('inference', {}), ensure_ascii=False)[:200]}")

    # 记录 detector 实际生效后端（.om=NPU / .onnx=CPU 回退），证据自校验
    try:
        det_dbg = requests.get(f"{API}/api/v1/debug/detector", timeout=5).json().get("detector", {})
        backend_resolved = det_dbg.get("onnx_path_resolved", "")
        print(f"[thr] detector 生效模型: {backend_resolved.split('/')[-1]}")
    except Exception:
        backend_resolved = ""

    r = requests.post(f"{API}/api/v1/analyze/stream",
                      json={"stream_url": args.source, "duration": args.duration + 30}, timeout=10)
    task_id = r.json()["task_id"]
    print(f"[thr] task={task_id} source={args.source} 测量 {args.duration}s")

    samples = []
    stop_flag = threading.Event()

    def sampler():
        while not stop_flag.is_set():
            row = {"t": round(time.monotonic(), 2)}
            try:
                task = requests.get(f"{API}/api/v1/tasks/{task_id}", timeout=5).json()
                row["stats"] = task.get("stream_stats", {})
            except Exception as e:
                row["task_err"] = str(e)
            try:
                row["metrics"] = requests.get(f"{API}/api/v1/metrics", timeout=5).json()
            except Exception:
                pass
            row["npu"] = npu_smi()
            row["sys_cpu_pct"] = psutil.cpu_percent(None)
            if eng:
                try:
                    row["engine_cpu_pct"] = eng.cpu_percent(None)
                except psutil.NoSuchProcess:
                    pass
            samples.append(row)
            stop_flag.wait(2)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    time.sleep(3)  # 起流预热
    events = asyncio.run(ws_collect(task_id, args.duration))
    stop_flag.set()
    th.join(timeout=8)
    try:
        requests.post(f"{API}/api/v1/tasks/{task_id}/stop", timeout=5)
    except Exception:
        pass

    # ── 管道计数器差分（推理吞吐为核心指标）──
    stats_rows = [s["stats"] for s in samples if s.get("stats")]
    pipe = {}
    if len(stats_rows) >= 2:
        a, b = stats_rows[0], stats_rows[-1]
        dt = samples[-1]["t"] - samples[0]["t"]
        for key in ("read_frames", "inference_frames", "engine_process_frames",
                    "encoded_frames", "pushed_frames"):
            if key in a and key in b:
                pipe[key.replace("_frames", "") + "_fps"] = round((b[key] - a[key]) / dt, 2)
        pipe["dropped_decode"] = b.get("dropped_frames_decode", 0) - a.get("dropped_frames_decode", 0)
        pipe["dropped_result"] = b.get("dropped_frames_result", 0) - a.get("dropped_frames_result", 0)
        sb = b.get("stage_ms", {}).get("stage_b_inference", {})
        pipe["stage_b_inference_avg_ms"] = sb.get("avg_ms")
        pipe["stage_ms_final"] = b.get("stage_ms", {})
        pipe["profile"] = {k: b.get(k) for k in ("max_fps", "frame_stride", "push_stride",
                                                 "detector_stride", "source_fps")}

    # ── WS 视角：detector 单帧延迟 + 有框覆盖 ──
    det_lat = [e["detector_ms"] for e in events if isinstance(e.get("detector_ms"), (int, float))]
    ws_view = {"events": len(events)}
    if len(events) >= 2:
        span = events[-1]["t"] - events[0]["t"]
        ws_view.update({
            "span_s": round(span, 1),
            "push_event_fps": round(len(events) / span, 2),
            "frames_with_boxes": sum(e["dets"] > 0 for e in events),
            "box_coverage": round(sum(e["dets"] > 0 for e in events) / len(events), 3),
            "dets_avg": round(sum(e["dets"] for e in events) / len(events), 2),
            "alert_events": sum(e["alert"] for e in events),
        })

    # ── NPU / 资源 ──
    ai = [s["npu"]["ai_core_pct"] for s in samples if s.get("npu", {}).get("ai_core_pct") is not None]
    pw = [s["npu"]["power_w"] for s in samples if s.get("npu", {}).get("power_w") is not None]
    tp = [s["npu"]["temp_c"] for s in samples if s.get("npu", {}).get("temp_c") is not None]
    mem = [s["npu"]["npu_mem_used_mb"] for s in samples if s.get("npu", {}).get("npu_mem_used_mb") is not None]
    eng_cpu = [s["engine_cpu_pct"] for s in samples if s.get("engine_cpu_pct") is not None]
    sys_cpu = [s["sys_cpu_pct"] for s in samples if s.get("sys_cpu_pct") is not None]
    resources = {
        "npu_ai_core_pct_avg": round(sum(ai) / len(ai), 1) if ai else None,
        "npu_ai_core_pct_max": max(ai) if ai else None,
        "power_w_avg": round(sum(pw) / len(pw), 2) if pw else None,
        "power_w_max": max(pw) if pw else None,
        "temp_c_max": max(tp) if tp else None,
        "npu_mem_used_mb_max": max(mem) if mem else None,
        "engine_cpu_pct_avg": round(sum(eng_cpu) / len(eng_cpu), 1) if eng_cpu else None,
        "system_cpu_pct_avg": round(sum(sys_cpu) / len(sys_cpu), 1) if sys_cpu else None,
        "loadavg": list(psutil.getloadavg()),
    }

    result = {
        "meta": {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "task_id": task_id,
            "profile": args.profile, "source": args.source, "duration_s": args.duration,
            "detector_backend": backend_resolved, "no_switch": bool(args.no_switch),
            "method": "管道计数器差分(推理fps) + WS逐帧detector延迟 + npu-smi采样 + psutil",
        },
        "pipeline": pipe,
        "detector_latency_ms": {"p50": pctl(det_lat, 50), "p95": pctl(det_lat, 95),
                                "max": pctl(det_lat, 100), "samples": len(det_lat)} if det_lat else None,
        "ws_view": ws_view,
        "resources": resources,
        "samples": samples,
        "ws_events": events,
    }
    out = BASE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    # 恢复演示档（--no-switch 时由调用方自行重启引擎恢复）
    if not args.no_switch:
        try:
            switch_profile(args.restore)
            print(f"[thr] 已切回 {args.restore} 档")
        except Exception as e:
            print(f"[thr] 警告：切回 {args.restore} 失败 {e}")

    print(json.dumps({k: result[k] for k in ("pipeline", "detector_latency_ms", "ws_view", "resources")},
                     ensure_ascii=False, indent=1))
    print(f"[thr] 证据已落盘: {out}")


if __name__ == "__main__":
    main()