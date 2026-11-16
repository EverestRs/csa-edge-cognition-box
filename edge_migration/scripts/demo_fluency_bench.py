#!/usr/bin/env python3
"""演示流畅度实测（浏览器视角）。

看板架构：浏览器本地播源视频（HTTP /api/v1/media）+ WS 收检测事件画框，
maybeSyncPlayback 用事件 frame_id 门控视频进度 → 流畅度取决于：
  ① WS 事件到达率与间隔抖动（框更新）
  ② frame_id 推进速率 ≈ 源 fps（实时覆盖，视频才能 1x 不卡顿）
本脚本以 WS 客户端模拟浏览器收流，同时采样管道计数器与 CPU/内存/功耗，落盘证据 JSON。

用法: ~/.conda/envs/csa/bin/python edge_migration/scripts/demo_fluency_bench.py \
        --duration 75 --out results/edge_test/demo_fluency_20261111.json
"""
import argparse
import asyncio
import json
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
    return round(s[k], 1)


def find_procs():
    want = {"engine": "run.py", "ui": "streamlit", "llm": "llama-server",
            "sddm": "sddm-greeter", "xorg": "Xorg"}
    found = {}
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        for tag, marker in want.items():
            if tag in found:
                continue
            if marker in cmd and ("grep" not in cmd):
                if tag == "engine" and "python" not in cmd:
                    continue
                found[tag] = p
    return found


def media_probe(video_rel):
    """视频 HTTP 供流能力：Range 下载吞吐 vs 视频码率（浏览器 1x 播放的需求）。"""
    import cv2
    path = BASE.parent / video_rel
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    dur = nframes / fps if fps else 0
    size = path.stat().st_size
    bitrate_mbps = size * 8 / dur / 1e6 if dur else None
    t0 = time.monotonic()
    r = requests.get(f"{API}/api/v1/media", params={"path": video_rel},
                     headers={"Range": "bytes=0-8388607"}, timeout=30, stream=True)
    got = 0
    for chunk in r.iter_content(1 << 20):
        got += len(chunk)
        if got >= 8 << 20:
            break
    r.close()
    dt = time.monotonic() - t0
    return {
        "video": video_rel, "source_fps": round(fps, 2), "duration_s": round(dur, 1),
        "size_mb": round(size / 1e6, 1), "bitrate_mbps": round(bitrate_mbps, 2) if bitrate_mbps else None,
        "range_status": r.status_code, "fetched_mb": round(got / 1e6, 2),
        "throughput_mbps": round(got * 8 / dt / 1e6, 1),
        "headroom_x": round((got * 8 / dt / 1e6) / bitrate_mbps, 1) if bitrate_mbps else None,
    }


async def ws_collect(task_id, duration):
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
                continue  # 回放/乱序去重，与看板 pushEvent 同口径
            last_fid = fid
            events.append({
                "t": round(t - t0, 4), "frame_id": fid,
                "dets": len(msg.get("detections") or []),
                "bytes": len(raw),
                "alert": bool((msg.get("alert") or {}).get("triggered")),
            })
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="rtsp://127.0.0.1:8554/live")
    ap.add_argument("--video", default="data/laying_corner_1-1.mp4")
    ap.add_argument("--duration", type=int, default=75)
    ap.add_argument("--out", default="results/edge_test/demo_fluency.json")
    args = ap.parse_args()

    procs = find_procs()
    for p in procs.values():
        try:
            p.cpu_percent(None)
        except psutil.NoSuchProcess:
            pass
    psutil.cpu_percent(None)

    media = media_probe(args.video)

    r = requests.post(f"{API}/api/v1/analyze/stream",
                      json={"stream_url": args.source, "duration": args.duration + 30}, timeout=10)
    task_id = r.json()["task_id"]
    print(f"[bench] task={task_id} source={args.source} 测量 {args.duration}s")

    samples = []
    stop_flag = threading.Event()

    def sampler():
        while not stop_flag.is_set():
            row = {"t": round(time.monotonic(), 2)}
            try:
                task = requests.get(f"{API}/api/v1/tasks/{task_id}", timeout=5).json()
                row["stats"] = task.get("stream_stats", {})
                row["status"] = task.get("status")
            except Exception as e:
                row["task_err"] = str(e)
            try:
                row["metrics"] = requests.get(f"{API}/api/v1/metrics", timeout=5).json()
            except Exception as e:
                row["metrics_err"] = str(e)
            row["sys_cpu_pct"] = psutil.cpu_percent(None)
            row["proc_cpu"] = {}
            for tag, p in procs.items():
                try:
                    row["proc_cpu"][tag] = p.cpu_percent(None)
                except psutil.NoSuchProcess:
                    pass
            samples.append(row)
            stop_flag.wait(5)

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

    # ── WS 事件流（浏览器视角）──
    ws_stats = {"events": len(events)}
    if len(events) >= 2:
        span = events[-1]["t"] - events[0]["t"]
        gaps = [(b["t"] - a["t"]) * 1000 for a, b in zip(events, events[1:])]
        fid_rate = (events[-1]["frame_id"] - events[0]["frame_id"]) / span
        win = {}
        for e in events:
            win.setdefault(int(e["t"] // 5), []).append(e)
        ws_stats.update({
            "span_s": round(span, 1),
            "event_fps": round(len(events) / span, 2),
            "gap_ms_p50": pctl(gaps, 50), "gap_ms_p95": pctl(gaps, 95), "gap_ms_max": pctl(gaps, 100),
            "gaps_gt_300ms": sum(g > 300 for g in gaps),
            "gaps_gt_1000ms": sum(g > 1000 for g in gaps),
            "frame_id_rate": round(fid_rate, 1),
            "realtime_coverage": round(fid_rate / media["source_fps"], 3),
            "payload_bytes_avg": round(sum(e["bytes"] for e in events) / len(events)),
            "alert_events": sum(e["alert"] for e in events),
            "fps_per_5s_window": [round(len(v) / 5, 1) for _, v in sorted(win.items())],
        })

    # ── 管道计数器（首尾差分）──
    stats_rows = [s["stats"] for s in samples if s.get("stats")]
    pipe = {}
    if len(stats_rows) >= 2:
        a, b = stats_rows[0], stats_rows[-1]
        dt = samples[-1]["t"] - samples[0]["t"]
        for key in ("read_frames", "inference_frames", "engine_process_frames",
                    "encoded_frames", "pushed_frames", "replayed_frames"):
            if key in a and key in b:
                pipe[key.replace("_frames", "") + "_fps"] = round((b[key] - a[key]) / dt, 2)
        pipe["dropped_decode"] = b.get("dropped_frames_decode", 0) - a.get("dropped_frames_decode", 0)
        pipe["dropped_result"] = b.get("dropped_frames_result", 0) - a.get("dropped_frames_result", 0)
        pipe["dropped_packet"] = b.get("dropped_frames_packet", 0) - a.get("dropped_frames_packet", 0)
        pipe["stage_ms_final"] = b.get("stage_ms", {})
        pipe["profile"] = {k: b.get(k) for k in ("max_fps", "frame_stride", "push_stride", "source_fps")}

    # ── 资源 ──
    def series(path):
        out = []
        for s in samples:
            cur = s
            for k in path:
                cur = cur.get(k, {}) if isinstance(cur, dict) else {}
            if isinstance(cur, (int, float)):
                out.append(cur)
        return out

    power, temp, aicore, mem = series(("metrics", "power_w")), series(("metrics", "temp_c")), \
        series(("metrics", "ai_core_util_pct")), series(("metrics", "mem_mb"))
    cpu_avg = {}
    for tag in procs:
        vals = [s["proc_cpu"].get(tag) for s in samples if s.get("proc_cpu", {}).get(tag) is not None]
        if vals:
            cpu_avg[tag] = round(sum(vals) / len(vals), 1)
    sys_vals = [s["sys_cpu_pct"] for s in samples if s.get("sys_cpu_pct") is not None]
    resources = {
        "proc_cpu_pct_avg": cpu_avg,
        "system_cpu_pct_avg": round(sum(sys_vals) / len(sys_vals), 1) if sys_vals else None,
        "power_w_avg": round(sum(power) / len(power), 2) if power else None,
        "power_w_max": max(power) if power else None,
        "temp_c_max": max(temp) if temp else None,
        "ai_core_util_pct_avg": round(sum(aicore) / len(aicore), 1) if aicore else None,
        "engine_mem_mb_max": max(mem) if mem else None,
        "loadavg": list(psutil.getloadavg()),
    }

    # ── 判定 ──
    verdict, reasons = "PASS", []
    efps = ws_stats.get("event_fps", 0)
    cov = ws_stats.get("realtime_coverage", 0)
    g95 = ws_stats.get("gap_ms_p95") or 9999
    if efps < 6:
        verdict = "FAIL"; reasons.append(f"事件率 {efps}fps < 6")
    if cov < 0.9:
        verdict = "FAIL"; reasons.append(f"实时覆盖 {cov} < 0.9（视频会被同步门控拖慢）")
    if g95 > 500:
        verdict = "WARN" if verdict == "PASS" else verdict
        reasons.append(f"事件间隔 p95 {g95}ms > 500（框更新有顿挫）")
    if ws_stats.get("gaps_gt_1000ms", 0) > 2:
        verdict = "WARN" if verdict == "PASS" else verdict
        reasons.append(f">1s 断流 {ws_stats['gaps_gt_1000ms']} 次")

    result = {
        "meta": {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "task_id": task_id,
            "source": args.source, "duration_s": args.duration,
            "sddm_running": "sddm" in procs, "llm_nice": 19,
            "method": "WS 客户端模拟浏览器收流 + 管道计数器差分 + psutil 采样",
        },
        "media_serving": media, "ws_browser_view": ws_stats,
        "pipeline": pipe, "resources": resources,
        "verdict": verdict, "reasons": reasons,
        "samples": samples, "ws_events": events,
    }
    out = BASE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps({k: result[k] for k in ("media_serving", "ws_browser_view", "pipeline", "resources", "verdict", "reasons")},
                     ensure_ascii=False, indent=1))
    print(f"[bench] 证据已落盘: {out}")


if __name__ == "__main__":
    main()