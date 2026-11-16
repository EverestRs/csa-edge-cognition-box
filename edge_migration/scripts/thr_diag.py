#!/usr/bin/env python3
"""吞吐档失速诊断：活体采样管道队列积压/真实完成率/线程数/AICore。"""
import json
import subprocess
import time
from pathlib import Path

import psutil
import requests

API = "http://127.0.0.1:8000"
BASE = Path(__file__).resolve().parents[2]


def engine_pid():
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "run.py" in cmd and "python" in cmd:
            return p.pid
    return None


def npu():
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {}
    import re
    d = re.search(r"\|\s*0\s+0\s*\|\s*NA\s*\|\s*(\d+)\s+(\d+)\s*/\s*(\d+)", out)
    return {"ai_core": int(d.group(1))} if d else {}


def threads(pid):
    try:
        return len(list(Path(f"/proc/{pid}/task").iterdir()))
    except Exception:
        return None


def switch_profile(name):
    """切档；OM 档会触发引擎 exec 自重启（PID 不变），轮询直至新档就绪。"""
    r = requests.post(f"{API}/api/v1/profile", params={"profile": name}, timeout=60)
    r.raise_for_status()
    if r.json().get("status") != "restarting":
        return
    deadline = time.time() + 150
    while time.time() < deadline:
        time.sleep(3)
        try:
            cur = requests.get(f"{API}/api/v1/profile", timeout=5).json()
        except Exception:
            continue
        if cur.get("profile") == name:
            print(f"[diag] 引擎自重启完成，profile={name}")
            return
    raise RuntimeError(f"等待引擎重启进入 {name} 档超时（150s）")


def main():
    pid = engine_pid()
    switch_profile("throughput")
    print(f"[diag] throughput 档已加载 engine_pid={pid}")
    r = requests.post(f"{API}/api/v1/analyze/stream",
                      json={"stream_url": "rtsp://127.0.0.1:8554/live", "duration": 40}, timeout=10)
    tid = r.json()["task_id"]
    print(f"[diag] task={tid}")
    rows = []
    prev = None
    t_end = time.monotonic() + 30
    while time.monotonic() < t_end:
        time.sleep(3)
        dbg = requests.get(f"{API}/api/v1/debug/action", timeout=5).json().get("pipeline", {})
        st = requests.get(f"{API}/api/v1/tasks/{tid}", timeout=5).json().get("stream_stats", {})
        row = {
            "q1": dbg.get("q1"), "q2": dbg.get("q2"), "q3": dbg.get("q3"),
            "pipe_dropped": dbg.get("dropped"), "pending": dbg.get("pending_results"),
            "last_err": dbg.get("last_error"),
            "read": st.get("read_frames"), "infer": st.get("inference_frames"),
            "engine_proc": st.get("engine_process_frames"), "replayed": st.get("replayed_frames"),
            "pushed": st.get("pushed_frames"),
            "ai_core": npu().get("ai_core"), "threads": threads(pid),
            "load1": round(psutil.getloadavg()[0], 1),
        }
        if prev:
            dt = 3.0
            if row["engine_proc"] is not None and prev["engine_proc"] is not None:
                row["real_complete_fps"] = round((row["engine_proc"] - prev["engine_proc"]) / dt, 2)
            if row["replayed"] is not None and prev["replayed"] is not None:
                row["replay_fps"] = round((row["replayed"] - prev["replayed"]) / dt, 2)
        rows.append(row)
        prev = row
        print(json.dumps(row, ensure_ascii=False))
    requests.post(f"{API}/api/v1/tasks/{tid}/stop", timeout=5)
    switch_profile("edge")
    print("[diag] 已停任务并切回 edge")
    (BASE / "results/edge_test/thr_diag.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()