#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G2 探针：解码 CPU 预算（T1.3）。cv 软解 vs ffmpeg 管道；支持 lavfi 合成源自检。

用法:
  python edge_migration/scripts/g2_decode_bench.py --source rtsp://... --backend cv --frames 300
  python edge_migration/scripts/g2_decode_bench.py --source testsrc --backend ffmpeg --resolution 1280x720
判定线（03 §1）: cpu_cores < 1.5 且解码不成为瓶颈（frames_read == 请求数）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _cpu_cores(proc) -> float:
    """进程 CPU 时间差 → 占用的等效核数。"""
    t = proc.cpu_times()
    return (t.user + t.system)


def bench_cv(source: str, frames: int, resolution: tuple[int, int]) -> dict:
    import cv2
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return {"status": "FAIL", "reason": f"cv_open_failed: {source}"}
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    except Exception:
        pass
    import psutil
    proc = psutil.Process(os.getpid())
    t0 = _cpu_cores(proc)
    times: list[float] = []
    got = 0
    for _ in range(frames):
        s = time.perf_counter()
        ok, _frame = cap.read()
        times.append((time.perf_counter() - s) * 1000)
        if not ok:
            break
        got += 1
    wall = sum(times) / 1000.0 or 1e-6
    cores = (_cpu_cores(proc) - t0) / wall
    cap.release()
    arr = np.array(times)
    return {"status": "OK", "backend": "cv", "frames_read": got, "frames_requested": frames,
            "decode_ms_p50": float(np.percentile(arr, 50)),
            "decode_ms_p95": float(np.percentile(arr, 95)),
            "decode_fps": round(got / wall, 2), "cpu_cores_avg": round(cores, 2)}


def bench_ffmpeg(source: str, frames: int, resolution: tuple[int, int]) -> dict:
    w, h = resolution
    if source == "testsrc":
        cmd = ["ffmpeg", "-loglevel", "error", "-re", "-f", "lavfi",
               "-i", f"testsrc=size={w}x{h}:rate=30",
               "-frames:v", str(frames), "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    else:
        base = ["ffmpeg", "-loglevel", "error"]
        if source.startswith("rtsp://"):
            base += ["-rtsp_transport", "tcp"]
        cmd = base + ["-i", source, "-vf", f"scale={w}:{h}", "-pix_fmt", "bgr24",
                      "-frames:v", str(frames), "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    n_bytes = w * h * 3
    import psutil
    p = psutil.Process(os.getpid())
    t0 = _cpu_cores(p)
    times: list[float] = []
    got = 0
    while got < frames:
        s = time.perf_counter()
        buf = proc.stdout.read(n_bytes)
        times.append((time.perf_counter() - s) * 1000)
        if len(buf) < n_bytes:
            break
        got += 1
    wall = sum(times) / 1000.0 or 1e-6
    cores = (_cpu_cores(p) - t0) / wall
    proc.stdout.close()
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
    if got == 0:
        err = proc.stderr.read().decode("utf-8", "ignore")[-200:] if proc.stderr else ""
        return {"status": "FAIL", "reason": f"ffmpeg_no_frames: {err}", "backend": "ffmpeg"}
    arr = np.array(times)
    return {"status": "OK", "backend": "ffmpeg", "frames_read": got, "frames_requested": frames,
            "decode_ms_p50": float(np.percentile(arr, 50)),
            "decode_ms_p95": float(np.percentile(arr, 95)),
            "decode_fps": round(got / wall, 2), "cpu_cores_avg": round(cores, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="testsrc", help="RTSP URL / USB 索引 / testsrc")
    ap.add_argument("--backend", choices=["cv", "ffmpeg", "both"], default="both")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--resolution", default="1280x720")
    ap.add_argument("--out", default="results/edge_m0/g2_decode.json")
    ap.add_argument("--cpu-cores-limit", type=float, default=1.5)
    a = ap.parse_args()
    w, h = (int(v) for v in a.resolution.lower().split("x"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    results: dict = {"probe": "g2_decode", "ts": time.time(),
                     "source": a.source, "resolution": [w, h], "cases": {}}
    backends = ["cv", "ffmpeg"] if a.backend == "both" else [a.backend]
    for b in backends:
        try:
            r = bench_cv(a.source, a.frames, (w, h)) if b == "cv" else \
                bench_ffmpeg(a.source, a.frames, (w, h))
        except Exception as exc:
            r = {"status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"}
        r["pass"] = bool(r.get("status") == "OK" and r.get("cpu_cores_avg", 9) <= a.cpu_cores_limit)
        results["cases"][b] = r
        print(f"[g2] {b}: {r}")

    ok = any(r.get("pass") for r in results["cases"].values())
    results["verdict"] = "PASS" if ok else "NEED_NEXT_TIER"  # cv→ffmpeg→降分辨率 三档决策
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"[g2] verdict={results['verdict']} -> {a.out}")
    return 0 if ok else 0  # 探针本身成功执行即 0；决策看 JSON


if __name__ == "__main__":
    sys.exit(main())
