#!/usr/bin/env python3
"""NPU 检测吞吐上限微基准（参赛头条指标，隔离 CPU/系统争用）。

冷启动独占 NPU：加载 yolov8n.om，用真实帧预处理张量紧循环 adapter.run，
测纯 NPU 单帧延迟与持续吞吐 fps，并采样 npu-smi AICore%/功耗。
与端到端管道吞吐（throughput_bench.py）区分：本基准只量 NPU 推理本身。

用法（须先停引擎释放 NPU，正确 CANN 环境顺序：先 unset 再 source）:
  unset LD_LIBRARY_PATH; source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ~/.conda/envs/csa/bin/python edge_migration/scripts/npu_throughput_probe.py --duration 12
"""
import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[2]


def pctl(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return round(s[k], 2)


def npu_smi():
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {}
    res = {}
    m = re.search(r"310B\d\s*\|\s*\w+\s*\|\s*([\d.]+)\s+(\d+)", out)
    if m:
        res["power_w"] = float(m.group(1)); res["temp_c"] = int(m.group(2))
    d = re.search(r"\|\s*0\s+0\s*\|\s*NA\s*\|\s*(\d+)\s+(\d+)\s*/\s*(\d+)", out)
    if d:
        res["ai_core_pct"] = int(d.group(1))
    return res


def make_tensor(video_rel):
    path = BASE.parent / video_rel
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
    # OM 编译输入为 640×640（与 detector 内部 letterbox 目标一致）
    frame = cv2.resize(frame, (640, 640))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = rgb.astype(np.float32) / 255.0
    t = np.transpose(t, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--video", default="data/540p/laying_corner_1-1.mp4")
    ap.add_argument("--out", default="results/edge_test/npu_throughput.json")
    args = ap.parse_args()

    from edge_migration.inference_backend import ort_compatible_om_session
    t0 = time.time()
    adapter, err = ort_compatible_om_session("models/detector/yolov8n.om", device_id=0)
    if adapter is None:
        print(f"[npu] OM 加载失败: {err}")
        return
    print(f"[npu] OM 加载 {round(time.time()-t0,2)}s")

    tensor = make_tensor(args.video)
    # 预热 5 帧
    for _ in range(5):
        adapter.run(None, {"images": tensor})

    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append(npu_smi())
            stop.wait(1.0)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    lats = []
    out_shape = None
    t_end = time.monotonic() + args.duration
    n = 0
    while time.monotonic() < t_end:
        ts = time.perf_counter()
        outs = adapter.run(None, {"images": tensor})
        lats.append((time.perf_counter() - ts) * 1000.0)
        if outs:
            out_shape = list(outs[0].shape)
        n += 1
    stop.set(); th.join(timeout=3)
    span = sum(lats) / 1000.0
    adapter.close()

    ai = [s["ai_core_pct"] for s in samples if s.get("ai_core_pct") is not None]
    pw = [s["power_w"] for s in samples if s.get("power_w") is not None]
    tp = [s["temp_c"] for s in samples if s.get("temp_c") is not None]
    result = {
        "meta": {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": "yolov8n.om",
                 "input": list(tensor.shape), "duration_s": args.duration,
                 "method": "独占 NPU 紧循环 adapter.run（隔离 CPU/系统争用）"},
        "npu_throughput_fps": round(n / span, 2) if span else None,
        "frames": n,
        "latency_ms": {"p50": pctl(lats, 50), "p95": pctl(lats, 95),
                       "max": pctl(lats, 100), "mean": round(sum(lats)/len(lats), 2)},
        "output_shape": out_shape,
        "resources": {
            "ai_core_pct_avg": round(sum(ai)/len(ai), 1) if ai else None,
            "ai_core_pct_max": max(ai) if ai else None,
            "power_w_avg": round(sum(pw)/len(pw), 2) if pw else None,
            "temp_c_max": max(tp) if tp else None,
        },
    }
    out = BASE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"[npu] 证据已落盘: {out}")


if __name__ == "__main__":
    main()