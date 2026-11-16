#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三档功耗/温度/AICore 实测采样（空闲 / 推理 / 推理+LLM），各 60s、1Hz。

数据源：npu-smi info（板级功耗、温度、AICore）+ /proc/meminfo（可用内存）。
输出：results/edge_test/power_three_modes_<date>.json + PNG 折线图。
用法: ~/.conda/envs/csa/bin/python tools/power_three_modes.py [--seconds 60]
"""
import argparse
import json
import re
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "results" / "edge_test"
SUB_DIR = BASE.parent / "SUBMISSION"
API = "http://127.0.0.1:8000"
LLM = "http://127.0.0.1:8001/v1/chat/completions"
STREAM = "rtsp://127.0.0.1:8554/live"
FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
FONT_LAT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONTS = {}


def _font(path: str, size: int):
    key = (path, size)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(path, size)
    return _FONTS[key]


def text_mixed(d: ImageDraw.ImageDraw, xy, s: str, size: int, fill: str) -> None:
    """CJK 字形用 Droid，拉丁/数字用 DejaVu（Droid 缺拉丁字形会显示成方框）。"""
    x, y = xy
    for ch in str(s):
        cjk = ("\u4e00" <= ch <= "\u9fff") or ch in "，。：；（）、《》—…·！？【】“”‘’"
        f = _font(FONT if cjk else FONT_LAT, size)
        d.text((x, y), ch, font=f, fill=fill)
        x += d.textlength(ch, font=f)


def read_npu() -> dict:
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {}
    res = {}
    m = re.search(r"310B\d?\s*\|\s*\w+\s*\|\s*([\d.]+)\s+(\d+)", out)
    if m:
        res["power_w"] = float(m.group(1))
        res["temp_c"] = int(m.group(2))
    d = re.search(r"\|\s*0\s+0\s*\|\s*NA\s*\|\s*(\d+)\s+(\d+)\s*/\s*(\d+)", out)
    if d:
        res["ai_core_pct"] = int(d.group(1))
        res["npu_mem_used_mb"] = int(d.group(2))
        res["npu_mem_total_mb"] = int(d.group(3))
    return res


def read_mem_avail_mb() -> int:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def cpu_times():
    try:
        parts = open("/proc/stat").readline().split()[1:]
        vals = [int(x) for x in parts]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return sum(vals), idle
    except Exception:
        return 0, 0


def sample(seconds: int, tag: str) -> dict:
    rows = []
    t0 = time.time()
    pre_t, pre_i = cpu_times()
    while time.time() - t0 < seconds:
        ts = time.time() - t0
        row = {"t": round(ts, 2), "mem_avail_mb": read_mem_avail_mb()}
        row.update(read_npu())
        rows.append(row)
        time.sleep(1.0)
    post_t, post_i = cpu_times()
    busy = 100.0 * (1 - (post_i - pre_i) / max(1, post_t - pre_t))
    vals = lambda k: [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    pw, tp, ai = vals("power_w"), vals("temp_c"), vals("ai_core_pct")
    mean = lambda a: round(sum(a) / len(a), 2) if a else None
    return {
        "mode": tag, "seconds": seconds, "samples": rows,
        "summary": {
            "power_mean_w": mean(pw), "power_max_w": max(pw) if pw else None,
            "temp_mean_c": mean(tp), "temp_max_c": max(tp) if tp else None,
            "ai_core_mean_pct": mean(ai), "ai_core_max_pct": max(ai) if ai else None,
            "cpu_busy_mean_pct": round(busy, 1),
            "mem_avail_mean_mb": mean(vals("mem_avail_mb")),
        },
    }


def llm_load(stop: threading.Event, worker: int) -> None:
    """持续请求 LLM，制造『推理+LLM』并发负载（多 worker）。"""
    prompt = "只输出一个词：安全"
    while not stop.is_set():
        try:
            requests.post(LLM, json={"model": "qwen", "messages": [{"role": "user", "content": prompt}],
                                     "max_tokens": 24}, timeout=30)
        except Exception:
            pass
        stop.wait(0.5)


def draw_chart(result: dict, png: Path) -> None:
    W, H = 1400, 1040
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(FONT, 34)
    f_lab = ImageFont.truetype(FONT, 20)
    f_small = ImageFont.truetype(FONT, 17)
    d.text((40, 24), "", font=f_title, fill="white")
    text_mixed(d, (40, 24), "三档实测：板级功耗 / NPU 温度 / AICore 占用（各 60s，1Hz）", 34, "#1f2937")

    modes = result["modes"]
    colors = {"idle": "#6b7280", "infer": "#2563eb", "infer_llm": "#ea580c"}
    names = {"idle": "空闲", "infer": "推理", "infer_llm": "推理+LLM"}
    metrics = [("power_w", "板级功耗 (W)", 0, None), ("temp_c", "NPU 温度 (℃)", 0, None),
               ("ai_core_pct", "AICore 占用 (%)", 0, None)]
    top, ph, gap, left, right = 100, 240, 55, 110, 60
    for mi, (key, label, lo, hi) in enumerate(metrics):
        y0 = top + mi * (ph + gap)
        d.rectangle([left, y0, W - right, y0 + ph], outline="#d1d5db")
        allv = []
        for m in modes:
            allv += [r[key] for r in m["samples"] if isinstance(r.get(key), (int, float))]
        vmin = min(allv) if allv else 0
        vmax = max(allv) if allv else 1
        if hi is not None:
            vmin, vmax = lo, hi
        pad = max(1.0, (vmax - vmin) * 0.15)
        vmin, vmax = vmin - pad, vmax + pad
        if lo is not None:
            vmin = max(lo, vmin)  # 百分比/功耗不出现负刻度
        text_mixed(d, (left, y0 - 30), label, 20, "#111827")
        for k in range(4):
            gy = y0 + ph * k / 3
            d.line([left, gy, W - right, gy], fill="#f3f4f6")
            val = vmax - (vmax - vmin) * k / 3
            text_mixed(d, (18, gy - 9), f"{val:.1f}", 17, "#6b7280")
        n = max(1, len(modes[0]["samples"]) - 1)
        for m in modes:
            pts = []
            for r in m["samples"]:
                if not isinstance(r.get(key), (int, float)):
                    continue
                x = left + (W - left - right) * min(1.0, r["t"] / max(1, m["seconds"] - 1))
                y = y0 + ph - (r[key] - vmin) / max(1e-6, vmax - vmin) * ph
                pts.append((x, y))
            if len(pts) > 1:
                d.line(pts, fill=colors[m["mode"]], width=3)
        suffix = {"power_w": "power_mean_w", "temp_c": "temp_mean_c", "ai_core_pct": "ai_core_mean_pct"}[key]
        text_mixed(d, (left + 10, y0 + 8), f"空闲 {modes[0]['summary'].get(suffix)}", 17, colors["idle"])
        text_mixed(d, (left + 10, y0 + 30), f"推理 {modes[1]['summary'].get(suffix)}", 17, colors["infer"])
        text_mixed(d, (left + 10, y0 + 52), f"推理+LLM {modes[2]['summary'].get(suffix)}", 17, colors["infer_llm"])

    text_mixed(d, (left, H - 40), "数据源：npu-smi info（1Hz）+ /proc/meminfo", 17, "#6b7280")
    img.save(png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=60)
    args = ap.parse_args()
    date = datetime.now().strftime("%Y%m%d")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    print("[power] 档位1/3：空闲", flush=True)
    idle = sample(args.seconds, "idle")

    task_id = ""
    try:
        r = requests.post(f"{API}/api/v1/analyze/stream", json={"stream_url": STREAM, "duration": 0}, timeout=10)
        task_id = r.json().get("task_id", "")
        print(f"[power] 任务已起 {task_id}，预热 10s", flush=True)
        time.sleep(10)
    except Exception as exc:
        print(f"[power] 任务启动失败: {exc}", flush=True)

    print("[power] 档位2/3：推理", flush=True)
    infer = sample(args.seconds, "infer")
    try:
        infer["task_stats"] = requests.get(f"{API}/api/v1/tasks/{task_id}", timeout=5).json().get("stream_stats", {})
    except Exception:
        infer["task_stats"] = {}

    print("[power] 档位3/3：推理+LLM", flush=True)
    stop = threading.Event()
    ths = [threading.Thread(target=llm_load, args=(stop, i), daemon=True) for i in range(3)]
    for th in ths:
        th.start()
    infer_llm = sample(args.seconds, "infer_llm")
    stop.set()
    try:
        infer_llm["task_stats"] = requests.get(f"{API}/api/v1/tasks/{task_id}", timeout=5).json().get("stream_stats", {})
    except Exception:
        infer_llm["task_stats"] = {}

    if task_id:
        try:
            requests.post(f"{API}/api/v1/tasks/{task_id}/stop", timeout=5)
        except Exception:
            pass

    result = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "source": "npu-smi info 1Hz", "modes": [idle, infer, infer_llm]}
    jpath = OUT_DIR / f"power_three_modes_{date}.json"
    jpath.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    ppath = SUB_DIR / f"power_three_modes_{date}.png"
    draw_chart(result, ppath)
    print(json.dumps({m["mode"]: m["summary"] for m in result["modes"]}, ensure_ascii=False, indent=1))
    print(f"[power] JSON={jpath}\n[power] PNG={ppath}")


if __name__ == "__main__":
    main()
