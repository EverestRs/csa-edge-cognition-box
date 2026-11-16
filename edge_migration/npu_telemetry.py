from __future__ import annotations

"""昇腾板级遥测：npu-smi 功耗/温度/占用采集（底牌角标数据源，01 §2.7）。

- x86 开发机：available()=False，read() 返回 None（上层显示 mock 或隐藏该列）；
- 解析策略防御式：npu-smi 输出随固件版本变化，逐字段容错，绝不抛异常；
- 带 TTL 缓存：npu-smi 子进程开销 0.3–0.6s 级，2s 缓存 + 单次调用足够角标刷新。
"""

import re
import shutil
import subprocess
import threading
import time
from typing import Any

_CACHE_TTL_S = 2.0
_cache: dict[str, Any] = {"ts": 0.0, "data": None}
_lock = threading.Lock()


def available() -> bool:
    return shutil.which("npu-smi") is not None


def mock_board_stats() -> dict[str, Any]:
    """开发机/演示彩排用的合成数据（在看板上显式标注 mock，不作证据）。"""
    return {"power_w": 6.4, "current_ma": 533, "voltage_mv": 12005,
            "temp_c": 52, "health": "OK", "ai_core_util_pct": 41.0, "mock": True,
            "ts": time.time()}


def _run(args: list[str], timeout: float = 2.0) -> str:
    try:
        r = subprocess.run(["npu-smi", *args], capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout or ""
    except Exception:
        return ""


def _parse_board(text: str) -> dict[str, Any]:
    """解析 `npu-smi info -t board -i <dev>` 的功耗行（Health/Power/Current/Voltage）。"""
    out: dict[str, Any] = {}
    m = re.search(r"OK", text)
    out["health"] = "OK" if m else ("NA" if not text.strip() else "WARN")
    m = re.search(r"\|\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", text)
    if m:  # Power(W) | Current(mA) | Voltage(mV)
        out["power_w"] = float(m.group(1))
        out["current_ma"] = int(m.group(2))
        out["voltage_mv"] = int(m.group(3))
    m = re.search(r"(\d{2,3}(?:\.\d)?)\s*$", text, re.M)  # 温度行固件差异大，兜底不强求
    temp = re.search(r"[Tt]emp.*?(\d{2,3})", text)
    if temp:
        out["temp_c"] = int(temp.group(1))
    return out


def _parse_info_table(text: str) -> dict[str, Any]:
    """解析裸 `npu-smi info` 汇总表。

    盒上适配: 310B4 固件的 `info -t board -i 0` 只回 NA，不含 Power/Current/Voltage，
    功耗与温度只在总表里；原先只走 board 接口 → power_w 恒为 null。
    """
    out: dict[str, Any] = {}
    m = re.search(r"\|\s*\d+\s+\S+\s*\|\s*(\S+)\s*\|\s*(\d+(?:\.\d+)?)\s+(\d+)\s+", text)
    if m:  # Health | Power(W) | Temp(C)
        out["health"] = "OK" if m.group(1).upper() in ("OK", "HEALTHY") else "WARN"
        out["power_w"] = float(m.group(2))
        out["temp_c"] = int(m.group(3))
    m = re.search(r"\|\s*\d+\s+\d+\s*\|\s*\S+\s*\|\s*(\d+(?:\.\d+)?)\s+\d+\s*/\s*\d+\s*", text)
    if m:  # AICore(%) | Memory-Usage(MB)
        out["ai_core_util_pct"] = float(m.group(1))
    return out


def _parse_usage(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        out["ai_core_util_pct"] = float(m.group(1))
    return out


def read(device: int = 0, *, ttl: float = _CACHE_TTL_S) -> dict[str, Any] | None:
    """板级状态聚合读；x86/失败返回 None。

    盒上适配: npu-smi 单次调用 ~0.3–0.6s，1s 轮询下多接口并发（角标 + MQTT status + 看板）
    会把它打成瓶颈（/api/v1/metrics 曾达 6–11s）。改成：
      1) 优先只调一次裸 `npu-smi info`（总表已含功耗/温度/AICore）；
      2) 仅当总表解析不到功耗时，才回退 board/usages；
      3) 加锁 + 2s TTL，避免并发风暴。
    """
    now = time.time()
    if now - _cache["ts"] < ttl and _cache["data"] is not None:
        return _cache["data"]  # type: ignore[return-value]
    if not available():
        return None
    with _lock:
        # 双检：等锁期间可能已有其它线程刷新
        now = time.time()
        if now - _cache["ts"] < ttl and _cache["data"] is not None:
            return _cache["data"]  # type: ignore[return-value]
        data: dict[str, Any] = {"ts": now}
        try:
            data.update({k: v for k, v in _parse_info_table(_run(["info"])).items()
                         if v is not None})
            if "power_w" not in data:
                board = _parse_board(_run(["info", "-t", "board", "-i", str(device)]))
                usage = _parse_usage(_run(["info", "-t", "usages", "-i", str(device)]))
                data.update({k: v for k, v in {**board, **usage}.items() if v is not None})
        except Exception:
            return None
        if len(data) <= 1:
            return None
        _cache["ts"], _cache["data"] = now, data
        return data


def read_or_mock(*, allow_mock: bool = False) -> dict[str, Any]:
    """看板/角标统一入口：真机数据优先；演示彩排允许 mock。"""
    data = read()
    if data is not None:
        return {**data, "mock": False}
    if allow_mock:
        return mock_board_stats()
    return {"mock": False, "available": False, "ts": time.time()}
