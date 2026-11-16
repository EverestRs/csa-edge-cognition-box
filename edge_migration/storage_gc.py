from __future__ import annotations

"""results/ 存储治理（§4.5 / T5.x）：截图与事件轮转、scene_log 归档、磁盘水位告警。

边缘盒存储有限：告警截图与事件 JSON 无限增长会写满盘。策略：
- snapshots/：保留最新 N 张；
- event_*.json：保留最新 N 个；
- scene_log/*.jsonl：单文件超阈值归档为 .1（保留一份旧档）；
- 磁盘水位 ≥ warn_pct：追加清理最旧文件直至水位低于阈值（每次最多清 200 个，防误删风暴）；
- 全程只删 results/ 下本系统产物，任何异常静默降级（不阻塞推理）。
"""

import time
from pathlib import Path
from typing import Any

DEFAULTS = {
    "enabled": True,
    "interval_minutes": 10,
    "max_snapshots": 400,
    "max_events": 600,
    "scene_log_max_mb": 100,
    "disk_warn_pct": 80,
    "min_free_gb": 2.0,
}


def _newest_first(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _disk_used_pct(path: Path) -> float:
    try:
        total, _used, _free = __import__("shutil").disk_usage(str(path))
        return round(_used / total * 100.0, 1)
    except Exception:
        return 0.0


def _free_gb(path: Path) -> float:
    try:
        _total, _used, free = __import__("shutil").disk_usage(str(path))
        return round(free / 1e9, 2)
    except Exception:
        return 999.0


def run_gc(results_dir: str | Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULTS, **(config or {})}
    root = Path(results_dir)
    report: dict[str, Any] = {"ts": time.time(), "deleted": 0, "archived": 0, "actions": []}
    if not cfg.get("enabled", True) or not root.exists():
        report["skipped"] = True
        return report

    def _unlink(paths: list[Path], keep: int, tag: str) -> None:
        victims = _newest_first(paths)[keep:]
        for p in victims:
            try:
                p.unlink()
                report["deleted"] += 1
            except Exception:
                pass
        if victims:
            report["actions"].append(f"{tag}: -{len(victims)}")

    # 1) 截图与事件按数量保留
    _unlink(list((root / "snapshots").glob("*.jpg")), int(cfg["max_snapshots"]), "snapshots")
    _unlink(list(root.glob("event_*.json")), int(cfg["max_events"]), "events")

    # 2) scene_log 单文件超限归档（保留 .1 一份旧档）
    max_bytes = int(float(cfg["scene_log_max_mb"]) * 1e6)
    for f in (root / "scene_log").glob("*.jsonl"):
        try:
            if f.stat().st_size > max_bytes:
                archive = f.with_suffix(".jsonl.1")
                if archive.exists():
                    archive.unlink()
                f.rename(archive)
                report["archived"] += 1
                report["actions"].append(f"scene_log: archived {f.name}")
        except Exception:
            pass

    # 3) 磁盘水位：超阈值追加清理最旧截图/事件
    used_pct = _disk_used_pct(root)
    report["disk_used_pct"] = used_pct
    report["free_gb"] = _free_gb(root)
    if used_pct >= float(cfg["disk_warn_pct"]) or report["free_gb"] <= float(cfg["min_free_gb"]):
        snaps = list((root / "snapshots").glob("*.jpg"))
        events = list(root.glob("event_*.json"))
        for tag, pool in (("disk_pressure:snapshots", snaps), ("disk_pressure:events", events)):
            if report["free_gb"] > float(cfg["min_free_gb"]) and used_pct < float(cfg["disk_warn_pct"]):
                break
            victims = sorted(pool, key=lambda p: p.stat().st_mtime)[:200]
            for p in victims:
                try:
                    p.unlink()
                    report["deleted"] += 1
                except Exception:
                    pass
            if victims:
                report["actions"].append(f"{tag}: -{len(victims)}")
            used_pct = _disk_used_pct(root)
            report["disk_used_pct"] = used_pct
            report["free_gb"] = _free_gb(root)
    report["ok"] = True
    return report


class StorageGcLoop:
    """引擎后台线程：定时执行 run_gc（缺省 10 分钟一次，启动时先跑一遍）。"""

    def __init__(self, results_dir: str, config: dict[str, Any] | None = None):
        cfg = {**DEFAULTS, **(config or {})}
        self.results_dir = str(results_dir)
        self.interval_s = float(cfg.get("interval_minutes", 10)) * 60.0
        self.gc_cfg = cfg
        self.last_report: dict[str, Any] = {}
        self._stop = False
        import threading
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="storage-gc")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def _loop(self) -> None:
        try:
            self.last_report = run_gc(self.results_dir, self.gc_cfg)
        except Exception:
            pass
        while not self._stop:
            time.sleep(min(self.interval_s, 60.0))
            if self._stop:
                break
            try:
                self.last_report = run_gc(self.results_dir, self.gc_cfg)
            except Exception:
                pass
