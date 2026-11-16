from __future__ import annotations

"""VLM 路由器：全局并发槽位（B3）。

并发槽位背景：场景叙述器、告警裁决器、NL 规则引擎、引擎异步复核四个调用方
各自有独立工作线程，vLLM continuous batching 天然支持并发——但并发无上限时
多路大图请求会挤爆 KV cache 引发 preempt-recompute。本路由器用信号量把全局
在途请求数限制在 max_concurrent_requests（默认 4），各调用方在发出 vLLM
请求前取槽位，完成后归还。

取槽位语义：
  - try_acquire(timeout=0)：非阻塞，失败返回 False（周期性旁路用于跳过本轮）
  - try_acquire(timeout>0)：阻塞等待至多 timeout 秒（复核类调用用于排队）

历史注记：早期版本另有基于 p95 延迟的降级档位（level 0/1/2 + allow_review），
但引擎侧长期以硬编码 latency 调用导致档位恒为 0、从不生效（WEAKNESS_AUDIT #10），
已移除——VLM 过载保护由并发槽位 + 各调用方退避/单飞承担，复核类调用一律允许提交，
由槽位排队与超时兜底。
"""

import threading
import time


class VLMRouter:
    def __init__(self, config: dict):
        cfg = config.get("vlm_router", {}) if isinstance(config, dict) else {}
        self.max_concurrent = max(1, int(cfg.get("max_concurrent_requests", 4)))
        self._slots = threading.BoundedSemaphore(self.max_concurrent)
        self._stats_lock = threading.Lock()
        self._inflight = 0
        self._peak_inflight = 0
        self._total_acquired = 0
        self._total_rejected = 0
        self._total_waited_ms = 0.0

    # ── 并发槽位（B3） ────────────────────────────────────────

    def try_acquire(self, timeout: float = 0.0) -> bool:
        """取一个全局 VLM 并发槽位。timeout=0 非阻塞；>0 时最多等待 timeout 秒。"""
        wait_t0 = time.perf_counter()
        acquired = self._slots.acquire(blocking=timeout > 0, timeout=timeout if timeout > 0 else None)
        waited_ms = (time.perf_counter() - wait_t0) * 1000.0
        with self._stats_lock:
            if acquired:
                self._inflight += 1
                self._peak_inflight = max(self._peak_inflight, self._inflight)
                self._total_acquired += 1
                if waited_ms > 1.0:
                    self._total_waited_ms += waited_ms
            else:
                self._total_rejected += 1
        return acquired

    def release(self) -> None:
        with self._stats_lock:
            self._inflight = max(0, self._inflight - 1)
        try:
            self._slots.release()
        except ValueError:
            pass

    def stats(self) -> dict:
        with self._stats_lock:
            return {
                "max_concurrent": self.max_concurrent,
                "inflight": self._inflight,
                "peak_inflight": self._peak_inflight,
                "total_acquired": self._total_acquired,
                "total_rejected": self._total_rejected,
                "total_waited_ms": round(self._total_waited_ms, 1),
            }
