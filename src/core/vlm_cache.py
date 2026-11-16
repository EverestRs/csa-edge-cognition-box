from __future__ import annotations

"""VLM person crop 结果缓存：dHash 指纹 + LRU + TTL。

静态场景中同一 person crop 被重复送入 VLM——缓存命中直接返回上次结果，
减少 30-50% 的 VLM 调用。dHash 对轻微画面变化鲁棒（光照微变不影响指纹）。
"""

import hashlib
import time
from collections import OrderedDict

import cv2
import numpy as np


class VLMCropCache:
    def __init__(self, max_size: int = 100, ttl_seconds: float = 30.0):
        self._cache: OrderedDict[str, tuple[str, float, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _dhash(crop: np.ndarray) -> str:
        """差值哈希：缩放到 9×8 灰度图，比较相邻像素亮度。"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        return hashlib.md5(diff.tobytes()).hexdigest()

    def get(self, crop: np.ndarray) -> tuple[str, float] | None:
        key = self._dhash(crop)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        label, conf, ts = entry
        if time.time() - ts > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return (label, conf)

    def put(self, crop: np.ndarray, label: str, conf: float) -> None:
        key = self._dhash(crop)
        self._cache[key] = (label, conf, time.time())
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """任务隔离：新任务开始时清空缓存，上一任务的 crop 结果不得复用。"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, total), 3),
        }
