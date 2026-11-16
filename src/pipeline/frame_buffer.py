from __future__ import annotations

"""环形帧缓冲：以降采样帧保存最近 N 秒画面。

供场景叙述器（按间隔采样关键帧）与告警事件裁决（取告警前后窗口）共用，
只存缩小后的帧引用，避免全分辨率长缓冲的内存开销。
"""

import threading
from collections import deque

import cv2
import numpy as np


class FrameRingBuffer:
    def __init__(self, max_seconds: float = 30.0, samples_per_second: float = 2.0, target_width: int = 480,
                 max_frames: int | None = None):
        self.max_seconds = float(max(1.0, max_seconds))
        self.samples_per_second = float(max(0.1, samples_per_second))
        # target_width<=0 表示直存原始分辨率（复核用原帧通道）
        self.target_width = int(max(160, target_width)) if target_width and target_width > 0 else 0
        self._lock = threading.Lock()
        self._items: deque[tuple[int, np.ndarray]] = deque()
        self._source_fps = 25.0
        # 每个处理帧都入队（与处理步长解耦，避免步长互质导致长期无入队）。
        # 窗口淘汰按 max_seconds 视频时间执行（push 内），与处理帧率无关；
        # _maxlen 仅作内存硬护栏——原实现按"处理≤10fps"隐含假设设 maxlen，
        # 高处理帧率下缓冲覆盖的视频时长不足（WEAKNESS_AUDIT #7）。
        # max_frames 供高分辨率缓冲独立收口（WEAKNESS_AUDIT #9）。
        self._maxlen = int(max(1, max_frames)) if max_frames else int(self.max_seconds * 10)

    def set_source_fps(self, fps: float) -> None:
        fps = float(fps if fps and fps > 1.0 else 25.0)
        with self._lock:
            self._source_fps = fps

    def push(self, frame: np.ndarray, frame_id: int) -> None:
        if frame_id <= 0:
            return
        h, w = frame.shape[:2]
        if self.target_width and w > self.target_width:
            scale = self.target_width / float(w)
            frame = cv2.resize(frame, (self.target_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        with self._lock:
            self._items.append((int(frame_id), frame))
            # 时间窗淘汰：按视频时间只保留最近 max_seconds 秒，处理帧率变化不影响覆盖窗口
            oldest_ok = int(frame_id) / self._source_fps - self.max_seconds
            while self._items and (self._items[0][0] / self._source_fps) < oldest_ok:
                self._items.popleft()
            while len(self._items) > self._maxlen:
                self._items.popleft()

    def recent_entries(self, k: int) -> list[tuple[int, np.ndarray]]:
        """最近 k 个缓冲帧（按帧号升序），供复核主动选帧等按帧访问。"""
        with self._lock:
            items = list(self._items)
        items.sort(key=lambda t: t[0])
        return items[-max(1, int(k)):]

    def sample(self, k: int, window_seconds: float | None = None) -> list[tuple[int, np.ndarray]]:
        """均匀取 k 帧（时间升序）。window_seconds 限定只看最近 N 秒。"""
        with self._lock:
            items = list(self._items)
        if not items:
            return []
        if window_seconds is not None:
            min_frame_id = items[-1][0] - int(window_seconds * self._source_fps)
            items = [item for item in items if item[0] >= min_frame_id] or items[-1:]
        if len(items) <= k:
            return items
        indices = [int(round(i * (len(items) - 1) / max(1, k - 1))) for i in range(k)]
        picked: list[tuple[int, np.ndarray]] = []
        seen: set[int] = set()
        for idx in indices:
            if idx in seen:
                continue
            seen.add(idx)
            picked.append(items[idx])
        return picked

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def window_between(self, start_frame_id: int, end_frame_id: int) -> list[tuple[int, np.ndarray]]:
        """取 [start_frame_id, end_frame_id] 区间内的缓冲帧（时间升序）。"""
        with self._lock:
            items = list(self._items)
        return [item for item in items if start_frame_id <= item[0] <= end_frame_id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
