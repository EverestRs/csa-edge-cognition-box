from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

from fastapi import WebSocket


@dataclass
class _ClientConnection:
    websocket: WebSocket
    queue: asyncio.Queue
    sender_task: asyncio.Task


class ConnectionManager:
    def __init__(self, history_size: int = 24):
        self.connections: dict[str, list[_ClientConnection]] = {}
        self.history_size = int(max(0, history_size))
        self.history: dict[str, deque[dict]] = {}
        self.task_history_sizes: dict[str, int] = {}

    def set_history_size(self, history_size: int) -> None:
        self.history_size = int(max(0, history_size))

    def prepare_task(self, task_id: str, history_size: int | None = None) -> None:
        size = self.history_size if history_size is None else int(max(0, history_size))
        self.task_history_sizes[task_id] = size
        self.history[task_id] = deque(maxlen=size)

    def get_task_history_size(self, task_id: str) -> int:
        return int(self.task_history_sizes.get(task_id, self.history_size))

    def record_message(self, task_id: str, message: dict) -> None:
        task_history_size = self.get_task_history_size(task_id)
        if task_history_size <= 0:
            return
        buffer = self.history.get(task_id)
        if buffer is None or buffer.maxlen != task_history_size:
            previous = list(buffer) if buffer else []
            buffer = deque(previous, maxlen=task_history_size)
            self.history[task_id] = buffer
        buffer.append(message)

    def clear_task(self, task_id: str) -> None:
        self.connections.pop(task_id, None)
        self.history.pop(task_id, None)
        self.task_history_sizes.pop(task_id, None)

    async def _sender_loop(self, task_id: str, connection: _ClientConnection):
        websocket = connection.websocket
        while True:
            message = await connection.queue.get()
            await websocket.send_json(message)

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()

        queue_size = max(8, self.get_task_history_size(task_id) * 2)
        queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        placeholder_task = asyncio.create_task(asyncio.sleep(0))
        connection = _ClientConnection(websocket=websocket, queue=queue, sender_task=placeholder_task)
        sender_task = asyncio.create_task(self._sender_loop(task_id, connection))
        connection.sender_task = sender_task
        self.connections.setdefault(task_id, []).append(connection)

        for cached_message in list(self.history.get(task_id, [])):
            if queue.full():
                break
            queue.put_nowait(cached_message)

    def disconnect(self, task_id: str, websocket: WebSocket):
        connections = self.connections.get(task_id, [])
        for connection in list(connections):
            if connection.websocket is websocket:
                if not connection.sender_task.done():
                    connection.sender_task.cancel()
                connections.remove(connection)
        if task_id in self.connections and not self.connections[task_id]:
            self.connections.pop(task_id, None)

    async def broadcast(self, task_id: str, message: dict):
        connections = list(self.connections.get(task_id, []))
        if not connections:
            return

        dead_connections: list[_ClientConnection] = []
        for connection in connections:
            if connection.sender_task.done():
                dead_connections.append(connection)
                continue

            queue = connection.queue
            if queue.full():
                try:
                    queue.get_nowait()
                except Exception:
                    pass

            try:
                queue.put_nowait(message)
            except Exception:
                dead_connections.append(connection)

        for connection in dead_connections:
            self.disconnect(task_id, connection.websocket)
