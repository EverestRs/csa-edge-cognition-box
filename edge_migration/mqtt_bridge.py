from __future__ import annotations

"""MQTT 桥：状态/事件上行、规则下行（端云联动，01 §2.6）。

- paho-mqtt 缺失或 broker 不可达 → dry_run 模式：publish 记入 last_debug 与缓冲，
  绝不抛异常、不阻塞推理线程（对齐项目"模块内捕获异常"约定）；
- event 上行断网缓冲 100 条，重连 flush；status 仅保留最新（QoS0 实时值）；
- 下行 csa/{box}/rule/set → 注册的回调（对接 NLRuleEngine CRUD / update_zones）。
"""

import json
import threading
import time
from collections import deque
from typing import Any, Callable

try:
    import paho.mqtt.client as mqtt  # type: ignore
    _PAHO = True
except Exception:
    mqtt = None
    _PAHO = False

STATUS_INTERVAL_S = 1.0


class MqttBridge:
    def __init__(self, config: dict[str, Any] | None = None,
                 on_rule_set: Callable[[dict[str, Any]], None] | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.host = str(cfg.get("host", "127.0.0.1"))
        self.port = int(cfg.get("port", 1883))
        self.box_id = str(cfg.get("box_id", "edge1"))
        self.prefix = str(cfg.get("topic_prefix", "csa"))
        self.user = cfg.get("user") or None
        self.password = cfg.get("password") or None
        self.buffer_size = int(cfg.get("buffer_size", 100))
        self._on_rule_set = on_rule_set

        self.topic_status = f"{self.prefix}/{self.box_id}/status"
        self.topic_event = f"{self.prefix}/{self.box_id}/event"
        self.topic_rule_set = f"{self.prefix}/{self.box_id}/rule/set"
        self.topic_rule_ack = f"{self.prefix}/{self.box_id}/rule/ack"

        self._client = None
        self._connected = False
        self._event_buffer: deque[str] = deque(maxlen=self.buffer_size)
        self._status_provider: Callable[[], dict[str, Any]] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.last_debug: dict[str, Any] = {"dry_run": not _PAHO or not self.enabled}

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return _PAHO and self.enabled

    def register_status_provider(self, fn: Callable[[], dict[str, Any]]) -> None:
        self._status_provider = fn

    def set_rule_handler(self, fn: Callable[[dict[str, Any]], None]) -> None:
        self._on_rule_set = fn

    def start(self) -> None:
        if not self.enabled:
            self.last_debug["skipped"] = "disabled"
            return
        if not _PAHO:
            self.last_debug["skipped"] = "paho_mqtt_missing: pip install paho-mqtt"
            return
        try:
            self._client = mqtt.Client(client_id=f"csa-{self.box_id}",
                                       clean_session=True)
            if self.user:
                self._client.username_pw_set(self.user, self.password)
            self._client.will_set(self.topic_status, json.dumps(
                {"online": False, "box_id": self.box_id}, ensure_ascii=False),
                qos=1, retain=True)
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message
            self._client.connect_async(self.host, self.port, keepalive=30)
            self._client.loop_start()
            self._thread = threading.Thread(target=self._status_loop, daemon=True)
            self._thread.start()
            self.last_debug["started"] = True
        except Exception as exc:
            self.last_debug["start_error"] = str(exc)
            self._client = None

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._client is not None:
                self._client.loop_stop()
                self._client.disconnect()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _on_connect(self, _client, _userdata, _flags, rc, *_args) -> None:
        self._connected = (rc == 0)
        if self._connected:
            try:
                self._client.subscribe(self.topic_rule_set, qos=1)
            except Exception:
                pass
            while self._event_buffer:
                self._raw_publish(self.topic_event, self._event_buffer.popleft(), qos=1)

    def _on_disconnect(self, _client, _userdata, _rc, *_args) -> None:
        self._connected = False

    def _on_message(self, _client, _userdata, msg) -> None:
        try:
            payload = json.loads(str(msg.payload.decode("utf-8")))
            if self._on_rule_set is not None:
                self._on_rule_set(payload)
                self.publish_rule_ack({"rule_id": payload.get("rule_id"),
                                       "applied": True, "ts": time.time()})
        except Exception as exc:
            self.last_debug["rule_set_error"] = str(exc)

    # ------------------------------------------------------------------ #
    def _raw_publish(self, topic: str, payload_json: str, *, qos: int = 0,
                     retain: bool = False) -> None:
        if self._client is not None and self._connected:
            self._client.publish(topic, payload_json, qos=qos, retain=retain)
        else:
            self.last_debug["dry_run_last"] = {"topic": topic, "bytes": len(payload_json)}

    def publish_status(self, payload: dict[str, Any]) -> None:
        """status 上行（含 npu-smi 合并）；仅最新值语义，不缓冲。"""
        try:
            body = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception as exc:
            self.last_debug["status_json_error"] = str(exc)
            return
        if self._client is not None and self._connected:
            self._raw_publish(self.topic_status, body, qos=0, retain=True)
        else:
            self.last_debug["dry_run_last_status"] = body[:200]

    def publish_event(self, event: dict[str, Any]) -> None:
        try:
            body = json.dumps(event, ensure_ascii=False, default=str)
        except Exception as exc:
            self.last_debug["event_json_error"] = str(exc)
            return
        if self._client is not None and self._connected:
            self._raw_publish(self.topic_event, body, qos=1)
        else:
            self._event_buffer.append(body)

    def publish_rule_ack(self, ack: dict[str, Any]) -> None:
        try:
            self._raw_publish(self.topic_rule_ack,
                              json.dumps(ack, ensure_ascii=False), qos=1)
        except Exception:
            pass

    def publish_rule_set(self, rule: dict[str, Any]) -> None:
        """测试/看板侧下行入口（盒上自检用）。"""
        try:
            self._raw_publish(self.topic_rule_set,
                              json.dumps(rule, ensure_ascii=False), qos=1, retain=True)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _status_loop(self) -> None:
        while not self._stop.wait(STATUS_INTERVAL_S):
            if self._status_provider is None:
                continue
            try:
                self.publish_status(self._status_provider())
            except Exception as exc:
                self.last_debug["status_provider_error"] = str(exc)
