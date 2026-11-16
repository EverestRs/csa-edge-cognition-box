from __future__ import annotations

"""告警事件裁决器 (AlertAdjudicator)：告警触发后异步做"描述→裁决"两级复核。

数据流:
  InferenceEngine 告警触发 → submit(alert, frame_id) 记录待裁决事件
  → 引擎每帧调用 on_frame(frame_id)，等到收集完告警后 after_seconds 的缓冲帧
  → 单工作线程两趟 vLLM 调用（事件描述 → 成立性裁决）
  → 回写 results/event_{frame_id}.json 的 alert.final_decision / alert.adjudication
  → 裁决记录经 consume_pending_push() 附到下一次 FrameResult 推送前端

裁决只做标记（confirmed/rejected/uncertain），事件始终保留，不隐藏不删除；
vLLM 失败时 final_decision 保持 pending 并记录 last_error。
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from src.core.vlm_client import parse_json_loose, vlm_chat

LABEL_DESCRIPTIONS = {
    "H01": "打架斗殴（互殴）",
    "H02": "校园霸凌（单向攻击）",
    "H03": "跌倒/昏厥",
    "H04": "疑似轻生/极端危险行为",
    "H05": "破坏公共设施",
    "M01": "吸烟/点火",
    "M02": "长时间使用手机",
    "M03": "遮挡/干扰摄像头",
    "S01": "异常徘徊/滞留",
    "S02": "翻越围栏/闯入禁区",
}

DESCRIBE_PROMPT = (
    "你是校园安防告警复核员。以下是某次告警前后时段的监控画面（按时间顺序）。\n"
    "[场景上下文]\n{context}\n\n"
    "请客观描述这段时间画面中发生了什么（人物、动作、地点、先后过程），不超过120字。"
    "直接输出描述文字，不要任何前后缀。"
)

JUDGE_PROMPT = (
    "[系统角色] 你是校园安防告警审核员，请基于以下信息判断本次告警是否成立。\n"
    "[告警类型] {alert_type}（{label_desc}）\n"
    "{rule_line}"
    "[告警时段画面描述]\n{description}\n"
    "[场景上下文]\n{context}\n\n"
    "判定标准：\n"
    "- confirmed：画面或描述中确实存在与告警类型一致的异常行为\n"
    "- rejected：告警类型对应的异常并未发生（误报，例如正常活动被误判）\n"
    "- uncertain：信息不足以判断\n\n"
    '仅输出JSON（不要其他文字）：{{"decision": "confirmed/rejected/uncertain", "confidence": 0.85, "reasoning": "简要理由"}}'
)


class AlertAdjudicator:
    def __init__(self, vlm_config: dict, output_dir: str = "results"):
        cfg = vlm_config.get("alert_adjudicator", {}) if isinstance(vlm_config, dict) else {}
        self.enabled = bool(cfg.get("enabled", True))
        self.before_seconds = float(cfg.get("before_seconds", 5))
        self.after_seconds = float(cfg.get("after_seconds", 5))
        self.frames_per_call = int(max(2, cfg.get("frames_per_call", 6)))
        self.max_latency_ms = int(cfg.get("max_latency_ms", 30000))
        self.max_image_px = int(cfg.get("max_image_px", 512))
        self.backoff_seconds = float(cfg.get("failure_backoff_seconds", 30))
        # 待裁决队列上限：告警爆发期旧实现单槽直接丢弃（WEAKNESS_AUDIT #1），改为有界排队
        self.max_queue = int(max(1, cfg.get("max_pending_alerts", 4)))

        self.use_vllm = bool(vlm_config.get("use_vllm", False))
        self.vllm_url = str(vlm_config.get("vllm_url", "http://localhost:8001/v1/chat/completions"))
        self.vllm_api_key = str(vlm_config.get("vllm_api_key", "EMPTY"))
        self.model_name = str(vlm_config.get("model_name", "qwen2.5-vl-7b-instruct"))
        if not self.use_vllm:
            self.enabled = False
            self._last_error = "alert_adjudicator_requires_vllm_mode"

        self._source_fps = 25.0
        self._frame_buffer = None
        self._scene_narrator = None
        # 修复（WEAKNESS_AUDIT #1 关联）：on_frame 需按序号弹出到期项（pop(idx)），
        # deque.pop() 不接受索引——改用 list（线程安全由外层 _lock 保证）
        self._queue: list[dict[str, Any]] = []
        self._inflight = False
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alert-adjudicator")
        self._pending_push: dict[str, Any] | None = None
        self._last_error = ""
        self._adjudication_count = 0
        self._dropped_total = 0

        self._results_dir = Path(output_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    # ── 引擎接线 ─────────────────────────────────────────────

    def attach_buffer(self, frame_buffer) -> None:
        self._frame_buffer = frame_buffer

    def attach_vlm_router(self, vlm_router) -> None:
        """B3: 接入全局 VLM 并发槽位——槽满时保留 pending，下一帧重试。"""
        self._vlm_router = vlm_router

    def attach_narrator(self, scene_narrator) -> None:
        self._scene_narrator = scene_narrator

    def set_source_fps(self, fps: float) -> None:
        self._source_fps = float(fps if fps and fps > 1.0 else 25.0)

    def reset_for_task(self) -> None:
        """新任务开始：清除上个任务遗留的待裁决事件与推送（帧号已归零，旧等待作废）。"""
        with self._lock:
            self._queue.clear()
            self._pending_push = None
            self._dropped_total = 0

    # ── 主入口（推理线程调用，非阻塞） ────────────────────────

    def submit(self, alert: dict, frame_id: int, event_filename: str = "") -> bool:
        """提交告警进入裁决队列。队列满返回 False（引擎侧计数，不阻塞推理）。"""
        if not self.enabled:
            return False
        after_frames = max(1, int(round(self.after_seconds * self._source_fps)))
        with self._lock:
            if len(self._queue) >= self.max_queue:
                self._dropped_total += 1
                return False
            self._queue.append({
                "alert": dict(alert),
                "frame_id": int(frame_id),
                "due_frame_id": int(frame_id) + after_frames,
                "event_filename": str(event_filename),
            })
        return True

    def on_frame(self, frame_id: int) -> None:
        with self._lock:
            if self._inflight or self._frame_buffer is None or not self._queue:
                return
            idx = next((i for i, item in enumerate(self._queue) if frame_id >= item["due_frame_id"]), -1)
            if idx == -1:
                return
            # B3: 槽满时不消费队列，下一帧重试（裁决不丢失）
            router = getattr(self, "_vlm_router", None)
            if router is not None and not router.try_acquire(timeout=0.0):
                return
            pending = self._queue.pop(idx)
            before_frames = int(round(self.before_seconds * self._source_fps))
            start_id = max(1, pending["frame_id"] - before_frames)
            window = self._frame_buffer.window_between(start_id, frame_id)
            if len(window) < 2:
                if router is not None:
                    router.release()
                self._queue.append(pending)  # 缓冲未就绪，放回队尾稍后重试
                return
            self._inflight = True

        frames = [item[1] for item in window]
        frame_ids = [item[0] for item in window]
        self._executor.submit(self._adjudicate_job, frames, frame_ids, pending)

    # ── 工作线程 ─────────────────────────────────────────────

    def _even_sample(self, frames: list, frame_ids: list[int]) -> tuple[list, list[int]]:
        k = min(self.frames_per_call, len(frames))
        if len(frames) <= k:
            return frames, frame_ids
        indices = [int(round(i * (len(frames) - 1) / max(1, k - 1))) for i in range(k)]
        picked_f, picked_ids, seen = [], [], set()
        for idx in indices:
            if idx in seen:
                continue
            seen.add(idx)
            picked_f.append(frames[idx])
            picked_ids.append(frame_ids[idx])
        return picked_f, picked_ids

    def _adjudicate_job(self, frames: list[np.ndarray], frame_ids: list[int], pending: dict[str, Any]) -> None:
        start = time.perf_counter()
        alert = pending["alert"]
        frame_id = int(pending["frame_id"])
        try:
            context = self._scene_narrator.get_context() if self._scene_narrator is not None else ""
            call_frames, call_ids = self._even_sample(frames, frame_ids)

            # T3.5 后半（edge 事实化）：告警附带 facts_text 时第一趟不发图——
            # 结构化事实文本直接作为事件描述，省一次多模态调用且零幻觉面
            facts_text = str(alert.get("facts_text", "") or "").strip()
            if facts_text:
                description = (
                    f"{facts_text}\n[结构化事实] "
                    + json.dumps(alert.get("facts", {}), ensure_ascii=False)[:600]
                )
            else:
                description = vlm_chat(
                    base_url=self.vllm_url,
                    api_key=self.vllm_api_key,
                    model=self.model_name,
                    prompt=DESCRIBE_PROMPT.format(context=context or "（暂无场景上下文）"),
                    frames=call_frames,
                    max_tokens=300,
                    timeout=self.max_latency_ms / 1000.0,
                    max_image_px=self.max_image_px,
                )

            alert_type = str(alert.get("type", "unknown_detected"))
            label = alert_type.replace("_detected", "").upper()
            rule_text = str(alert.get("rule_text", "") or "")
            # NL 规则告警按用户原文裁决（"告警有人吸烟"），而不是泛化的类型名
            rule_line = f"[告警规则原文] {rule_text}\n" if rule_text else ""
            judge_text = vlm_chat(
                base_url=self.vllm_url,
                api_key=self.vllm_api_key,
                model=self.model_name,
                prompt=JUDGE_PROMPT.format(
                    alert_type=alert_type,
                    label_desc=LABEL_DESCRIPTIONS.get(label, label),
                    rule_line=rule_line,
                    description=description.strip()[:400],
                    context=context or "（暂无场景上下文）",
                ),
                frames=None,
                max_tokens=200,
                timeout=self.max_latency_ms / 1000.0,
            )
            parsed = parse_json_loose(judge_text) or {}
            decision = str(parsed.get("decision", "uncertain")).lower()
            if decision not in {"confirmed", "rejected", "uncertain"}:
                decision = "uncertain"

            adjudication = {
                "decision": decision,
                "confidence": float(max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))),
                "reasoning": str(parsed.get("reasoning", ""))[:300],
                "description": description.strip()[:300],
                "judged_frame_ids": call_ids,
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "judged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": self.model_name,
            }
            self._update_event_file(frame_id, adjudication, str(pending.get("event_filename", "")))

            push = {
                "type": "alert_adjudication",
                "frame_id": frame_id,
                "alert_type": alert_type,
                **adjudication,
            }
            with self._lock:
                self._pending_push = push
                self._adjudication_count += 1
            self._last_error = ""
        except Exception as exc:
            self._last_error = f"adjudicate_failed: {exc}"
            with self._lock:
                self._pending_push = {
                    "type": "alert_adjudication",
                    "frame_id": frame_id,
                    "alert_type": str(alert.get("type", "")),
                    "decision": "uncertain",
                    "confidence": 0.0,
                    "reasoning": f"裁决失败: {exc}",
                    "latency_ms": int((time.perf_counter() - start) * 1000),
                }
        finally:
            router = getattr(self, "_vlm_router", None)
            if router is not None:
                router.release()
            with self._lock:
                self._inflight = False

    def _update_event_file(self, frame_id: int, adjudication: dict[str, Any], event_filename: str = "") -> None:
        # 规则告警事件为 event_{fid}_rule_{rid}.json，主告警为 event_{fid}.json；
        # 旧实现固定写主告警文件名，规则告警的裁决结果会静默丢失
        event_path = self._results_dir / event_filename if event_filename else self._results_dir / f"event_{frame_id}.json"
        try:
            event = json.loads(event_path.read_text(encoding="utf-8"))
        except Exception:
            return
        alert_section = event.get("alert", {})
        alert_section["final_decision"] = adjudication["decision"]
        alert_section["adjudication"] = adjudication
        event["alert"] = alert_section
        try:
            event_path.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self._last_error = f"event_write_failed: {exc}"

    # ── 读取接口 ─────────────────────────────────────────────

    def consume_pending_push(self) -> dict[str, Any] | None:
        with self._lock:
            pending, self._pending_push = self._pending_push, None
        return pending

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "use_vllm": self.use_vllm,
                "healthy": bool(self.enabled and not self._last_error),
                "adjudication_count": self._adjudication_count,
                "inflight": self._inflight,
                "queue_length": len(self._queue),
                "max_queue": self.max_queue,
                "dropped_total": self._dropped_total,
                "last_error": self._last_error,
            }

    def stop(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
