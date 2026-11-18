from __future__ import annotations

"""场景叙述器 (SceneNarrator)：周期性用本地 Qwen2.5-VL 对整个场景生成中文叙述。

数据流:
  InferenceEngine.process() 每帧调用 on_frame()
    → 每隔 narrate_interval_seconds 视频时间，从 FrameRingBuffer 均匀取 K 帧
    → 单工作线程调用 vLLM（单飞锁，进行中则跳过本轮）
    → 解析 JSON → 追加 scene_log.jsonl → 更新滚动摘要（低频纯文本调用）
    → 记录挂到 consume_pending_push()，由引擎附带在下一次 FrameResult 推送

设计约束:
  - 永不阻塞推理流水线：所有 vLLM 调用在独立线程
  - 失败降级：异常后指数退避暂停，不重试堆积
  - 原始 JSONL 是真值来源，滚动摘要仅用于展示与上下文注入
"""

import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from src.core.vlm_client import parse_json_loose, vlm_chat

DEFAULT_NARRATE_PROMPT = (
    "你是校园监控系统的场景分析员。以下是按时间顺序排列的一组监控画面关键帧。"
    "请分析这组画面，输出JSON（不要包含任何其他文字）：\n"
    '{"scene_summary": "一段话客观描述场景中的人员与正在发生的行为", '
    '"notable_activities": ["值得注意或异常的活动，逐条列出，无则空数组"], '
    '"environment": "场景环境简述（室内/室外、光照、区域类型）"}'
)

# 行为标签 → 中文描述（检测事实注入用，口径与 VLM 判定 prompt 一致）
LABEL_NAMES = {
    "H01": "打架斗殴", "H02": "校园霸凌", "H03": "跌倒/昏厥", "H04": "疑似轻生",
    "H05": "破坏设施", "M01": "吸烟", "M02": "使用手机", "M03": "遮挡摄像头",
    "S01": "异常徘徊", "S02": "翻越围栏/闯入禁区", "Normal": "正常",
}

DEFAULT_SUMMARY_PROMPT = (
    "[系统角色] 你将接收到一系列按时间顺序排列的校园监控场景描述。"
    "请整合为一段不超过150字的连贯摘要，保留人员数量变化、所在区域与重要事件。\n"
    "[历史描述]\n{entries}\n\n直接输出摘要内容，不要任何前后缀。"
)

# facts 模式（T3.2）：不发图，结构化观测事实是唯一信息源（EDGE v3.0"语言只做语言"）
# 注意：JSON 示例花括号必须双写（{{ }}）——本模板经 .format() 注入 facts_json
# 注意：可变的事实 JSON 必须放在提示词末尾——llama-server 按前缀复用 KV 缓存，
# 静态指令在前可让后续叙述的 prompt eval 只处理末尾事实（0.5B 上从 ~40s 降到几秒）
FACTS_NARRATE_PROMPT = (
    "你是校园监控场景分析员。只可复述最后给出的结构化事实，禁止推断画面外观或身份。\n"
    "只输出JSON，不要其他文字：\n"
    '{{"scene_summary":"逐人客观描述（只可复述事实中的人数与行为名）",'
    '"notable_activities":["事实中标记为异常的行为，无则空数组"],'
    '"environment":"未知"}}\n'
    "[结构化事实]\n{facts_json}"
)


class SceneNarrator:
    def __init__(self, vlm_config: dict, output_dir: str = "results"):
        cfg = vlm_config.get("scene_narrator", {}) if isinstance(vlm_config, dict) else {}
        self.enabled = bool(cfg.get("enabled", True))
        self.interval_seconds = float(cfg.get("narrate_interval_seconds", 20))
        self.frames_per_narration = int(max(2, cfg.get("frames_per_narration", 4)))
        self.summary_window = int(max(2, cfg.get("summary_window", 10)))
        self.summary_refresh_every = int(max(1, cfg.get("summary_refresh_every", 5)))
        self.max_latency_ms = int(cfg.get("max_latency_ms", 60000))
        self.max_tokens = int(max(64, cfg.get("max_tokens", 160)))
        self.max_image_px = int(cfg.get("max_image_px", 512))
        self.backoff_seconds = float(cfg.get("failure_backoff_seconds", 30))
        # T3.2: facts 模式——叙述信息源从"关键帧图像"切换为"结构化观测事实"
        # （edge 档语言层只做语言；输出 JSON schema 与 images 模式完全一致，下游零改动）
        self.narration_mode = str(cfg.get("narration_mode", "images"))

        # vLLM 客户端配置继承 vlm_config 顶层字段
        self.use_vllm = bool(vlm_config.get("use_vllm", False))
        self.vllm_url = str(vlm_config.get("vllm_url", "http://localhost:8001/v1/chat/completions"))
        self.vllm_api_key = str(vlm_config.get("vllm_api_key", "EMPTY"))
        self.model_name = str(vlm_config.get("model_name", "qwen2.5-vl-7b-instruct"))
        if not self.use_vllm:
            self.enabled = False
            self._last_error = "scene_narrator_requires_vllm_mode"

        self._source_fps = 25.0
        self._frame_buffer = None
        self._next_trigger_frame: int | None = None
        self._inflight = False
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene-narrator")

        self._narrations: deque[str] = deque(maxlen=self.summary_window)
        self._entries_since_summary = 0
        self._rolling_summary = ""
        self._pending_push: dict[str, Any] | None = None
        self._log_lock = threading.Lock()
        self._last_error = ""
        self._backoff_until = 0.0
        self._narration_count = 0

        log_dir = Path(output_dir) / "scene_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = str(output_dir)
        self.current_task_id = ""
        self._log_path = log_dir / "scene_log.jsonl"

    # ── 引擎接线 ─────────────────────────────────────────────

    def attach_buffer(self, frame_buffer) -> None:
        self._frame_buffer = frame_buffer

    def attach_vlm_router(self, vlm_router) -> None:
        """B3: 接入全局 VLM 并发槽位——提交叙述任务前先取槽，槽满则本轮顺延。"""
        self._vlm_router = vlm_router

    def set_source_fps(self, fps: float) -> None:
        self._source_fps = float(fps if fps and fps > 1.0 else 25.0)

    # ── 主入口（推理线程调用，非阻塞） ────────────────────────

    def build_detection_summary(self, detections: list[dict] | None, frame_shape: tuple | None) -> str:
        """把引擎的逐目标行为判定整理为注入文本（A：检测事实注入）。

        detections: [{"track_id", "label", "confidence", "bbox"}, ...]
        同时把结构化行缓存到 self.last_detection_refs_struct（供 _missing_abnormal_refs
        消费，消除字符串切分耦合——WEAKNESS_AUDIT 复查#三-3）。
        """
        self.last_detection_refs_struct = self._build_detection_struct(detections, frame_shape)
        if not detections:
            return ""
        return "\n".join(self._format_ref_line(item) for item in self.last_detection_refs_struct)

    @staticmethod
    def _build_detection_struct(detections: list[dict] | None, frame_shape: tuple | None) -> list[dict]:
        h, w = (frame_shape or (1080, 1920))[:2]
        rows = []
        for item in detections or []:
            try:
                x, y, bw, bh = [float(v) for v in item.get("bbox", [0, 0, 0, 0])]
                cx = (x + bw / 2.0) / max(1.0, float(w))
                cy = (y + bh / 2.0) / max(1.0, float(h))
                col = "左" if cx < 0.33 else ("中" if cx < 0.67 else "右")
                row = "上" if cy < 0.33 else ("中" if cy < 0.67 else "下")
                pos = f"画面{row}{col}部"
            except Exception:
                pos = "画面中"
            label = str(item.get("label", "Normal"))
            rows.append({
                "track_id": str(item.get("track_id", "?")),
                "pos": pos,
                "label": label,
                "name": LABEL_NAMES.get(label, label),
                "confidence": round(float(item.get("confidence", 0.0)), 2),
            })
        return rows

    @staticmethod
    def _format_ref_line(row: dict) -> str:
        return f"- {row['track_id']}（{row['pos']}）：{row['name']}（{row['label']}），置信度{row['confidence']:.2f}"

    def on_frame(self, frame_id: int, person_count: int = 0,
                 detections: list[dict] | None = None,
                 frame_shape: tuple | None = None) -> None:
        if not self.enabled:
            return
        interval_frames = max(1, int(round(self.interval_seconds * self._source_fps)))
        with self._lock:
            if self._inflight or time.time() < self._backoff_until:
                return
            if self._next_trigger_frame is None:
                self._next_trigger_frame = frame_id + interval_frames
                return
            if frame_id < self._next_trigger_frame:
                return
            # B3: 槽满时不动触发点，下一帧重试（本帧改判下一轮）
            router = getattr(self, "_vlm_router", None)
            if router is not None and not router.try_acquire(timeout=0.0):
                return
            self._next_trigger_frame = frame_id + interval_frames

            if self._frame_buffer is None:
                if router is not None:
                    router.release()
                return
            sampled = self._frame_buffer.sample(self.frames_per_narration, window_seconds=self.interval_seconds)
            if len(sampled) < 2:
                if router is not None:
                    router.release()
                return

            self._inflight = True
            frames = [item[1] for item in sampled]
            sample_frame_ids = [item[0] for item in sampled]
            log_path = str(self._log_path)
            detection_summary = self.build_detection_summary(detections, frame_shape)
            detection_struct = list(getattr(self, "last_detection_refs_struct", []) or [])
            facts_json = None
            if self.narration_mode == "facts":
                # T3.2: 结构化事实为唯一信息源（不发图，语言层 token 减半且零幻觉面）
                from edge_migration.fact_compiler import build_scene_facts
                facts_json = build_scene_facts(
                    frame_id=frame_id, fps=self._source_fps,
                    tracks=[{"track_id": r["track_id"], "label": r["label"],
                             "confidence": r["confidence"]} for r in detection_struct])

        self._executor.submit(self._narrate_job, list(frames) if facts_json is None else [],
                              sample_frame_ids, frame_id,
                              int(person_count), log_path, detection_summary, detection_struct,
                              facts_json)

    # ── 工作线程 ─────────────────────────────────────────────

    @staticmethod
    def _build_narrate_prompt(detection_summary: str) -> str:
        """叙述 prompt：基础填空式 + 可选检测事实注入（要求逐人核对后采纳）。"""
        prompt = (
            "你是校园监控系统的场景分析员。以下是按时间顺序排列的一组监控画面关键帧。"
            "请逐人核对画面中每个人正在做什么，输出JSON（不要包含任何其他文字）：\n"
            '{"scene_summary": "一段话客观描述场景中的人员与正在发生的行为，须逐人说明其状态与动作", '
            '"notable_activities": ["值得注意或异常的活动，逐条列出，无则空数组"], '
            '"environment": "场景环境简述（室内/室外、光照、区域类型）"}'
        )
        if detection_summary:
            prompt += (
                "\n\n行为检测模块参考（姿态/动作模型基于更高清的人员裁剪序列与骨骼关键点判定，"
                "其分辨率高于你看到的全景帧，动作可能仅在部分帧中可见。"
                "请对照画面核对：能在画面中确认的，在叙述中明确写出该行为；"
                "部分帧可见或看不清的，也应写入并标注（细节待确认）；明显不符的不要采纳）：\n" + detection_summary
            )
        return prompt

    @staticmethod
    def _missing_abnormal_refs_struct(rows: list[dict], narrated_text: str) -> str:
        """确定性兜底（结构化版）：检测报出的异常行为若未被叙述提及，返回需追加的联动项。"""
        if not rows:
            return ""
        lines = []
        for row in rows:
            if row["name"] in ("正常", "Normal"):
                continue
            if row["name"] and row["name"] not in narrated_text:
                lines.append(f"{row['track_id']} {row['name']}")
        return "；".join(lines)

    def _narrate_job(self, frames: list[np.ndarray], sample_frame_ids: list[int], frame_id: int, person_count: int,
                     log_path: str = "", detection_summary: str = "", detection_struct: list | None = None,
                     facts_json: dict[str, Any] | None = None) -> None:
        start = time.perf_counter()
        try:
            if facts_json is not None:
                prompt = FACTS_NARRATE_PROMPT.format(
                    facts_json=json.dumps(facts_json, ensure_ascii=False))
                chat_frames = None  # facts 模式零图像
            else:
                prompt = self._build_narrate_prompt(detection_summary)
                chat_frames = frames
            text = vlm_chat(
                base_url=self.vllm_url,
                api_key=self.vllm_api_key,
                model=self.model_name,
                prompt=prompt,
                frames=chat_frames,
                max_tokens=self.max_tokens,
                timeout=self.max_latency_ms / 1000.0,
                max_image_px=self.max_image_px,
            )
            parsed = parse_json_loose(text) or {}
            scene_summary = str(parsed.get("scene_summary", ""))[:500]
            # VLM 原始输出留档（不含检测联动拼接行）：叙述一致率评估（#8）以它为口径
            vlm_summary_raw = scene_summary
            notable_text = " ".join(str(a) for a in parsed.get("notable_activities", []))
            # A-兜底：检测报出的异常行为若 VLM 未提及，追加检测联动行——
            # 但降级为"待核实"表述（#4）：不冒充叙述事实，防止检测误报被洗进真值日志；
            # 事实仍保留（不删除），由读者/检索方自行加权。
            # 结构化行优先（消除字符串切分耦合）；无结构时回退旧解析
            # 结构化行由 build_detection_summary 总是产出（空 detections → 空 refs）
            missing = self._missing_abnormal_refs_struct(detection_struct or [], scene_summary + " " + notable_text)
            unverified_refs = [m.strip() for m in missing.split("；") if m.strip()] if missing else []
            if missing:
                scene_summary = (scene_summary + " " if scene_summary else "") + (
                    f"【检测联动·待核实】检测模块报告：{missing}"
                    "（该行为来自行为检测模型，本组画面帧中未获叙述确认）。"
                )
            record = {
                "type": "scene_narration",
                "task_id": self.current_task_id,
                "video_time_sec": round(frame_id / self._source_fps, 2),
                "frame_id": int(frame_id),
                "sample_frame_ids": sample_frame_ids,
                "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scene_summary": scene_summary,
                "vlm_summary_raw": vlm_summary_raw,
                "unverified_refs": unverified_refs,
                "notable_activities": [str(a)[:200] for a in parsed.get("notable_activities", []) if str(a).strip()][:8],
                "environment": str(parsed.get("environment", ""))[:200],
                "detection_refs": detection_summary,
                "detection_refs_struct": detection_struct or [],
                "person_count": int(person_count),
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "model": self.model_name,
            }
            self._persist(record, Path(log_path) if log_path else self._log_path)
            with self._lock:
                self._pending_push = record
                self._narration_count += 1
                if record["scene_summary"]:
                    self._narrations.append(record["scene_summary"])
                    self._entries_since_summary += 1
                need_summary = self._entries_since_summary >= self.summary_refresh_every
            if need_summary:
                self._rebuild_rolling_summary()
            self._last_error = ""
        except Exception as exc:
            self._last_error = f"narrate_failed: {exc}"
            with self._lock:
                self._backoff_until = time.time() + self.backoff_seconds
        finally:
            router = getattr(self, "_vlm_router", None)
            if router is not None:
                router.release()
            with self._lock:
                self._inflight = False

    def _rebuild_rolling_summary(self) -> None:
        """用纯文本调用重建滚动摘要（低频，工作线程内串行执行）。"""
        entries = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(list(self._narrations)[-self.summary_window:]))
        if not entries.strip():
            return False
        try:
            text = vlm_chat(
                base_url=self.vllm_url,
                api_key=self.vllm_api_key,
                model=self.model_name,
                prompt=DEFAULT_SUMMARY_PROMPT.format(entries=entries),
                frames=None,
                max_tokens=max(64, self.max_tokens // 2),
                timeout=self.max_latency_ms / 1000.0,
            )
            with self._lock:
                self._rolling_summary = text.strip()[:400]
                self._entries_since_summary = 0
        except Exception as exc:
            self._last_error = f"summary_failed: {exc}"

    def _persist(self, record: dict, log_path: Path | None = None) -> None:
        path = log_path or self._log_path
        with self._log_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── 任务隔离 ─────────────────────────────────────────────

    def begin_task(self, task_id: str) -> None:
        """新任务开始：切换到该任务专属日志并重置调度状态。

        帧号在任务间会归零重来，若不重置 _next_trigger_frame，
        新任务要跑到上个任务的残留触发点之后才会叙述。
        """
        with self._lock:
            self.current_task_id = str(task_id)
            self._log_path = Path(self._output_dir) / "scene_log" / f"scene_log_{task_id}.jsonl"
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._next_trigger_frame = None
            self._pending_push = None
            # 新任务清掉上个任务的失败态/退避，避免 UI 继续显示旧的“异常”
            self._last_error = ""
            self._backoff_until = 0.0
            self._narrations.clear()
            self._entries_since_summary = 0
            self._rolling_summary = ""

    # ── 读取接口 ─────────────────────────────────────────────

    def consume_pending_push(self) -> dict[str, Any] | None:
        with self._lock:
            pending, self._pending_push = self._pending_push, None
        return pending

    def get_context(self) -> str:
        """供告警裁决 / 灰区复核注入的场景上下文（滚动摘要 + 最近一条叙述）。"""
        with self._lock:
            summary = self._rolling_summary
            latest = self._narrations[-1] if self._narrations else ""
        parts = []
        if summary:
            parts.append(f"近期场景摘要：{summary}")
        if latest:
            parts.append(f"最近一段场景：{latest}")
        return "\n".join(parts)

    def get_log(self, limit: int = 50, task_id: str | None = None) -> list[dict[str, Any]]:
        path = self._log_path
        if task_id:
            candidate = Path(self._output_dir) / "scene_log" / f"scene_log_{task_id}.jsonl"
            if candidate.exists():
                path = candidate
        try:
            with path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()[-max(1, int(limit)):]
            return [json.loads(line) for line in lines if line.strip()]
        except FileNotFoundError:
            return []
        except Exception:
            return []

    def status(self) -> dict[str, Any]:
        with self._lock:
            inflight = self._inflight
            backoff = max(0.0, self._backoff_until - time.time())
            count = self._narration_count
        return {
            "enabled": self.enabled,
            "use_vllm": self.use_vllm,
            "healthy": bool(self.enabled and not self._last_error),
            "interval_seconds": self.interval_seconds,
            "source_fps": self._source_fps,
            "narration_count": count,
            "inflight": inflight,
            "backoff_seconds_left": round(backoff, 1),
            "rolling_summary": self._rolling_summary,
            "current_task_id": self.current_task_id,
            "last_error": self._last_error,
        }

    def stop(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
