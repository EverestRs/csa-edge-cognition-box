from __future__ import annotations

"""自然语言告警规则引擎 (NLAlertRuleEngine)。

赛题核心能力：用户输入一句话（如"告警有人翻越围栏"/"识别有人做出不文明手势"），
系统实时理解并触发告警。设计为两级匹配：

  ① 闭集语义映射（零成本，逐帧）：规则解析器把需求映射到内置 11 类之一
     （如"有人跌倒"→H03），主路径识别结果直接命中，毫秒级响应。
  ② 开放词汇判定（VLM，周期/加急）：体系外需求（如"不文明手势"）由本地
     Qwen2.5-VL 对环形缓冲关键帧做批量判定——所有启用中的开放规则合并为
     一次调用；场景叙述器报告异常活动时加急触发。

规则命中 → 写入 event_{frame_id}_rule_{rule_id}.json（自动并入 /api/v1/events）
         → 挂到 FrameResult.rule_alerts 随 WebSocket 推送。

解析失败降级：VLM 不可用时用内置中文关键词表做映射，无法映射的归为开放规则。
"""

import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.core.vlm_client import parse_json_loose, vlm_chat

RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "alert_rules.json"

# 规则解析 prompt（含少量示例锚定输出格式）
PARSE_PROMPT = (
    "你是校园安防系统的告警规则解析器。用户会用一句自然语言描述告警需求。\n"
    "系统内置以下闭集行为类别（由姿态模型实时识别）：\n"
    "H01=打架斗殴(互殴), H02=校园霸凌(单向攻击), H03=跌倒/昏厥, H04=疑似轻生/极端危险,\n"
    "H05=破坏公共设施, M01=吸烟/点火, M02=长时间使用手机, M03=遮挡/干扰摄像头,\n"
    "S01=异常徘徊(来回走动), S02=翻越围栏/闯入禁区。\n\n"
    "示例：\n"
    '输入"告警有人翻越围栏" → {"matched_classes": ["S02"], "open_description": "有人翻越围栏", '
    '"use_open_vocabulary": false, "confidence_threshold": 0.6, "reasoning": "对应内置S02"}\n'
    '输入"识别有人做出不文明手势" → {"matched_classes": [], "open_description": "有人做出竖中指等侮辱性不文明手势", '
    '"use_open_vocabulary": true, "confidence_threshold": 0.6, "reasoning": "超出内置类别，需开放词汇判定"}\n\n'
    "请把下面的用户需求解析为JSON（不要输出其他文字）：\n"
    '输入"{user_text}"'
)

OPEN_CHECK_PROMPT = (
    "你是校园安防系统的开放事件判定器。以下是按时间顺序排列的一组监控关键帧，它们覆盖最近的一个时间窗口。\n"
    "请仔细观察每一帧画面，判断该时间窗口内是否发生了以下事件：{rule_desc}\n\n"
    "判定标准：只要任意一帧中出现了与事件描述相符的动作或行为，就判 true；"
    "只有当所有帧都与事件无关时才判 false。confidence 表示你对该判定的把握（0到1）。\n"
    '必须基于画面实际内容作答。仅输出JSON（不要输出其他文字）：'
    '{{"match": <true或false>, "confidence": <0到1的小数>, "evidence": "<你看到的画面内容>"}}'
)

# 模板占位回显特征：模型照抄示例默认值 = 未基于画面作答，需重试
_PLACEHOLDER_EVIDENCE = "画面依据的简短描述"


def _is_placeholder_response(parsed: dict[str, Any]) -> bool:
    evidence = str(parsed.get("evidence", "")).strip()
    if evidence == _PLACEHOLDER_EVIDENCE:
        return True
    return (not parsed.get("match", False)) and float(parsed.get("confidence", 0.0)) == 0.0 and not evidence

# 解析失败时的关键词降级映射（与 vlm_reviewer 的中文关键词表保持一致口径）
KEYWORD_CLASS_MAP: dict[str, list[str]] = {
    "H01": ["打架", "斗殴", "互殴", "挥拳", "踢打", "扭打", "推搡", "打斗", "搏斗"],
    "H02": ["霸凌", "欺凌", "欺负", "殴打", "单方攻击"],
    "H03": ["跌倒", "摔倒", "倒地", "昏厥", "晕倒", "滑倒"],
    "H04": ["轻生", "跳楼", "撞墙", "自缢", "上吊"],
    "H05": ["破坏", "砸", "踹门", "踢门", "毁坏"],
    "M01": ["吸烟", "抽烟", "点火"],
    "M02": ["使用手机", "玩手机", "看手机", "打电话"],
    "M03": ["遮挡", "挡住", "遮摄像头"],
    "S01": ["徘徊", "来回走", "踱步", "逗留"],
    "S02": ["翻越", "翻围栏", "翻栏杆", "闯入", "跨栏"],
}

VALID_CLASSES = set(KEYWORD_CLASS_MAP.keys())


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class NLAlertRuleEngine:
    def __init__(self, vlm_config: dict, frame_buffer=None, output_dir: str = "results",
                 rules_path: str | Path | None = None):
        cfg = vlm_config.get("nl_rule_engine", {}) if isinstance(vlm_config, dict) else {}
        self.enabled = bool(cfg.get("enabled", True))
        self.open_check_interval_seconds = float(max(5.0, cfg.get("open_check_interval_seconds", 15.0)))
        self.expedited_check_gap_seconds = float(max(2.0, cfg.get("expedited_check_gap_seconds", 6.0)))
        self.frames_per_check = int(max(2, cfg.get("frames_per_check", 4)))
        self.rule_cooldown_seconds = float(max(0.0, cfg.get("rule_cooldown_seconds", 10.0)))
        self.default_min_confidence = float(min(1.0, max(0.1, cfg.get("default_min_confidence", 0.6))))
        self.max_image_px = int(cfg.get("max_image_px", 512))
        self.max_latency_ms = int(cfg.get("max_latency_ms", 60000))
        self.use_images = bool(cfg.get("use_images", True))
        self.backoff_seconds = float(cfg.get("failure_backoff_seconds", 30))

        self.use_vllm = bool(vlm_config.get("use_vllm", False))
        self.vllm_url = str(vlm_config.get("vllm_url", "http://localhost:8001/v1/chat/completions"))
        self.vllm_api_key = str(vlm_config.get("vllm_api_key", "EMPTY"))
        self.model_name = str(vlm_config.get("model_name", "qwen2.5-vl-7b-instruct"))

        self._frame_buffer = frame_buffer
        self._source_fps = 25.0
        self._rules_path = Path(rules_path) if rules_path else RULES_PATH
        self._rules: dict[str, dict[str, Any]] = {}
        self._load_rules()

        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nl-rule-engine")
        self._inflight = False
        self._next_check_frame: int | None = None
        self._expedite_until_frame = 0  # 有叙述异常活动时，该帧号前每帧都尝试加急（受单飞限制）
        self._notable_activities: deque[str] = deque(maxlen=6)
        self._pending_pushes: list[dict[str, Any]] = []
        self._last_error = ""
        self._backoff_until = 0.0
        self._check_count = 0
        # ROADMAP Stage 2 前置：单轮开放词汇判定调用预算 + 轮转游标。
        # 每规则一次独立小调用是既有教训（合并输出会被 7B 漏答），不能合并；
        # 规则增多时按预算分轮判定，轮转游标保证公平（原型自写规则会持续增多）
        self.max_open_checks_per_cycle = int(max(1, cfg.get("max_open_checks_per_cycle", 3)))
        self._rule_cursor = 0
        self._checks_skipped = 0
        self._results_dir = Path(output_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    # ── 规则持久化与 CRUD ────────────────────────────────────

    def _load_rules(self) -> None:
        try:
            doc = json.loads(self._rules_path.read_text(encoding="utf-8"))
            rules = doc.get("rules", [])
            self._rules = {r["rule_id"]: r for r in rules if isinstance(r, dict) and r.get("rule_id")}
        except FileNotFoundError:
            self._rules = {}
        except Exception:
            self._rules = {}

    def _save_rules(self) -> None:
        self._rules_path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"rules": sorted(self._rules.values(), key=lambda r: r.get("rule_id", ""))}
        self._rules_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    def parse_rule_text(self, text: str) -> dict[str, Any]:
        """把自然语言需求解析为结构化规则。VLM 优先，失败降级关键词映射。"""
        text = (text or "").strip()
        if self.use_vllm:
            try:
                out = vlm_chat(
                    base_url=self.vllm_url,
                    api_key=self.vllm_api_key,
                    model=self.model_name,
                    prompt=PARSE_PROMPT.replace("{user_text}", text),
                    frames=None,
                    max_tokens=160,
                    timeout=self.max_latency_ms / 1000.0,
                )
                parsed = parse_json_loose(out)
                if parsed is not None:
                    classes = [c for c in parsed.get("matched_classes", []) if str(c) in VALID_CLASSES]
                    return {
                        "matched_classes": classes,
                        "open_description": str(parsed.get("open_description", text))[:200] or text,
                        "use_open_vocabulary": bool(parsed.get("use_open_vocabulary", not classes)),
                        "confidence_threshold": float(min(1.0, max(0.1, parsed.get("confidence_threshold", self.default_min_confidence)))),
                        "reasoning": str(parsed.get("reasoning", ""))[:200],
                        "parser": "vlm",
                    }
            except Exception as exc:
                self._last_error = f"parse_failed: {exc}"
        # 关键词降级
        matched = [cls for cls, kws in KEYWORD_CLASS_MAP.items() if any(kw in text for kw in kws)]
        return {
            "matched_classes": matched,
            "open_description": text,
            "use_open_vocabulary": not matched,
            "confidence_threshold": self.default_min_confidence,
            "reasoning": "关键词降级解析" + ("" if matched else "，未匹配到内置类别，归为开放词汇规则"),
            "parser": "keyword_fallback",
        }

    def add_rule(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ValueError("规则文本不能为空")
        parsed = self.parse_rule_text(text)
        with self._lock:
            rule_id = f"rule_{(max((int(k.split('_')[1]) for k in self._rules if k.startswith('rule_')), default=0)) + 1:04d}"
            rule = {
                "rule_id": rule_id,
                "text": text,
                "enabled": True,
                "created_at": _now_str(),
                "parsed": parsed,
                "stats": {"evaluated": 0, "triggered": 0, "last_triggered_frame": None, "last_triggered_at": None},
            }
            self._rules[rule_id] = rule
            self._save_rules()
        return dict(rule)

    def update_rule(self, rule_id: str, text: str | None = None, enabled: bool | None = None,
                    min_confidence: float | None = None) -> dict[str, Any]:
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule is None:
                raise KeyError(f"rule not found: {rule_id}")
            if text is not None and str(text).strip() and str(text).strip() != rule["text"]:
                rule["text"] = str(text).strip()
                rule["parsed"] = self.parse_rule_text(rule["text"])
            if enabled is not None:
                rule["enabled"] = bool(enabled)
            if min_confidence is not None:
                rule.setdefault("parsed", {})["confidence_threshold"] = float(min(1.0, max(0.1, min_confidence)))
            self._save_rules()
            return dict(rule)

    def remove_rule(self, rule_id: str) -> bool:
        with self._lock:
            if rule_id not in self._rules:
                return False
            del self._rules[rule_id]
            self._save_rules()
        return True

    def list_rules(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in sorted(self._rules.values(), key=lambda r: r.get("rule_id", ""))]

    def get_rule(self, rule_id: str) -> dict[str, Any] | None:
        with self._lock:
            rule = self._rules.get(rule_id)
            return dict(rule) if rule else None

    def reset_scheduling(self) -> None:
        """新任务开始：重置按帧号调度的状态（帧号已归零，旧调度点作废）。"""
        with self._lock:
            self._next_check_frame = None
            self._expedite_until_frame = 0
            self._pending_pushes = []

    # ── 运行时接线 ───────────────────────────────────────────

    def attach_buffer(self, frame_buffer) -> None:
        self._frame_buffer = frame_buffer

    def attach_vlm_router(self, vlm_router) -> None:
        """B3: 接入全局 VLM 并发槽位——槽满时顺延本轮调度，不丢失规则。"""
        self._vlm_router = vlm_router

    def set_source_fps(self, fps: float) -> None:
        self._source_fps = float(fps if fps and fps > 1.0 else 25.0)

    def note_activities(self, activities: list[str], frame_id: int) -> None:
        """场景叙述器报告值得注意的活动时调用——加急开放词汇判定。"""
        clean = [str(a) for a in activities if str(a).strip()]
        if not clean:
            return
        with self._lock:
            for a in clean:
                if a not in self._notable_activities:
                    self._notable_activities.append(a)
            if frame_id > self._expedite_until_frame:
                self._expedite_until_frame = frame_id + int(self.expedited_check_gap_seconds * self._source_fps)

    # ── ① 闭集语义映射（逐帧、零成本） ────────────────────────

    def match_closed(self, track_label: str, confidence: float, frame_id: int) -> list[dict[str, Any]]:
        """对单个 track 的最终标签做规则命中检查（无副作用，命中需经 commit_hits 落盘）。"""
        if not self.enabled:
            return []
        hits: list[dict[str, Any]] = []
        with self._lock:
            for rule in self._rules.values():
                if not rule.get("enabled", True):
                    continue
                parsed = rule.get("parsed", {})
                if str(track_label) not in set(parsed.get("matched_classes", [])):
                    continue
                threshold = float(parsed.get("confidence_threshold", self.default_min_confidence))
                if float(confidence) < threshold:
                    continue
                if not self._cooldown_ok(rule, frame_id):
                    continue
                hits.append({
                    "rule_id": rule["rule_id"],
                    "rule_text": rule["text"],
                    "matched_by": "closed_class",
                    "matched_class": str(track_label),
                    "confidence": round(float(confidence), 4),
                    "evidence": f"闭集映射：识别结果 {track_label}({confidence:.2f}) 命中规则「{rule['text']}」",
                })
        return hits

    def commit_hits(self, hits: list[dict[str, Any]], frame_id: int,
                    detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """把本帧命中的规则落盘为告警事件（冷却检查 + 统计 + WebSocket 推送）。"""
        committed: list[dict[str, Any]] = []
        if not hits:
            return committed
        with self._lock:
            seen: set[str] = set()
            for hit in hits:
                rule_id = hit["rule_id"]
                if rule_id in seen or rule_id not in self._rules:
                    continue
                rule = self._rules[rule_id]
                if not self._cooldown_ok(rule, frame_id):
                    continue
                seen.add(rule_id)
                rule["stats"]["triggered"] += 1
                rule["stats"]["last_triggered_frame"] = int(frame_id)
                rule["stats"]["last_triggered_at"] = _now_str()
                rule["stats"]["last_triggered_ts"] = time.time()
                event = {
                    "frame_id": int(frame_id),
                    "timestamp": _now_str(),
                    "source": "nl_rule",
                    "rule": dict(hit),
                    "alert": {
                        "triggered": True,
                        "type": "nl_rule_hit",
                        "source": "nl_rule",
                        "final_decision": "pending",
                        "rule_id": rule_id,
                        "rule_text": hit.get("rule_text", ""),
                        "snapshot_url": "",
                        "clip_path": "",
                    },
                    "detections": detections,
                }
                event_path = self._results_dir / f"event_{int(frame_id)}_rule_{rule_id}.json"
                try:
                    event_path.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as exc:
                    self._last_error = f"event_write_failed: {exc}"
                self._pending_pushes.append({
                    "type": "nl_rule_hit",
                    "frame_id": int(frame_id),
                    **hit,
                })
                committed.append(hit)
            self._save_rules()
        return committed

    def _cooldown_ok(self, rule: dict[str, Any], frame_id: int) -> bool:
        # 墙钟时间口径：帧号会随任务/片段重启归零，帧号差会造成跨任务误压制
        last_ts = float(rule.get("stats", {}).get("last_triggered_ts", 0.0))
        return (time.time() - last_ts) >= self.rule_cooldown_seconds

    # ── ② 开放词汇判定（VLM 周期/加急，批量合并调用） ──────────

    def on_frame(self, frame_id: int) -> None:
        """引擎每处理帧调用：按周期或加急信号发起开放词汇判定（异步、单飞）。"""
        if not self.enabled or not self.use_vllm:
            return
        with self._lock:
            if self._inflight or time.time() < self._backoff_until:
                return
            has_open_rules = any(
                r.get("enabled", True) and (r.get("parsed", {}).get("use_open_vocabulary") or not r.get("parsed", {}).get("matched_classes"))
                for r in self._rules.values()
            )
            if not has_open_rules:
                return
            interval_frames = self.open_check_interval_seconds * self._source_fps
            expedited = frame_id <= self._expedite_until_frame
            min_gap_frames = self.expedited_check_gap_seconds * self._source_fps
            if self._next_check_frame is None:
                # 首次判定不延迟：缓冲帧就绪即判（短片段也能覆盖）
                self._next_check_frame = frame_id
            due = frame_id >= self._next_check_frame
            # 加急：叙述器报告异常活动后，受最小间隔约束提前判定
            expedited_due = expedited and (self._next_check_frame - frame_id) >= (interval_frames - min_gap_frames)
            if not (due or expedited_due):
                return
            # B3: 全局槽位满时不消费调度槽位，下一帧重试
            router = getattr(self, "_vlm_router", None)
            if router is not None and not router.try_acquire(timeout=0.0):
                return
            if self._frame_buffer is None:
                if router is not None:
                    router.release()
                return
            sampled = self._frame_buffer.sample(self.frames_per_check)
            if len(sampled) < 2:
                # 缓冲未就绪：不消费本次调度槽位，下一帧重试
                if router is not None:
                    router.release()
                self._next_check_frame = frame_id
                return
            self._next_check_frame = frame_id + (min_gap_frames if expedited_due else interval_frames)
            self._inflight = True

        frames = [item[1] for item in sampled]
        # 加急轮预算翻倍：异常活动线索优先覆盖（仍受单飞与全局槽位约束）
        budget = self.max_open_checks_per_cycle * (2 if expedited_due else 1)
        self._executor.submit(self._open_check_job, frames, int(frame_id), budget)

    def _open_check_job(self, frames: list, frame_id: int, budget: int | None = None) -> None:
        try:
            hits = self._judge_frames(frames, frame_id, budget=budget)
            if hits:
                self.commit_hits(hits, frame_id, detections=[])
            self._check_count += 1
            self._last_error = ""
        except Exception as exc:
            self._last_error = f"open_check_failed: {exc}"
            self._backoff_until = time.time() + self.backoff_seconds
        finally:
            router = getattr(self, "_vlm_router", None)
            if router is not None:
                router.release()
            with self._lock:
                self._inflight = False

    def check_now(self, frames: list, frame_id: int) -> list[dict[str, Any]]:
        """同步开放词汇判定（评估/按需检查用）：对给定帧逐条判定启用中的开放规则。"""
        try:
            hits = self._judge_frames(frames, frame_id)
            committed = self.commit_hits(hits, frame_id, detections=[])
            self._check_count += 1
            return committed
        except Exception as exc:
            self._last_error = f"open_check_failed: {exc}"
            return []

    def _judge_frames(self, frames: list, frame_id: int, budget: int | None = None) -> list[dict[str, Any]]:
        """对启用中的开放规则逐条判定给定帧（每规则一次独立小调用）。

        budget：本轮最大调用次数（None=不限）。规则数超预算时按轮转游标分轮判定，
        保证每条规则最终都被覆盖——原型自写规则会持续增多，不能无限并发（ROADMAP Stage 2 前置）。
        """
        if not self.use_images:
            return []  # 纯文本 0.5B：开放规则不做视觉检查，闭集规则仍按行为标签匹配
        open_rules = [
            r for r in self.list_rules()
            if r.get("enabled", True)
            and (r.get("parsed", {}).get("use_open_vocabulary") or not r.get("parsed", {}).get("matched_classes"))
        ]
        if not open_rules:
            return []
        if budget is not None and len(open_rules) > budget:
            self._checks_skipped += len(open_rules) - budget
            total = len(open_rules)
            start = self._rule_cursor % total
            ordered = open_rules[start:] + open_rules[:start]
            open_rules = ordered[:budget]
            self._rule_cursor = (start + budget) % total
        hits: list[dict[str, Any]] = []
        for rule in open_rules:
            desc = str(rule.get("parsed", {}).get("open_description", rule["text"]))[:200]
            prompt = OPEN_CHECK_PROMPT.format(rule_desc=desc)
            text = vlm_chat(
                base_url=self.vllm_url,
                api_key=self.vllm_api_key,
                model=self.model_name,
                prompt=prompt,
                frames=frames,
                max_tokens=160,
                timeout=self.max_latency_ms / 1000.0,
                max_image_px=self.max_image_px,
            )
            parsed = parse_json_loose(text) or {}
            # 占位回显（照抄模板默认值）→ 追加一次强调观察的追问
            if _is_placeholder_response(parsed):
                retry_prompt = (
                    f"{prompt}\n\n注意：你上一次的回复是在复述输出模板而不是描述画面。"
                    "请再次仔细逐帧观察画面中的动作，基于实际所见内容重新作答。"
                )
                text2 = vlm_chat(
                    base_url=self.vllm_url,
                    api_key=self.vllm_api_key,
                    model=self.model_name,
                    prompt=retry_prompt,
                    frames=frames,
                    max_tokens=160,
                    timeout=self.max_latency_ms / 1000.0,
                    max_image_px=self.max_image_px,
                )
                parsed2 = parse_json_loose(text2) or {}
                if parsed2 and not _is_placeholder_response(parsed2):
                    parsed = parsed2
            if not bool(parsed.get("match", False)):
                continue
            conf = float(max(0.0, min(1.0, float(parsed.get("confidence", 0.0)))))
            threshold = float(rule.get("parsed", {}).get("confidence_threshold", self.default_min_confidence))
            if conf < threshold:
                continue
            with self._lock:
                stored = self._rules.get(rule["rule_id"])
                if stored is not None and not self._cooldown_ok(stored, frame_id):
                    continue
            hits.append({
                "rule_id": rule["rule_id"],
                "rule_text": rule["text"],
                "matched_by": "open_vocabulary",
                "matched_class": None,
                "confidence": round(conf, 4),
                "evidence": str(parsed.get("evidence", ""))[:200],
            })
        return hits

    # ── 读取接口 ─────────────────────────────────────────────

    def consume_pending_pushes(self) -> list[dict[str, Any]]:
        with self._lock:
            pending, self._pending_pushes = self._pending_pushes, []
        return pending

    def status(self) -> dict[str, Any]:
        with self._lock:
            rules = list(self._rules.values())
        return {
            "enabled": self.enabled,
            "use_vllm": self.use_vllm,
            "healthy": bool(self.enabled and not self._last_error),
            "rules_total": len(rules),
            "rules_enabled": sum(1 for r in rules if r.get("enabled", True)),
            "open_rules": sum(1 for r in rules if r.get("enabled", True)
                              and (r.get("parsed", {}).get("use_open_vocabulary")
                                   or not r.get("parsed", {}).get("matched_classes"))),
            "open_check_count": self._check_count,
            "open_checks_budget": self.max_open_checks_per_cycle,
            "open_checks_skipped": self._checks_skipped,
            "inflight": self._inflight,
            "last_error": self._last_error,
        }

    def stop(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
