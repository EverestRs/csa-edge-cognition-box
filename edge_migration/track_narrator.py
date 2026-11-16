from __future__ import annotations

"""按目标旁白合成（时刻②"点谁说谁"的语言层，01 §2.5c）。

反幻觉设计：prompt 中事实 JSON 是唯一信息源；LLM 不可用/超时时返回
fact_compiler.facts_to_text 的确定性短句（source="fallback"），
上层按同字段渲染——降级路径与真实路径输出同构，演示不断档。
"""

import time
from typing import Any

from .fact_compiler import CLASS_CN, facts_to_text

NARRATE_PROMPT = (
    "你是校园安防系统的行为观察播报员。下面是一个监控目标的结构化观测事实（JSON），"
    "它是你唯一的信息来源。\n"
    "{facts_json}\n"
    "请用一句不超过40字的中文播报该目标当前在做什么。要求：\n"
    "1) 只可复述事实中出现的行为名、次数、时长、位移；"
    "2) 禁止推测画面外观、身份、情绪或事实之外的任何内容；"
    "3) 不要输出JSON、不要解释、只输出这一句话。"
)

SCENE_NARRATE_PROMPT = (
    "你是监控场景播报员。以下为一个时间窗口的场景观测事实（JSON），是唯一信息来源。\n"
    "{facts_json}\n"
    "请用不超过50字中文概括此刻场景。仅复述事实中的数字与行为名，禁止添加观察之外的描述。"
    '仅输出JSON：{{"scene_summary": "<概括>", "notable_activities": "<异常点或空字符串>"}}'
)

SCENE_CHAT_PROMPT = (
    "你是校园监控系统的问答助手。以下是系统场景记忆中检索到的历史记录（JSON 数组，"
    "按时间排序），它们是你唯一的信息来源：\n{context_json}\n"
    "用户问题：{question}\n"
    "要求：1) 只可复述/整合记录中出现的Time、行为、人数与叙述文本；"
    "2) 记录中没有答案时 answer 必须为「记忆窗口内无相关记录」；"
    "3) 不要编造任何画面细节。\n"
    '仅输出JSON：{{"answer": "<不超过80字的中文回答>", '
    '"evidence_times": ["<引用记录的 video_time_sec>"]}}'
)


class TrackNarrator:
    def __init__(self, vlm_config: dict[str, Any] | None = None):
        cfg = dict(vlm_config or {})
        self.vllm_url = str(cfg.get("vllm_url", "http://localhost:8001/v1/chat/completions"))
        self.api_key = str(cfg.get("vllm_api_key", "EMPTY"))
        self.model_name = str(cfg.get("model_name", "qwen2.5-1.5b-instruct"))
        self.max_tokens = int(cfg.get("narrate_max_tokens", 80))
        self.timeout = float(cfg.get("narrate_timeout_s", 6.0))
        self.last_debug: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def _chat(self, prompt: str) -> str:
        from src.core.vlm_client import vlm_chat  # 延迟导入：脱离引擎独立可测
        return vlm_chat(self.vllm_url, self.api_key, self.model_name,
                        prompt, frames=None, max_tokens=self.max_tokens,
                        timeout=self.timeout)

    def narrate_track(self, track_facts: dict[str, Any]) -> dict[str, Any]:
        return self._narrate(NARRATE_PROMPT, track_facts, key="summary")

    def narrate_scene(self, scene_facts: dict[str, Any]) -> dict[str, Any]:
        return self._narrate(SCENE_NARRATE_PROMPT, scene_facts, key="summary",
                             json_out=True)

    def answer_scene(self, question: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
        """时刻③和录像聊天：场景记忆检索记录 → 文本 LLM 合成回答（记录=唯一信息源）。"""
        import json as _json
        t0 = time.perf_counter()
        context = [{"video_time_sec": h.get("video_time_sec"), "task_id": h.get("task_id"),
                    "doc_type": h.get("doc_type"), "text": str(h.get("text", ""))[:200]}
                   for h in (hits or [])]
        fallback = ("记忆窗口内无相关记录" if not context else
                    "；".join(f"t={h.get('video_time_sec')}s {str(h.get('text', ''))[:40]}"
                              for h in context[:2]))
        out: dict[str, Any] = {"ok": False, "answer": fallback, "source": "fallback",
                               "evidence_times": [], "latency_ms": 0}
        try:
            raw = self._chat(SCENE_CHAT_PROMPT.format(
                context_json=_json.dumps(context, ensure_ascii=False),
                question=question))
            from src.core.vlm_client import parse_json_loose
            parsed = parse_json_loose(raw) or {}
            answer = str(parsed.get("answer", "")).strip()
            if answer and answer != "None":
                out.update({"ok": True, "answer": answer[:160], "source": "llm",
                            "evidence_times": [str(t) for t in parsed.get("evidence_times", [])][:5]})
        except Exception as exc:
            self.last_debug["chat_error"] = str(exc)
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    # ------------------------------------------------------------------ #
    def _narrate(self, prompt_tpl: str, facts: dict[str, Any], *, key: str,
                 json_out: bool = False) -> dict[str, Any]:
        import json as _json
        t0 = time.perf_counter()
        fallback = facts_to_text(facts).get(key, "")
        prompt = prompt_tpl.format(facts_json=_json.dumps(facts, ensure_ascii=False))
        out: dict[str, Any] = {"ok": False, "text": fallback, "source": "fallback",
                               "latency_ms": 0}
        try:
            raw = self._chat(prompt)
            text = (raw or "").strip().strip('"')
            if json_out:
                from src.core.vlm_client import parse_json_loose
                parsed = parse_json_loose(raw) or {}
                text = str(parsed.get("scene_summary", "")).strip() or fallback
            if text and text != "None":
                out.update({"ok": True, "text": text[:80], "source": "llm"})
            else:
                out["text"] = fallback
        except Exception as exc:
            self.last_debug["llm_error"] = f"{type(exc).__name__}: {exc}"
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self.last_debug.update({"facts_kind": facts.get("kind"), "source": out["source"],
                                "latency_ms": out["latency_ms"], "ts": time.time()})
        return out


def anti_hallucination_check(narration: str, facts: dict[str, Any]) -> dict[str, Any]:
    """03 §3 时刻②反幻觉抽查：旁白中的数字与行为词必须能回溯到事实。"""
    import re
    problems: list[str] = []
    fact_blob = str(facts)
    for num in set(re.findall(r"\d+(?:\.\d+)?", narration or "")):
        if num not in fact_blob:
            problems.append(f"数字无出处: {num}")
    if not facts.get("crop_confirmed"):
        # 未被 crop 确认的敏感行为词不得凭空出现在旁白里
        for sensitive in ("吸烟", "跌倒", "打架", "霸凌", "手机"):
            if sensitive in (narration or "") and sensitive not in str(
                    {CLASS_CN.get(facts.get("dominant_label", ""), "")}):
                problems.append(f"未经确认的敏感行为词: {sensitive}")
    return {"ok": not problems, "problems": problems}
