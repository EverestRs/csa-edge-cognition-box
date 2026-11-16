from __future__ import annotations

"""演示剧本包（T6.5）：连排/拍摄日的防翻车保险。

设计原则（03 §5）：备用剧本走真实代码路径的降级入口，不是假数据——
- narrate：LLM 失败/降级时用预生成旁白（source=demo_preset，气泡上可见）；
- chat：预缓存问答命中直接返回（source=demo_cache），未命中走真实检索+合成；
- rules：预置三条口令供现场二选一输入（仍是真实解析路径，不是假规则）。

启用：model_config 顶层 demo.enabled=true 或环境变量 CSA_DEMO=1。
预热：POST /api/v1/demo/prime（调用现有 narrate/chat 真实链路生成并写回包文件）。
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_PACK = {"rules": [], "narrations": [], "chat": []}


class DemoPack:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False)) or os.getenv("CSA_DEMO") == "1"
        self.path = Path(str(cfg.get("pack_path", "configs/demo_pack.json")))
        self._lock = threading.Lock()
        self._rr_index = 0
        self.data: dict[str, Any] = dict(DEFAULT_PACK)
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = {**DEFAULT_PACK, **loaded}
        except Exception:
            pass
        self.last_debug: dict[str, Any] = {"enabled": self.enabled,
                                           "pack_loaded": bool(self.data.get("narrations") or self.data.get("chat"))}

    # ── 查询 ─────────────────────────────────────────────────
    def rule_texts(self) -> list[str]:
        return [str(t) for t in self.data.get("rules", []) if str(t).strip()]

    def next_narration(self) -> str:
        """轮转取预生成旁白（演示时每次点击看到不同剧本句）。"""
        arr = [str(t) for t in self.data.get("narrations", []) if str(t).strip()]
        if not arr:
            return ""
        with self._lock:
            text = arr[self._rr_index % len(arr)]
            self._rr_index += 1
        return text

    def cached_answer(self, question: str) -> str | None:
        """预缓存问答：问题包含关键词即命中（演示口径：问题基本按剧本问）。"""
        q = (question or "").strip().lower()
        if not q:
            return None
        for item in self.data.get("chat", []):
            keys = [str(k).lower() for k in item.get("keywords", []) if str(k).strip()]
            hit = any(k in q for k in keys) or (str(item.get("q", "")).lower() in q)
            if hit and str(item.get("a", "")).strip():
                return str(item["a"])
        return None

    # ── 预热写回 ─────────────────────────────────────────────
    def prime(self, narrations: list[str], chat: list[dict[str, Any]],
              rules: list[str] | None = None) -> dict[str, int]:
        with self._lock:
            if narrations:
                self.data["narrations"] = [str(n)[:120] for n in narrations if str(n).strip()][:12]
            if chat:
                self.data["chat"] = [{"q": c.get("q", ""), "a": str(c.get("a", ""))[:160],
                                      "keywords": c.get("keywords", [])} for c in chat if c.get("a")][:6]
            if rules:
                self.data["rules"] = [str(r)[:80] for r in rules if str(r).strip()][:4]
            self.data["primed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        self.last_debug["primed_at"] = self.data.get("primed_at")
        return {"narrations": len(self.data.get("narrations", [])),
                "chat": len(self.data.get("chat", [])),
                "rules": len(self.data.get("rules", []))}

    def status(self) -> dict[str, Any]:
        return {"enabled": self.enabled,
                "narrations": len(self.data.get("narrations", [])),
                "chat": len(self.data.get("chat", [])),
                "rules": len(self.data.get("rules", [])),
                "primed_at": self.data.get("primed_at", "")}
