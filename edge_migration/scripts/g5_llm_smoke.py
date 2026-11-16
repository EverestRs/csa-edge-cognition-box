#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G5 探针：文本 LLM 四类任务冒烟（T1.6）——布控解析/目标旁白/事件裁决/场景问答。

对任何 OpenAI 兼容端点可跑：现在打 7B vLLM（行为基线），盒上换 llama-server 1.5B（终态）。
JSON 合格率 < 要求值 → 触发 02 风险表"解析修复器"预案。
用法: python edge_migration/scripts/g5_llm_smoke.py [--base-url URL --model NAME | --config configs/vlm_config.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge_migration.fact_compiler import build_scene_facts, build_track_facts  # noqa: E402
from edge_migration.track_narrator import NARRATE_PROMPT  # noqa: E402
from src.core.nl_rule_engine import PARSE_PROMPT  # noqa: E402
from src.core.vlm_client import parse_json_loose, vlm_chat  # noqa: E402

ADJUDICATE_PROMPT = (
    "你是告警裁决器。以下为一次告警的结构化事实（JSON）：\n{facts_json}\n"
    "请仅依据事实判断告警是否成立。\n仅输出JSON：\n"
    '{{"decision": "<confirmed或rejected>", "confidence": <0到1>, "reason": "<不超过30字>"}}'
)
CHAT_PROMPT = (
    "你是监控场景问答器。场景观测事实（JSON）：\n{facts_json}\n"
    "用户问题：{question}\n"
    "仅依据事实回答；事实中找不到答案时 answer 输出「窗口内无相关记录」。\n"
    '仅输出JSON：{{"answer": "<不超过50字>", "evidence": "<引用事实中的字段名>"}}'
)

PARSE_SAMPLES = ["告警有人翻越围栏", "提醒我实验室门口有人抽烟", "识别有人鼓掌"]
FACTS_TRACK = build_track_facts(
    track_id="P001", frame_id=480, window_seconds=86.0,
    label_history=[("S01", 0.61)] * 9 + [("Normal", 0.7)] * 3,
    bboxes=[[300, 200, 60, 150]] * 12, crop_verdicts=["Normal"], speed_px_per_s=41.2)
FACTS_SCENE = build_scene_facts(frame_id=480, fps=4.8,
    tracks=[{"track_id": "P001", "bbox": [300, 200, 60, 150], "label": "S01", "confidence": 0.61},
            {"track_id": "P002", "bbox": [700, 240, 70, 160], "label": "Normal", "confidence": 0.83}],
    crop_verdicts={"P001": "Normal"}, zone_events=[])
FACTS_EVENT = {"alert": {"type": "h03_detected", "label": "H03", "confidence": 0.66,
                        "frame_id": 480, "snapshot_url": "results/snapshots/f480.jpg"}}


def case_llm_json(prompt: str, *, max_tokens: int, expect_keys: list[str],
                  url: str, key: str, model: str, timeout: float) -> dict:
    try:
        s = time.perf_counter()
        raw = vlm_chat(url, key, model, prompt, frames=None,
                       max_tokens=max_tokens, timeout=timeout)
        ms = (time.perf_counter() - s) * 1000
        parsed = parse_json_loose(raw)
        ok = isinstance(parsed, dict) and all(k in parsed and str(parsed[k]) not in ("", "None")
                                              for k in expect_keys)
        return {"ms": round(ms, 1), "ok": bool(ok),
                "raw_head": (raw or "")[:80], "parsed": parsed}
    except Exception as exc:
        return {"ms": None, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:160]}


def case_narrate(facts: dict, url: str, key: str, model: str, timeout: float) -> dict:
    try:
        s = time.perf_counter()
        raw = vlm_chat(url, key, model, NARRATE_PROMPT.format(
            facts_json=json.dumps(facts, ensure_ascii=False)), frames=None,
            max_tokens=80, timeout=timeout)
        ms = (time.perf_counter() - s) * 1000
        text = (raw or "").strip().strip('"')
        return {"ms": round(ms, 1), "ok": bool(1 <= len(text) <= 80), "raw_head": text[:80]}
    except Exception as exc:
        return {"ms": None, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:160]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--config", default="configs/vlm_config.yaml")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--require-valid", type=float, default=0.95)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--out", default="results/edge_m0/g5_llm_smoke.json")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    url, model = a.base_url, a.model
    if not url or not model:
        cfg = yaml.safe_load(open(a.config, encoding="utf-8")) or {}
        url = url or str(cfg.get("vllm_url", "http://localhost:8001/v1/chat/completions"))
        model = model or str(cfg.get("model_name", "qwen2.5-vl-7b-instruct"))

    suites = {
        # parse 与生产一致：PARSE_PROMPT.replace 注入（模板含字面 JSON 花括号，禁用 .format）
        "parse": [(lambda t=t: case_llm_json(PARSE_PROMPT.replace("{user_text}", t),
                                             max_tokens=256,
                                             expect_keys=["matched_classes", "open_description"],
                                             url=url, key=a.api_key, model=model, timeout=a.timeout))
                  for t in PARSE_SAMPLES for _ in range(a.reps)],
        "narrate_track": [lambda: case_narrate(FACTS_TRACK, url, a.api_key, model, a.timeout)
                          for _ in range(a.reps * 2)],
        "adjudicate": [lambda: case_llm_json(
            ADJUDICATE_PROMPT.format(facts_json=json.dumps(FACTS_EVENT, ensure_ascii=False)),
            max_tokens=128, expect_keys=["decision"],
            url=url, key=a.api_key, model=model, timeout=a.timeout)
            for _ in range(a.reps * 2)],
        "chat": [lambda: case_llm_json(
            CHAT_PROMPT.format(facts_json=json.dumps(FACTS_SCENE, ensure_ascii=False),
                               question="门口那个人在徘徊吗？"),
            max_tokens=160, expect_keys=["answer"],
            url=url, key=a.api_key, model=model, timeout=a.timeout)
            for _ in range(a.reps * 2)],
    }

    result = {"probe": "g5_llm_smoke", "ts": time.time(), "url": url, "model": model,
              "cases": {}}
    all_ok = True
    conn_errors = 0
    timeout_errors = 0
    any_valid = False
    for name, fns in suites.items():
        rows = [fn() for fn in fns]
        errs = [r.get("error", "") for r in rows if r.get("error")]
        conn_errors += sum(1 for e in errs if "Connection" in e)
        timeout_errors += sum(1 for e in errs if "Timeout" in e)
        oks = [r["ok"] for r in rows]
        any_valid |= any(oks)
        lat = [r["ms"] for r in rows if r.get("ms") is not None]
        result["cases"][name] = {
            "n": len(rows), "valid_rate": round(sum(oks) / max(len(oks), 1), 2),
            "p50_ms": round(sorted(lat)[len(lat) // 2], 1) if lat else None,
            "fail_samples": [r for r in rows if not r["ok"]][:2],
        }
        if sum(oks) / max(len(oks), 1) < a.require_valid:
            all_ok = False

    # D4: 服务失联(CONN)与模型过慢(SLOW)分流——处置预案完全不同
    if conn_errors and not any_valid:
        verdict = "UNREACHABLE"
    elif timeout_errors or not all_ok:
        verdict = "SLOW" if any_valid else "NEED_PARSER_OR_PROMPT_FIX"
    else:
        verdict = "PASS" if all_ok else "NEED_PARSER_OR_PROMPT_FIX"
    result["verdict"] = verdict
    result["error_counts"] = {"connection": conn_errors, "timeout": timeout_errors}
    result["latency_gates"] = {"parse_p50_s": 2.0, "narrate_p50_s": 1.0, "chat_p50_s": 10.0}
    json.dump(result, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=False))
    print(f"[g5] -> {a.out}")
    return 0 if not unreachable else 2


if __name__ == "__main__":
    sys.exit(main())
