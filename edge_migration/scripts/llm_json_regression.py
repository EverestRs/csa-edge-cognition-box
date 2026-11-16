#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T3.7：语言层 LLM 正式回归（100 prompts JSON 合格率）——g5 的正式口径薄封装。

默认 25 reps × 4 套件（布控解析/目标旁白/事件裁决/场景问答）= 100 次调用，
验收线：每套件 JSON 合格率 ≥95%（EDGE §4.2）。产物 results/edge_test/llm_json_regression.json。

用法:
  python edge_migration/scripts/llm_json_regression.py                # 打当前 vllm_url（盒上=1.5B）
  python edge_migration/scripts/llm_json_regression.py --tag server7b # 基线标记
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    g5 = ROOT / "edge_migration/scripts/g5_llm_smoke.py"
    out = ROOT / "results/edge_test/llm_json_regression.json"
    argv = [g5.name, "--reps", "25", "--require-valid", "0.95", "--out", str(out)]
    argv += sys.argv[1:]
    sys.argv = argv
    print("[regression] 100-prompt 语言层回归（4 套件 × 25）……")
    runpy.run_path(str(g5), run_name="__main__")
    try:
        data = __import__("json").loads(out.read_text(encoding="utf-8"))
        bad = [k for k, v in data.get("cases", {}).items()
               if float(v.get("valid_rate", 0)) < 0.95]
        if bad:
            print(f"[regression] 未达线套件: {bad} → 触发解析修复器/模板收紧预案（02 风险表）")
            return 1
    except Exception as exc:
        print(f"[regression] 结果读取失败: {exc}")
        return 1
    print("[regression] 全套件 ≥95% PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
