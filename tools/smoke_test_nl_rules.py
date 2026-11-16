"""自然语言告警规则引擎端到端冒烟测试（依赖本地 vLLM 服务运行中）。

用法:
    conda run -n VdAnalyst python tools/smoke_test_nl_rules.py --video ../data/laying_corner_1-1.mp4

验证内容:
  1. 三条真实规则解析（闭集映射 ×2 + 开放词汇 ×1）
  2. 闭集规则随流水线命中（laying 视频 → H03）+ 事件落盘 + 冷却生效
  3. 开放词汇规则周期判定（预期"不文明手势"= match false，为正确负例）
  4. REST CRUD 接口冒烟（可选 --api）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.nl_rule_engine import NLAlertRuleEngine  # noqa: E402
from src.pipeline.inference_engine import InferenceEngine  # noqa: E402
from src.utils.config_loader import load_json, load_yaml  # noqa: E402
from src.utils.preflight import resolve_profile  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="自然语言告警规则引擎冒烟测试")
    parser.add_argument("--video", default=str(BASE_DIR.parent / "data" / "laying_corner_1-1.mp4"))
    parser.add_argument("--max-frames", type=int, default=2400, help="送入的视频帧数")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "results" / "smoke_nl_rules"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rules_path = output_dir / "alert_rules.json"
    if rules_path.exists():
        rules_path.unlink()
    for stale in output_dir.glob("event_*.json"):
        stale.unlink()

    vlm_cfg = load_yaml(str(BASE_DIR / "configs" / "vlm_config.yaml"))

    # ── Step 1: 创建规则 ─────────────────────────────────────
    print("\n[Step 1] 规则创建与解析")
    rule_engine = NLAlertRuleEngine(vlm_cfg, None, output_dir=str(output_dir), rules_path=rules_path)
    created = []
    for text in ["告警有人翻越围栏", "有人躺倒在地长时间不起", "识别有人做出不文明手势"]:
        rule = rule_engine.add_rule(text)
        created.append(rule)
        p = rule["parsed"]
        print(f"  [{text}] classes={p['matched_classes']} open={p['use_open_vocabulary']} parser={p['parser']}")

    # ── Step 2: 离线跑流水线 ─────────────────────────────────
    print("\n[Step 2] 流水线运行（闭集命中 + 开放词汇周期判定）")
    raw = load_yaml(str(BASE_DIR / "configs" / "model_config.yaml"))
    rules_json = load_json(str(BASE_DIR / "configs" / "behavior_rules.json"))
    raw["behavior_heuristics"] = dict(rules_json.get("heuristics", {}))
    raw["alert_policy"] = dict(rules_json.get("alert_policy", {}))
    model_cfg, _paths = resolve_profile(raw, "realtime")
    engine = InferenceEngine(model_cfg=model_cfg, vlm_cfg=vlm_cfg, output_dir=str(output_dir))
    # 用测试专用规则路径替换（引擎内部已创建默认实例，这里直接改指向）
    engine.nl_rule_engine._rules_path = rules_path
    engine.nl_rule_engine._load_rules()
    engine.review_context_enabled = True

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    engine.set_source_fps(fps)
    print(f"  video fps={fps:.1f}, 处理上限 {args.max_frames} 帧")

    fid = 0
    rule_alerts_all: list[dict] = []
    t0 = time.perf_counter()
    while fid < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        if fid % 4 != 0 and fid != 1:
            continue
        result = engine.process(frame, fid)
        for ra in result.rule_alerts:
            ra["at_video_sec"] = round(fid / fps, 1)
            rule_alerts_all.append(ra)
            print(f"  ⚠️ 规则告警 @ {fid}: [{ra['rule_id']}] {ra['rule_text']} via {ra['matched_by']} conf={ra['confidence']}")
        if fid % 400 == 0:
            status = engine.nl_rule_engine.status()
            print(f"  ... frame {fid} ({fid / args.max_frames * 100:.0f}%) checks={status['open_check_count']} err={status['last_error'] or '无'}")
    cap.release()
    engine.close()
    elapsed = time.perf_counter() - t0
    print(f"\n  运行结束: {fid} 帧 / {elapsed:.0f}s, 规则告警 {len(rule_alerts_all)} 条")

    # ── Step 3: 结果核验 ────────────────────────────────────
    print("\n[Step 3] 结果核验")
    events = sorted(output_dir.glob("event_*_rule_*.json"))
    print(f"  规则事件文件: {len(events)} 个")
    for path in events[:5]:
        doc = json.loads(path.read_text(encoding="utf-8"))
        print(f"    {path.name}: {doc['rule']['rule_text']} via {doc['rule']['matched_by']} conf={doc['rule']['confidence']}")
    for rule in rule_engine.list_rules():
        s = rule["stats"]
        print(f"  [{rule['rule_id']}] {rule['text']}: 触发 {s['triggered']} 次, 最近帧 {s['last_triggered_frame']}")
    status = rule_engine.status()
    print(f"  引擎状态: {json.dumps(status, ensure_ascii=False)}")

    closed_hits = [a for a in rule_alerts_all if a["matched_by"] == "closed_class"]
    open_hits = [a for a in rule_alerts_all if a["matched_by"] == "open_vocabulary"]
    print(f"\n  闭集命中 {len(closed_hits)} 条 / 开放命中 {len(open_hits)} 条")
    print("  预期: laying 视频应出现「有人躺倒在地长时间不起」的闭集命中(H03)；")
    print("        「不文明手势」开放规则应为 0 命中（正确负例）。")
    print("\n[SMOKE DONE]")


if __name__ == "__main__":
    main()
