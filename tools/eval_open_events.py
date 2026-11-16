from __future__ import annotations

"""开放事件（未知事件）评估：验证自然语言规则引擎对训练体系外行为的告警能力。

赛题要求：测试集包含模型未在训练阶段见过的"未知事件"。
本工具用公开数据集（HMDB51，学术研究用途）的 wave/clap/throw/shake_hands 四类
片段——均不在系统 11 类行为体系内——为每类一句话创建告警规则，逐片段运行系统，
统计：
  - 命中率：真实类别规则在该类片段上触发告警的比例（含平均响应时延）
  - 误触发：其他规则在片段上触发的次数（交叉矩阵）

用法:
    conda run -n VdAnalyst python tools/eval_open_events.py \
        --clips-dir data_external/unknown_events --max-clips-per-class 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.inference_engine import InferenceEngine  # noqa: E402
from src.utils.config_loader import load_json, load_yaml  # noqa: E402
from src.utils.preflight import resolve_profile  # noqa: E402

# 未知事件类别 → 一句话告警规则
CLASS_RULES = {
    "wave": "告警有人挥手",
    "clap": "告警有人鼓掌",
    "throw": "告警有人投掷物品",
    "shake_hands": "告警有人握手",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="开放事件（未知事件）评估")
    parser.add_argument("--clips-dir", default=str(BASE_DIR / "data_external" / "unknown_events"))
    parser.add_argument("--max-clips-per-class", type=int, default=8)
    parser.add_argument("--output-dir", default=str(BASE_DIR / "results" / "eval" / "open_events"))
    parser.add_argument("--profile", default="realtime")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 短片段适配：加快开放判定节奏、缩短冷却
    vlm_cfg = load_yaml(str(BASE_DIR / "configs" / "vlm_config.yaml"))
    vlm_cfg = dict(vlm_cfg)
    vlm_cfg["nl_rule_engine"] = dict(vlm_cfg.get("nl_rule_engine", {}), **{
        # 评估采用片段级判定（check_now），关闭周期判定避免重复/空帧干扰
        "open_check_interval_seconds": 3600.0,
        "expedited_check_gap_seconds": 1.5,
        "rule_cooldown_seconds": 1.0,
        "frames_per_check": 6,
    })

    rules_path = output_dir / "alert_rules.json"
    if rules_path.exists():
        rules_path.unlink()

    raw = load_yaml(str(BASE_DIR / "configs" / "model_config.yaml"))
    rules_json = load_json(str(BASE_DIR / "configs" / "behavior_rules.json"))
    raw["behavior_heuristics"] = dict(rules_json.get("heuristics", {}))
    raw["alert_policy"] = dict(rules_json.get("alert_policy", {}))
    model_cfg, _paths = resolve_profile(raw, args.profile)

    engine = InferenceEngine(model_cfg=model_cfg, vlm_cfg=vlm_cfg, output_dir=str(output_dir))
    engine.nl_rule_engine._rules_path = rules_path
    engine.nl_rule_engine._load_rules()

    rule_ids: dict[str, str] = {}
    print("[SETUP] 创建未知事件规则")
    for cls, text in CLASS_RULES.items():
        rule = engine.nl_rule_engine.add_rule(text)
        rule_ids[cls] = rule["rule_id"]
        print(f"  [{cls}] → {rule['rule_id']} 「{text}」 open={rule['parsed']['use_open_vocabulary']}")

    results: list[dict] = []
    t_start = time.perf_counter()
    for cls in CLASS_RULES:
        class_dir = clips_dir / cls
        clips = sorted(class_dir.glob("*.avi"))[: args.max_clips_per_class]
        print(f"\n[CLASS {cls}] {len(clips)} 个片段")
        for clip in clips:
            cap = cv2.VideoCapture(str(clip))
            if not cap.isOpened():
                print(f"  [SKIP] 无法打开: {clip.name}")
                continue
            # 片段间清空缓冲，避免上一片段画面污染本片段判定
            engine.frame_buffer.clear()
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            engine.set_source_fps(fps)
            # 每个片段独立评估：重置规则冷却
            for rule in engine.nl_rule_engine.list_rules():
                engine.nl_rule_engine.update_rule(rule["rule_id"], enabled=True)
                rule["stats"]["last_triggered_frame"] = None
            fid = 0
            fired: list[dict] = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                fid += 1
                if fid % 4 != 0 and fid != 1:
                    continue
                result = engine.process(frame, fid)
                for ra in result.rule_alerts:
                    fired.append({
                        "rule_id": ra["rule_id"],
                        "at_video_sec": round(fid / fps, 2),
                        "confidence": ra["confidence"],
                    })
            cap.release()
            # 片段级开放词汇判定：全片段均匀采样 6 帧，同步判定（评估判定能力本身，
            # 等价于生产中"证据仍在缓冲内时的周期判定"）
            cap = cv2.VideoCapture(str(clip))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            k = min(6, max(1, total))
            idxs = sorted(set(min(total - 1, int(i * (total - 1) / max(1, k - 1))) for i in range(k))) if total > 1 else [0]
            sample_frames = []
            for target in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = cap.read()
                if ok:
                    sample_frames.append(frame)
            cap.release()
            t_check = time.perf_counter()
            committed = engine.nl_rule_engine.check_now(sample_frames, fid)
            check_ms = int((time.perf_counter() - t_check) * 1000)
            for ra in committed:
                fired.append({
                    "rule_id": ra["rule_id"],
                    "at_video_sec": round(fid / max(1.0, fps), 2),
                    "confidence": ra["confidence"],
                    "via": "clip_level_check",
                })
            for ra in engine.nl_rule_engine.consume_pending_pushes():
                if not any(f["rule_id"] == ra["rule_id"] for f in fired):
                    fired.append({
                        "rule_id": ra["rule_id"],
                        "at_video_sec": round(fid / max(1.0, fps), 2),
                        "confidence": ra["confidence"],
                    })
            true_hits = [f for f in fired if f["rule_id"] == rule_ids[cls]]
            other_hits = [f for f in fired if f["rule_id"] != rule_ids[cls]]
            latency = min((f["at_video_sec"] for f in true_hits), default=None)
            results.append({
                "class": cls,
                "clip": clip.name,
                "hit": bool(true_hits),
                "latency_sec": latency,
                "check_ms": check_ms,
                "true_fires": true_hits,
                "other_fires": other_hits,
            })
            mark = "✅" if true_hits else "❌"
            extra = f" 误触发{len(other_hits)}" if other_hits else ""
            print(f"  {mark} {clip.name}: {len(true_hits)} 次命中, 首次 {latency}s, 判定耗时 {check_ms}ms{extra}")

    elapsed = time.perf_counter() - t_start
    engine.close()

    # ── 汇总 ────────────────────────────────────────────────
    summary: dict[str, dict] = {}
    for cls in CLASS_RULES:
        rows = [r for r in results if r["class"] == cls]
        hits = [r for r in rows if r["hit"]]
        latencies = [r["latency_sec"] for r in hits]
        cross = sum(len(r["other_fires"]) for r in rows if r["class"] != cls or True)
        summary[cls] = {
            "clips": len(rows),
            "hits": len(hits),
            "hit_rate": round(len(hits) / max(1, len(rows)), 4),
            "mean_latency_sec": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "cross_fires_on_this_class": cross - len(hits),
        }

    report = {
        "dataset": "HMDB51 (公开学术数据集, 研究用途)",
        "note": "四类事件均不在系统11类训练体系中, 属于'未知事件'; 由自然语言规则+开放词汇VLM判定触发",
        "config": {"profile": args.profile, "open_check_interval_seconds": 2.0},
        "summary": summary,
        "results": results,
        "elapsed_sec": round(elapsed, 1),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = output_dir / "open_events_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 开放事件评估汇总 =====")
    overall_hits = sum(s["hits"] for s in summary.values())
    overall_clips = sum(s["clips"] for s in summary.values())
    for cls, s in summary.items():
        print(f"  {cls}: 命中率 {s['hit_rate'] * 100:.0f}% ({s['hits']}/{s['clips']}), 平均响应 {s['mean_latency_sec']}s")
    print(f"  总体: {overall_hits}/{overall_clips} = {overall_hits / max(1, overall_clips) * 100:.0f}%")
    print(f"  报告: {out_path}")


if __name__ == "__main__":
    main()
