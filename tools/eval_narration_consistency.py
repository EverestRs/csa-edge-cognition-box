from __future__ import annotations

"""叙述-检测一致率评估（WEAKNESS_AUDIT #8，C6 遗留指标）。

口径：检测模块报出的异常行为（detection_refs 中非 Normal 行）被 VLM 叙述文本
提及的比例——一致率 = mentioned / total_abnormal_refs。

  - 叙述文本优先取 vlm_summary_raw（#4 修复后记录的 VLM 原始输出，不含
    【检测联动·待核实】拼接行）；旧日志无该字段时，用正则剔除
    scene_summary 中的【检测联动…】段后计算，两种口径结果一致。
  - detection_refs 解析与 SceneNarrator._missing_abnormal_refs 同口径。
  - C6 约定：一致率 < 60% 时应启用 P2"行为时刻帧混采"（关键帧采样）。

用法:
  python tools/eval_narration_consistency.py                      # 评估 results/ 下全部任务
  python tools/eval_narration_consistency.py --task-id task_xxx   # 只评单个任务
  python tools/eval_narration_consistency.py --out path.json      # 指定报告输出路径
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# 【检测联动·待核实】…（该行为来自行为检测模型，本组画面帧中未获叙述确认）。→ 剔除
_LINKAGE_RE = re.compile(r"【检测联动[^】]*】[^。]*。?")

# 标签中文名 → 叙述常用同义词（确定性词表，保证离线可复现；词面匹配是保守下界）
_SYNONYMS: dict[str, list[str]] = {
    "打架斗殴": ["打架", "斗殴", "互殴", "挥拳", "打斗"],
    "校园霸凌": ["霸凌", "欺凌", "欺负", "殴打"],
    "跌倒/昏厥": ["跌倒", "摔倒", "倒地", "晕倒", "昏厥", "躺地"],
    "疑似轻生": ["轻生", "跳楼", "翻越栏杆", "自缢", "爬高"],
    "破坏设施": ["破坏", "砸", "踢", "踹"],
    "吸烟": ["吸烟", "抽烟", "香烟", "打火机"],
    "使用手机": ["手机", "打电话", "看手机", "玩手机"],
    "遮挡摄像头": ["遮挡", "挡住", "遮镜头"],
    "异常徘徊": ["徘徊", "踱步", "来回走", "逗留"],
    "翻越围栏/闯入禁区": ["翻越", "围栏", "闯入", "跨栏"],
}


def _mention_tokens(name: str) -> list[str]:
    """标签名 → 用于词面匹配的 token 集：同义词表优先，复合标签按 '/' 拆分。"""
    tokens = set(_SYNONYMS.get(name, []))
    tokens.update(part for part in name.split("/") if len(part) >= 2)
    if not tokens:
        tokens.add(name)
    return sorted(tokens)


def parse_abnormal_refs(detection_refs: str) -> list[tuple[str, str]]:
    """解析检测注入摘要，返回 [(track_id, 行为中文名)]，仅非 Normal 项。"""
    refs: list[tuple[str, str]] = []
    for line in (detection_refs or "").split("\n"):
        try:
            head, rest = line.split("：", 1)
            track_id = head.lstrip("- ").split("（")[0].strip()
            name = rest.split("（")[0].strip()
        except ValueError:
            continue
        if name and name not in ("正常", "Normal"):
            refs.append((track_id, name))
    return refs


def narration_text(record: dict[str, Any]) -> str:
    raw = str(record.get("vlm_summary_raw", "") or "")
    if not raw:
        raw = _LINKAGE_RE.sub("", str(record.get("scene_summary", "") or ""))
    activities = " ".join(str(a) for a in record.get("notable_activities", []) or [])
    return f"{raw} {activities}"


def evaluate_log(log_path: Path) -> dict[str, Any]:
    narrations = 0
    narrations_with_refs = 0
    refs_total = 0
    refs_mentioned = 0
    misses: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            narrations += 1
            refs = parse_abnormal_refs(str(record.get("detection_refs", "") or ""))
            if not refs:
                continue
            narrations_with_refs += 1
            text = narration_text(record)
            for track_id, name in refs:
                refs_total += 1
                if any(tok in text for tok in _mention_tokens(name)):
                    refs_mentioned += 1
                else:
                    misses.append({
                        "video_time_sec": record.get("video_time_sec"),
                        "frame_id": record.get("frame_id"),
                        "track_id": track_id,
                        "behavior": name,
                    })
    rate = (refs_mentioned / refs_total) if refs_total else None
    return {
        "log": str(log_path),
        "narrations": narrations,
        "narrations_with_abnormal_refs": narrations_with_refs,
        "abnormal_refs_total": refs_total,
        "abnormal_refs_mentioned": refs_mentioned,
        "consistency_rate": round(rate, 4) if rate is not None else None,
        "misses": misses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="叙述-检测一致率评估（scene_log 离线分析）")
    parser.add_argument("--results-dir", default="results", help="results 目录（含 scene_log/）")
    parser.add_argument("--task-id", default=None, help="只评估指定任务的日志")
    parser.add_argument("--out", default=None, help="报告 JSON 输出路径，默认 results/eval/narration_consistency_report.json")
    args = parser.parse_args()

    scene_log_dir = Path(args.results_dir) / "scene_log"
    if args.task_id:
        log_files = [scene_log_dir / f"scene_log_{args.task_id}.jsonl"]
    else:
        log_files = sorted(scene_log_dir.glob("scene_log*.jsonl"))
    log_files = [p for p in log_files if p.exists()]
    if not log_files:
        print(f"未找到场景叙述日志：{scene_log_dir}")
        return 1

    per_task = [evaluate_log(p) for p in log_files]
    total = sum(t["abnormal_refs_total"] for t in per_task)
    mentioned = sum(t["abnormal_refs_mentioned"] for t in per_task)
    overall = round(mentioned / total, 4) if total else None

    report = {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "overall": {
            "tasks": len(per_task),
            "abnormal_refs_total": total,
            "abnormal_refs_mentioned": mentioned,
            "consistency_rate": overall,
            "threshold_note": "C6 约定：一致率 < 0.60 时建议启用 P2 行为时刻帧混采",
        },
        "per_task": per_task,
    }

    out_path = Path(args.out) if args.out else Path(args.results_dir) / "eval" / "narration_consistency_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'任务':<40} 叙述数  异常引用  已提及  一致率")
    for t in per_task:
        rate = "—" if t["consistency_rate"] is None else f"{t['consistency_rate']:.1%}"
        name = Path(t["log"]).name
        print(f"{name:<40} {t['narrations']:>5}  {t['abnormal_refs_total']:>7}  "
              f"{t['abnormal_refs_mentioned']:>5}  {rate:>6}")
    overall_str = "—" if overall is None else f"{overall:.1%}"
    print(f"\n总体一致率: {overall_str}（{mentioned}/{total}）")
    print(f"报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
