from __future__ import annotations

"""汇总评估报告：读取 results/eval/ 下的 eval_report_*.json，输出配置对比表。

用法:
    python tools/summarize_eval_reports.py [--eval-dir results/eval]
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_NAMES = {
    ("main", False, True): "主路径(PoseClassifier+上下文)",
    ("main", False, False): "消融:无场景上下文",
    ("baseline", True, True): "基线:SlowFast关键词映射",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总评估报告")
    parser.add_argument("--eval-dir", default=str(BASE_DIR / "results" / "eval"))
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    reports = sorted(eval_dir.glob("eval_report_*.json"))
    if not reports:
        print(f"[ERROR] 未找到评估报告: {eval_dir}")
        sys.exit(1)

    rows: list[dict] = []
    for path in reports:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = doc.get("config", {})
        if cfg.get("baseline"):
            name = "基线:SlowFast关键词映射"
        elif not cfg.get("use_scene_context", True):
            name = "消融:无场景上下文"
        else:
            name = "主路径(PoseClassifier+上下文)"
        m = doc.get("metrics", {})
        lat = doc.get("latency_ms", {})
        rows.append({
            "video": doc.get("video", path.stem),
            "config": name,
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "abnormal_recall_any": m.get("abnormal_recall_any"),
            "fnr": m.get("false_negative_rate"),
            "fpr": m.get("false_positive_rate"),
            "p95_ms": lat.get("p95"),
            "samples": doc.get("counts", {}).get("total"),
            "file": path.name,
        })

    # 按视频分组输出
    by_video: dict[str, list[dict]] = {}
    for row in rows:
        by_video.setdefault(row["video"], []).append(row)

    for video, group in by_video.items():
        print(f"\n### {video}（伪真值样本 {group[0]['samples']}）")
        header = f"{'配置':<28} {'准确率':>8} {'MacroF1':>8} {'异常检出':>8} {'漏报率':>8} {'误报率':>8} {'P95延迟':>9}"
        print(header)
        print("-" * len(header))
        for row in group:
            def fmt(v, pct=True):
                if v is None:
                    return "-"
                return f"{v * 100:.1f}%" if pct else f"{v:.0f}ms"
            print(
                f"{row['config']:<28} {fmt(row['accuracy']):>8} {fmt(row['macro_f1']):>8} "
                f"{fmt(row['abnormal_recall_any']):>8} {fmt(row['fnr']):>8} {fmt(row['fpr']):>8} "
                f"{fmt(row['p95_ms'], pct=False):>9}"
            )

    # 汇总 markdown 表（供方案文档直接引用）
    md_path = eval_dir / "eval_summary.md"
    lines = ["# 评估结果汇总（VLM 伪真值）", ""]
    lines.append("> 偏差声明：伪真值由 VLM 教师生成，与主路径训练数据同源，指标存在乐观偏差；326 帧人工抽检为独立锚点。")
    lines.append("")
    for video, group in by_video.items():
        lines.append(f"## {video}")
        lines.append("")
        lines.append("| 配置 | 准确率 | Macro-F1 | 异常检出率(any) | 漏报率 | 误报率 | P95延迟 |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in group:
            def pct(v):
                return f"{v * 100:.1f}%" if v is not None else "-"
            lines.append(
                f"| {row['config']} | {pct(row['accuracy'])} | {pct(row['macro_f1'])} | "
                f"{pct(row['abnormal_recall_any'])} | {pct(row['fnr'])} | {pct(row['fpr'])} | {row['p95_ms']}ms |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Markdown 汇总已写入: {md_path}")


if __name__ == "__main__":
    main()
