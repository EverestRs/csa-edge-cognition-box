from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parents[1]

TARGET_LABELS = {"H01", "H02", "H04"}
CLASSES = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"]
RISK_MAP = {
    "Normal": "LOW", "H01": "HIGH", "H02": "HIGH", "H03": "HIGH",
    "H04": "HIGH", "H05": "HIGH", "M01": "MEDIUM", "M02": "MEDIUM",
    "M03": "MEDIUM", "S01": "MEDIUM", "S02": "MEDIUM",
}
COLORS = {
    "H01": (0, 0, 255),   # 红
    "H02": (0, 165, 255),  # 橙
    "H04": (0, 0, 200),    # 暗红
}


def find_video(video_name: str, data_dir: Path) -> Path | None:
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".flv", ".m4v"):
        candidate = data_dir / video_name
        if not candidate.suffix:
            candidate = candidate.with_suffix(ext)
        if candidate.exists():
            return candidate
    for f in data_dir.iterdir():
        if f.is_file() and f.stem == Path(video_name).stem:
            return f
    return None


def extract_frames(
    video_path: Path,
    frame_ids: list[int],
    output_dir: Path,
    max_per_label: int = 30,
) -> list[dict]:
    """从视频中提取指定帧并保存为图片。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] 无法打开视频: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    saved = []
    target_set = set(frame_ids)
    current_frame = 0
    count = 0

    while True:
        ok, frame = cap.read()
        if not ok or current_frame > max(target_set):
            break

        if current_frame in target_set:
            out_path = output_dir / f"{video_path.stem}_frame{current_frame}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append({
                "frame_id": current_frame,
                "path": str(out_path),
                "timestamp": round(current_frame / fps, 2),
            })
            count += 1

        current_frame += 1

    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="高价值类人工抽检：提取 H01/H02/H04 样本帧")
    parser.add_argument("--data-dir", default=str(BASE_DIR.parent / "data"))
    parser.add_argument("--manifest", default=str(BASE_DIR / "results" / "training_data" / "manifest.json"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "results" / "manual_inspection"))
    parser.add_argument("--max-per-label", type=int, default=20, help="每个标签最多提取帧数")
    parser.add_argument("--labels", nargs="*", default=["H01", "H02", "H04"], help="要抽检的标签")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    target_labels = set(args.labels)
    print(f"[INFO] 目标标签: {sorted(target_labels)}")

    # 按视频和标签分组
    grouped: dict[str, dict[str, list[dict]]] = {}
    for item in manifest:
        label = item["refined_label"]
        if label not in target_labels:
            continue
        video = item["video"]
        if video not in grouped:
            grouped[video] = {}
        if label not in grouped[video]:
            grouped[video][label] = []
        grouped[video][label].append(item)

    # 统计
    total_samples = sum(len(samples) for labels in grouped.values() for samples in labels.values())
    print(f"[INFO] 找到 {total_samples} 个待抽检样本")

    # 创建子目录
    for label in target_labels:
        (output_dir / label).mkdir(parents=True, exist_ok=True)

    # 提取帧
    all_results: list[dict] = []
    for video_name in sorted(grouped.keys()):
        video_path = find_video(video_name, data_dir)
        if not video_path:
            print(f"  [WARN] 未找到视频: {video_name}")
            continue

        print(f"\n[VIDEO] {video_name}")
        for label in sorted(grouped[video_name].keys()):
            samples = grouped[video_name][label]
            # 按帧号排序，均匀采样
            samples.sort(key=lambda x: x["frame_id"])
            step = max(1, len(samples) // args.max_per_label)
            selected = samples[::step][:args.max_per_label]
            frame_ids = [s["frame_id"] for s in selected]

            print(f"  [{label}] {len(samples)} 样本，选取 {len(frame_ids)} 帧提取")
            saved = extract_frames(video_path, frame_ids, output_dir / label, args.max_per_label)

            # 保存元数据
            for s in saved:
                orig = next((x for x in samples if x["frame_id"] == s["frame_id"]), None)
                if orig:
                    all_results.append({
                        "video": video_name,
                        "frame_id": s["frame_id"],
                        "timestamp": s["timestamp"],
                        "label": label,
                        "confidence": orig["refined_confidence"],
                        "source": orig["source"],
                        "rule_tags": orig.get("rule_tags", []),
                        "image_path": s["path"],
                    })

    # 保存抽检清单
    report_path = output_dir / "inspection_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 生成 HTML 预览
    html_path = output_dir / "inspection_preview.html"
    html_content = _build_html(all_results, target_labels, args.max_per_label)
    html_path.write_text(html_content, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"抽检完成！")
    print(f"  总提取帧数: {len(all_results)}")
    print(f"  报告: {report_path}")
    print(f"  预览: {html_path}")
    print(f"  图片目录: {output_dir}")
    print(f"\n请打开 inspection_preview.html 查看提取的帧图片")


def _build_html(results: list[dict], labels: set[str], max_per_label: int) -> str:
    rows = ""
    for label in sorted(labels):
        label_results = [r for r in results if r["label"] == label]
        for r in label_results[:max_per_label]:
            img_rel = Path(r["image_path"]).name
            color = COLORS.get(label, (128, 128, 128))
            color_hex = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
            risk = RISK_MAP.get(label, "UNKNOWN")
            tags = ", ".join(r["rule_tags"][:3]) if r["rule_tags"] else "-"
            rows += f"""
            <tr>
                <td><span style="background:{color_hex};color:white;padding:2px 8px;border-radius:4px;font-weight:bold;">{label}</span></td>
                <td>{risk}</td>
                <td>{r['video']}</td>
                <td>{r['frame_id']}</td>
                <td>{r['timestamp']}s</td>
                <td>{r['confidence']:.3f}</td>
                <td style="font-size:11px;color:#666;">{tags}</td>
                <td><img src="{label}/{img_rel}" width="320" style="border:1px solid #ddd;border-radius:4px;" /></td>
            </tr>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>高价值类人工抽检</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th {{ background: #f0f0f0; padding: 10px; text-align: left; border-bottom: 2px solid #ddd; }}
td {{ padding: 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
tr:hover {{ background: #fafafa; }}
.summary {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
</style></head><body>
<h1>🔍 高价值类人工抽检报告</h1>
<div class="summary">
    <p><strong>目标标签:</strong> {', '.join(sorted(labels))}</p>
    <p><strong>总帧数:</strong> {len(results)}</p>
    <p><strong>操作说明:</strong> 请逐行检查每张图片，确认标签是否正确。重点关注：打斗(H01)是否有肢体冲突、欺凌(H02)是否有单方施暴、高危行为(H04)是否有危险动作。</p>
</div>
<table>
<tr><th>标签</th><th>风险</th><th>视频</th><th>帧号</th><th>时间</th><th>置信度</th><th>触发规则</th><th>截图</th></tr>
{rows}
</table>
</body></html>"""


if __name__ == "__main__":
    main()