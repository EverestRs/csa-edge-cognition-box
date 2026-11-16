from __future__ import annotations

"""将 VLM 教师标注转换为评估用伪真值 JSON。

用法:
    python tools/build_eval_ground_truth.py \
        --labels-dir results/vlm_labels_test \
        --output-dir results/eval/gt \
        --min-conf 0.6

输入: {video_stem}_vlm_labels.json（vlm_label_pipeline.py 产出）
输出: {video_stem}.json，格式:
    {"video": ..., "source": "vlm_teacher", "sample_interval": 60,
     "samples": [{"frame_id", "track_id", "label", "confidence"}]}
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

VALID_LABELS = {"Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"}


def build_gt(labels_file: Path, min_conf: float, sample_interval: int) -> dict:
    with open(labels_file, encoding="utf-8") as f:
        labels = json.load(f)

    samples = []
    dropped = 0
    for item in labels:
        label = str(item.get("label", "Normal"))
        if label not in VALID_LABELS:
            dropped += 1
            continue
        if float(item.get("confidence", 0.0)) < min_conf:
            dropped += 1
            continue
        if item.get("vlm_status") not in (None, "success"):
            dropped += 1
            continue
        samples.append({
            "frame_id": int(item["frame_id"]),
            "track_id": str(item["track_id"]),
            "label": label,
            "confidence": float(item["confidence"]),
            "bbox": [float(v) for v in item.get("bbox", [])],
        })

    samples.sort(key=lambda s: (s["frame_id"], s["track_id"]))
    from collections import Counter
    dist = Counter(s["label"] for s in samples)
    return {
        "video": labels[0].get("video", labels_file.stem) if labels else labels_file.stem,
        "source": "vlm_teacher",
        "labels_file": str(labels_file),
        "min_conf": min_conf,
        "sample_interval": sample_interval,
        "dropped_samples": dropped,
        "label_distribution": dict(dist),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构建评估伪真值")
    parser.add_argument("--labels-dir", default=str(BASE_DIR / "results" / "vlm_labels_test"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "results" / "eval" / "gt"))
    parser.add_argument("--min-conf", type=float, default=0.6)
    parser.add_argument("--sample-interval", type=int, default=60)
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(
        p for p in labels_dir.glob("*_vlm_labels.json") if p.name != "all_vlm_labels.json"
    )
    if not label_files:
        print(f"[ERROR] 未找到标注文件: {labels_dir}")
        sys.exit(1)

    for labels_file in label_files:
        gt = build_gt(labels_file, args.min_conf, args.sample_interval)
        out_path = output_dir / f"{Path(gt['video']).stem}.json"
        out_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"{out_path.name}: 样本 {len(gt['samples'])} (丢弃 {gt['dropped_samples']}) | "
            f"分布: {gt['label_distribution']}"
        )


if __name__ == "__main__":
    main()
