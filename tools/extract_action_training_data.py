from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.detector import Detector
from src.core.pose_estimator import PoseEstimator
from src.core.tracker import Tracker
from src.core.behavior_reasoner import BehaviorReasoner
from src.utils.config_loader import load_yaml

# 视频文件名 → 行为标签映射（弱标签）
FILENAME_LABEL_MAP: dict[str, str] = {
    "laying": "H03",
    "smoking": "M01",
}

# 项目 11 类
CLASSES = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def resolve_video_path(video_name: str, data_dir: Path) -> Path | None:
    """在 data_dir 中查找匹配的视频文件（支持扩展名）。"""
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".flv"):
        candidate = data_dir / video_name
        if not candidate.suffix:
            candidate = candidate.with_suffix(ext)
        if candidate.exists():
            return candidate
    # 模糊匹配
    for f in data_dir.iterdir():
        if f.is_file() and f.stem == Path(video_name).stem:
            return f
    return None


def infer_filename_label(video_path: Path) -> str | None:
    """从视频文件名推断行为标签。"""
    stem = video_path.stem.lower()
    for keyword, label in FILENAME_LABEL_MAP.items():
        if keyword in stem:
            return label
    return None


def extract_samples(
    video_path: Path,
    detector: Detector,
    pose_estimator: PoseEstimator,
    tracker: Tracker,
    reasoner: BehaviorReasoner,
    action_stride: int = 6,
    max_action_tracks: int = 2,
    input_clip_len: int = 32,
    progress_cb=None,
) -> list[dict[str, Any]]:
    """在单个视频上运行检测+姿态+追踪+规则推理，收集训练样本。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] 无法打开视频: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    filename_label = infer_filename_label(video_path)
    print(f"  [INFO] 视频: {video_path.name} 帧数={total_frames} fps={fps:.1f} 文件名标签={filename_label}")

    samples: list[dict[str, Any]] = []
    # 跟踪每个 track 的 pose 缓冲
    pose_buffers: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=input_clip_len))
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        # 1. 检测
        detections_raw = detector.infer(frame)
        tracks = tracker.update(detections_raw)

        # 2. 按 score 排序，取前 max_action_tracks 个 track 做姿态
        sorted_tracks = sorted(tracks, key=lambda t: float(t.get("score", 0.0)), reverse=True)
        pose_tracks = sorted_tracks[:max_action_tracks]

        # 3. 姿态估计
        pose_results = pose_estimator.infer_batch(frame, pose_tracks)

        for track, keypoints in zip(pose_tracks, pose_results):
            track_id = track["track_id"]
            keypoints = keypoints or []

            # 追加 pose 到缓冲
            if keypoints:
                arr = np.array(keypoints, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    # 确保 3 列 (x, y, conf)
                    if arr.shape[1] < 3:
                        conf = np.ones((arr.shape[0], 1), dtype=np.float32)
                        arr = np.concatenate([arr, conf], axis=1)
                    arr = arr[:, :3]
                else:
                    arr = np.zeros((17, 3), dtype=np.float32)
            else:
                arr = np.zeros((17, 3), dtype=np.float32)
            pose_buffers[track_id].append(arr)

            # 4. 在 action stride 上运行规则推理
            if (frame_count % action_stride) == 0 and len(pose_buffers[track_id]) >= input_clip_len:
                bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]

                # 用规则引擎推理（不依赖 action model）
                refined_label, refined_conf, source, rule_tags = reasoner.infer(
                    track_id=track_id,
                    bbox=bbox,
                    keypoints=keypoints,
                    frame_shape=frame.shape[:2],
                    base_label="Normal",  # 不给 action model 输出，让规则引擎自主判断
                    base_confidence=0.0,
                    peer_tracks=tracks,
                )

                # 只保存规则引擎有明确判断的样本（非 Normal）
                # 同时保存 Normal 样本作为负样本（采样率 1/10 控制数量）
                is_normal = refined_label == "Normal"
                if not is_normal or (frame_count % (action_stride * 10)) == 0:
                    # 提取 pose 序列
                    pose_seq = np.stack(list(pose_buffers[track_id]), axis=0)  # (T, V, 3)

                    sample = {
                        "video": video_path.name,
                        "frame_id": frame_count,
                        "track_id": track_id,
                        "filename_label": filename_label or "",
                        "refined_label": refined_label,
                        "refined_confidence": round(float(refined_conf), 4),
                        "source": source,
                        "rule_tags": rule_tags,
                        "pose_seq": pose_seq.astype(np.float16),  # 节省存储
                        "bbox": bbox,
                    }
                    samples.append(sample)

        if progress_cb and frame_count % 100 == 0:
            progress_cb(frame_count, total_frames)

    cap.release()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="从视频中提取动作训练数据（姿态序列 + 规则引擎标签）")
    parser.add_argument("--data-dir", default=str(BASE_DIR.parent / "data"), help="视频数据目录")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "results" / "training_data"), help="输出目录")
    parser.add_argument("--model-config", default=str(BASE_DIR / "configs" / "model_config.yaml"), help="模型配置")
    parser.add_argument("--action-stride", type=int, default=6, help="动作采样的帧间隔")
    parser.add_argument("--max-tracks", type=int, default=2, help="每帧最多处理的 track 数")
    parser.add_argument("--clip-len", type=int, default=32, help="输入片段长度（帧数）")
    parser.add_argument("--videos", nargs="*", default=None, help="指定要处理的视频名（不含路径），默认处理所有")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置
    model_cfg = load_yaml(args.model_config)
    runtime_cfg = dict(model_cfg.get("runtime", {}))
    behavior_cfg = dict(model_cfg.get("behavior_heuristics", {})) or load_yaml(
        str(BASE_DIR / "configs" / "behavior_rules.json")
    )

    # 初始化模型（只初始化需要的组件）
    print("[INFO] 初始化 Detector...")
    detector_cfg = dict(model_cfg.get("detector", {}))
    detector_cfg["runtime"] = runtime_cfg
    detector = Detector(detector_cfg)

    print("[INFO] 初始化 PoseEstimator...")
    pose_cfg = dict(model_cfg.get("pose", {}))
    pose_cfg["runtime"] = runtime_cfg
    pose_estimator = PoseEstimator(pose_cfg)

    print("[INFO] 初始化 Tracker...")
    tracker = Tracker(model_cfg.get("tracker", {}))

    print("[INFO] 初始化 BehaviorReasoner...")
    reasoner = BehaviorReasoner(behavior_cfg)

    # 收集视频文件
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}
    if args.videos:
        video_files = []
        for name in args.videos:
            found = resolve_video_path(name, data_dir)
            if found:
                video_files.append(found)
            else:
                print(f"  [WARN] 未找到视频: {name}")
    else:
        video_files = sorted(
            [f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
        )

    if not video_files:
        print(f"[ERROR] data_dir 中没有找到视频文件: {data_dir}")
        sys.exit(1)

    print(f"[INFO] 找到 {len(video_files)} 个视频")

    all_samples: list[dict[str, Any]] = []
    label_dist: dict[str, int] = {}

    for video_idx, video_path in enumerate(video_files):
        print(f"\n[{video_idx + 1}/{len(video_files)}] 处理: {video_path.name}")

        def progress_cb(current, total):
            pct = current / max(1, total) * 100
            print(f"   进度: {current}/{total} ({pct:.1f}%)", end="\r")

        samples = extract_samples(
            video_path=video_path,
            detector=detector,
            pose_estimator=pose_estimator,
            tracker=tracker,
            reasoner=reasoner,
            action_stride=args.action_stride,
            max_action_tracks=args.max_tracks,
            input_clip_len=args.clip_len,
            progress_cb=progress_cb,
        )
        print(f"   提取样本数: {len(samples)}")

        for s in samples:
            label = s["refined_label"]
            label_dist[label] = label_dist.get(label, 0) + 1

        all_samples.extend(samples)

        # 每个视频单独保存一份（方便调试）
        video_out = output_dir / f"{video_path.stem}.npz"
        video_samples = [s for s in samples]

        # 保存为 NPZ（只保存轻量字段）
        if video_samples:
            save_dict = {
                "video": np.array([s["video"] for s in video_samples], dtype=object),
                "frame_id": np.array([s["frame_id"] for s in video_samples], dtype=np.int32),
                "track_id": np.array([s["track_id"] for s in video_samples], dtype=object),
                "filename_label": np.array([s["filename_label"] for s in video_samples], dtype=object),
                "refined_label": np.array([s["refined_label"] for s in video_samples], dtype=object),
                "refined_confidence": np.array([s["refined_confidence"] for s in video_samples], dtype=np.float32),
                "source": np.array([s["source"] for s in video_samples], dtype=object),
                "pose_seq": np.stack([s["pose_seq"] for s in video_samples], axis=0),  # (N, T, V, 3)
            }
            np.savez_compressed(str(video_out), **save_dict)
            print(f"   已保存: {video_out} ({len(video_samples)} 样本)")

    # 整体统计
    print(f"\n{'='*60}")
    print(f"提取完成！总样本数: {len(all_samples)}")
    print(f"标签分布:")
    for label in sorted(label_dist.keys()):
        count = label_dist[label]
        pct = count / max(1, len(all_samples)) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    # 保存整体数据集索引
    index = {
        "total_samples": len(all_samples),
        "label_distribution": label_dist,
        "videos_processed": [f.name for f in video_files],
        "classes": CLASSES,
        "config": {
            "action_stride": args.action_stride,
            "max_tracks": args.max_tracks,
            "clip_len": args.clip_len,
        },
    }
    index_path = output_dir / "dataset_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"数据集索引: {index_path}")

    # 保存合并后的完整数据集（仅标签元数据，pose 序列在各自 npz 中）
    manifest = []
    for s in all_samples:
        manifest.append({
            "video": s["video"],
            "frame_id": s["frame_id"],
            "track_id": s["track_id"],
            "filename_label": s["filename_label"],
            "refined_label": s["refined_label"],
            "refined_confidence": s["refined_confidence"],
            "source": s["source"],
            "rule_tags": s["rule_tags"],
            "class_idx": CLASS_TO_IDX.get(s["refined_label"], 0),
        })
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"样本清单: {manifest_path}")

    print(f"\n[OK] 数据提取完成。运行 finetune_action_model.py 开始训练。")


if __name__ == "__main__":
    main()