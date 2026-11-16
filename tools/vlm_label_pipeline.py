from __future__ import annotations

"""VLM 批量标注管道：用 Qwen2.5-VL / Qwen3-VL 为未标注视频生成伪标签。

使用方法:
    # 方式一：通过 vLLM API 批量标注（推荐，高吞吐）
    python tools/vlm_label_pipeline.py --vllm-url http://localhost:8001/v1/chat/completions \\
        --data-dir ../data --output-dir results/vlm_labels

    # 方式二：本地 transformers 直接推理（无需 vLLM 服务）
    python tools/vlm_label_pipeline.py --model-path models/vlm/qwen3-vl-8b-instruct \\
        --data-dir ../data --output-dir results/vlm_labels
"""

import argparse
import json
import sys
from collections import defaultdict
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
from src.core.vlm_reviewer import VLMReviewer
from src.utils.config_loader import load_yaml


def resolve_video_dir(data_dir: str) -> Path:
    path = Path(data_dir)
    if not path.exists():
        # 尝试相对项目根目录
        candidate = BASE_DIR.parent / data_dir
        if candidate.exists():
            return candidate
    return path


def collect_videos(data_dir: Path) -> list[Path]:
    """收集所有视频文件。"""
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    videos = sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    return videos


def extract_frames_for_vlm(
    video_path: Path,
    detector: Detector,
    pose_estimator: PoseEstimator,
    tracker: Tracker,
    sample_interval: int = 30,
    max_tracks: int = 2,
    clip_len: int = 3,
) -> list[dict[str, Any]]:
    """对单个视频提取检测、姿态，并生成 VLM 标注所需的帧裁剪。

    Args:
        video_path: 视频路径
        sample_interval: 每隔多少帧采样一次
        max_tracks: 每帧最多处理的 track 数
        clip_len: 时序上下文帧数（默认 3，取 t-1, t, t+1）

    Returns:
        list[dict]: 每项包含 video, frame_id, track_id, frames (3帧RGB), bbox, keypoints
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] 无法打开视频: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"  [INFO] {video_path.name}: {total_frames} 帧, {fps:.1f} fps, 采样间隔={sample_interval}")

    # 环形缓冲区，缓存最近 clip_len 帧用于时序上下文
    from collections import deque
    frame_buffer: deque[np.ndarray] = deque(maxlen=clip_len)
    # 帧索引缓冲
    frame_id_buffer: deque[int] = deque(maxlen=clip_len)

    samples: list[dict[str, Any]] = []
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        # 推入缓冲区
        frame_buffer.append(frame.copy())
        frame_id_buffer.append(frame_count)

        # 只在采样点处理
        if frame_count % sample_interval != 0:
            continue

        # 检测
        detections_raw = detector.infer(frame)
        tracks = tracker.update(detections_raw)

        # 按 score 排序，取前 max_tracks 个
        sorted_tracks = sorted(tracks, key=lambda t: float(t.get("score", 0.0)), reverse=True)
        target_tracks = sorted_tracks[:max_tracks]

        for track in target_tracks:
            track_id = track["track_id"]
            bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]

            # 裁剪 person crop
            x, y, w, h = [int(round(v)) for v in bbox]
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_frame, x + w)
            y2 = min(h_frame, y + h)

            # 收集 3 帧时序上下文（从缓冲区取）
            clip_frames: list[np.ndarray] = []
            clip_ids: list[int] = []
            buf_list = list(frame_buffer)
            buf_ids = list(frame_id_buffer)
            for i in range(len(buf_list)):
                f = buf_list[i]
                fid = buf_ids[i]
                crop = f[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else np.zeros((64, 64, 3), dtype=np.uint8)
                if crop.size == 0:
                    crop = np.zeros((64, 64, 3), dtype=np.uint8)
                clip_frames.append(crop)
                clip_ids.append(fid)

            samples.append({
                "video": video_path.name,
                "frame_id": frame_count,
                "track_id": track_id,
                "bbox": bbox,
                "frames": clip_frames,      # list[np.ndarray] RGB
                "frame_ids": clip_ids,
                "keypoints": [],             # 暂不存姿态，可后续补充
            })

        if frame_count % (sample_interval * 20) == 0:
            print(f"    ... 已处理到第 {frame_count}/{total_frames} 帧, 已收集 {len(samples)} 个样本")

    cap.release()
    return samples


def run_vlm_labeling(
    samples: list[dict[str, Any]],
    reviewer: VLMReviewer,
    output_dir: Path,
    batch_size: int = 8,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """对提取的样本运行 VLM 推理，生成伪标签。

    Args:
        workers: 并发请求数（vLLM 支持连续批处理，4-6 可显著提速）

    Returns:
        list[dict]: 包含 video, frame_id, track_id, label, confidence, reasoning
    """
    results: list[dict] = []
    total = len(samples)
    print(f"\n[VLM] 开始标注 {total} 个样本，并发={workers}")

    def label_one(index: int) -> dict:
        sample = samples[index]
        frames = sample["frames"]
        if not frames:
            return {}
        vlm_result = reviewer.review_clip(
            frames=frames,
            track_id=sample["track_id"],
        )
        return {
            "video": sample["video"],
            "frame_id": sample["frame_id"],
            "track_id": sample["track_id"],
            "bbox": sample["bbox"],
            "label": vlm_result.get("label", "Normal"),
            "confidence": round(float(vlm_result.get("confidence", 0.0)), 4),
            "reasoning": vlm_result.get("reasoning", ""),
            "vlm_status": vlm_result.get("status", "error"),
            "frames_count": len(frames),
        }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(label_one, i): i for i in range(total)}
        done_count = 0
        for future in as_completed(futures):
            try:
                item = future.result()
            except Exception as exc:
                item = {"video": samples[futures[future]]["video"],
                        "frame_id": samples[futures[future]]["frame_id"],
                        "track_id": samples[futures[future]]["track_id"],
                        "label": "Normal", "confidence": 0.0,
                        "reasoning": f"label_failed: {exc}", "vlm_status": "error"}
            if item:
                results.append(item)
            done_count += 1
            if done_count % 20 == 0 or done_count == total:
                print(f"  [{done_count}/{total}] 已标注, 当前标签分布: {_label_distribution(results)}")

    results.sort(key=lambda r: (r.get("video", ""), r.get("frame_id", 0), r.get("track_id", "")))
    return results


def _label_distribution(results: list[dict]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for r in results:
        counts[r.get("label", "Normal")] += 1
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def merge_with_pose_manifest(
    vlm_labels: list[dict],
    manifest_path: Path,
    output_path: Path,
    min_vlm_conf: float = 0.6,
) -> None:
    """将 VLM 伪标签与已有的 pose sequence manifest 合并。

    读取现有的 manifest.json（包含 pose_seq 路径和原始标签），
    用 VLM 标签覆盖 refined_label/class_idx，生成训练用 manifest。
    """
    if not manifest_path.exists():
        print(f"[WARN] 未找到 manifest: {manifest_path}，跳过合并")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # 构建 VLM 标签查找表: (video, frame_id, track_id) → label
    vlm_lookup: dict[str, str] = {}
    vlm_conf_lookup: dict[str, float] = {}
    for r in vlm_labels:
        if r.get("confidence", 0) >= min_vlm_conf:
            key = f"{r['video']}:{r['frame_id']}:{r['track_id']}"
            vlm_lookup[key] = r["label"]
            vlm_conf_lookup[key] = r["confidence"]

    # 更新 manifest
    updated = 0
    for item in manifest:
        key = f"{item['video']}:{item['frame_id']}:{item['track_id']}"
        if key in vlm_lookup:
            new_label = vlm_lookup[key]
            item["refined_label"] = new_label
            item["refined_confidence"] = vlm_conf_lookup[key]
            item["source"] = "vlm_teacher"
            item["rule_tags"] = ["vlm_label"]
            # 更新 class_idx
            from tools.finetune_action_model import CLASS_TO_IDX
            item["class_idx"] = CLASS_TO_IDX.get(new_label, 0)
            updated += 1

    print(f"[MERGE] 已更新 {updated}/{len(manifest)} 个样本的标签")
    with open(output_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[MERGE] 已保存合并后的 manifest: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM 批量标注管道")
    parser.add_argument("--data-dir", default=str(BASE_DIR.parent / "data"),
                        help="视频数据目录")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "results" / "vlm_labels"),
                        help="伪标签输出目录")
    parser.add_argument("--model-config", default=str(BASE_DIR / "configs" / "model_config.yaml"),
                        help="模型配置路径")
    parser.add_argument("--vlm-config", default=str(BASE_DIR / "configs" / "vlm_config.yaml"),
                        help="VLM 配置路径")
    parser.add_argument("--sample-interval", type=int, default=30,
                        help="采样帧间隔（默认 30，约 1fps）")
    parser.add_argument("--max-tracks", type=int, default=2,
                        help="每帧最多处理的 track 数")
    parser.add_argument("--min-vlm-conf", type=float, default=0.6,
                        help="VLM 伪标签最低置信度过滤")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="指定要处理的视频名，默认处理所有")
    parser.add_argument("--merge-manifest", action="store_true",
                        help="是否与已有 pose manifest 合并")
    parser.add_argument("--vllm-url", default="",
                        help="vLLM API URL（设置此值则使用 vLLM 而非本地推理）")
    parser.add_argument("--model-path", default="",
                        help="Qwen3-VL/Qwen2.5-VL 模型路径（本地推理时使用）")
    parser.add_argument("--workers", type=int, default=4,
                        help="VLM 标注并发请求数（vLLM 连续批处理，4-6 推荐）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置
    model_cfg = load_yaml(args.model_config)
    vlm_cfg = load_yaml(args.vlm_config)
    runtime_cfg = dict(model_cfg.get("runtime", {}))

    # 如果命令行指定了 model-path 或 vllm-url，覆盖配置
    if args.model_path:
        vlm_cfg["model_path"] = args.model_path
    if args.vllm_url:
        vlm_cfg["use_vllm"] = True
        vlm_cfg["vllm_url"] = args.vllm_url

    # 初始化检测/跟踪（用于提取 person crop）
    print("[INFO] 初始化 Detector...")
    detector_cfg = dict(model_cfg.get("detector", {}))
    detector_cfg["runtime"] = runtime_cfg
    detector = Detector(detector_cfg)

    print("[INFO] 初始化 Tracker...")
    tracker = Tracker(model_cfg.get("tracker", {}))

    # 初始化 VLMReviewer
    print(f"[INFO] 初始化 VLMReviewer (use_vllm={vlm_cfg.get('use_vllm', False)})...")
    vlm_cfg["enabled"] = True
    reviewer = VLMReviewer(vlm_cfg)

    # 收集视频
    data_dir = Path(args.data_dir)
    all_videos = collect_videos(data_dir)
    if args.videos:
        all_videos = [v for v in all_videos if v.name in args.videos]
    print(f"[INFO] 找到 {len(all_videos)} 个视频待处理")

    all_labels: list[dict] = []

    for video_path in all_videos:
        print(f"\n{'='*60}")
        print(f"[PROCESS] {video_path.name}")

        # 提取样本
        samples = extract_frames_for_vlm(
            video_path=video_path,
            detector=detector,
            pose_estimator=None,  # 不需要姿态，只需 person crop
            tracker=tracker,
            sample_interval=args.sample_interval,
            max_tracks=args.max_tracks,
            clip_len=3,
        )
        print(f"  -> 提取到 {len(samples)} 个样本")

        if not samples:
            continue

        # VLM 标注
        labels = run_vlm_labeling(
            samples=samples,
            reviewer=reviewer,
            output_dir=output_dir,
            batch_size=1,
            workers=args.workers,
        )
        all_labels.extend(labels)

        # 保存单个视频的标注结果
        video_out = output_dir / f"{video_path.stem}_vlm_labels.json"
        serializable = []
        for r in labels:
            item = {k: v for k, v in r.items() if k != "frames"}
            item.setdefault("frames_count", len(r.get("frames", [])))
            serializable.append(item)
        with open(video_out, "w") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"  -> 已保存: {video_out}")

    # 保存全量标注结果
    all_out = output_dir / "all_vlm_labels.json"
    serializable_all = []
    for r in all_labels:
        item = {k: v for k, v in r.items() if k != "frames"}
        item.setdefault("frames_count", len(r.get("frames", [])))
        serializable_all.append(item)
    with open(all_out, "w") as f:
        json.dump(serializable_all, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] 全量标注已保存: {all_out}")
    print(f"       总标注样本数: {len(all_labels)}")
    print(f"       标签分布: {_label_distribution(all_labels)}")

    # 可选：与已有 pose manifest 合并
    if args.merge_manifest:
        manifest_path = BASE_DIR / "results" / "training_data" / "manifest.json"
        merged_path = output_dir / "manifest_vlm_merged.json"
        merge_with_pose_manifest(
            all_labels, manifest_path, merged_path,
            min_vlm_conf=args.min_vlm_conf,
        )


if __name__ == "__main__":
    main()