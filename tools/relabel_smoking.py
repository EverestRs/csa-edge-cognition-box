from __future__ import annotations
"""吸烟专用重标脚本：用吸烟专项 prompt 重标 4 个 smoking 视频。

与通用标注的区别：
- prompt 显式描述香烟视觉特征（手持香烟/烟雾/靠近嘴部）
- 输入用 person crop（非全场景），提高手部/香烟可见度
- 输出保留 reasoning 供人工抽检
"""

import json
import sys
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.detector import Detector
from src.core.tracker import Tracker
from src.core.vlm_reviewer import VLMReviewer
from src.utils.config_loader import load_yaml

SMOKE_PROMPT = """你是专业的安防视频分析师。请仔细观察画面中被红色框标注的人员，判断该人员是否正在吸烟。

重点观察：
1. 手指间是否夹有香烟（白色/棕色细长物体）
2. 是否有烟雾从嘴部或手中冒出
3. 是否有将手持物反复靠近嘴部的动作
4. 是否有打火机点火的动作

行为类别：
- M01: 吸烟（满足以上任意一条）
- Normal: 正常行为（无上述任何吸烟特征，包括手靠近脸但不持烟的情况如揉眼、擦汗、思考等）

注意：如果不确定或看不清手中的物体，请判定为 Normal。
仅输出JSON（不要其他文字）：{"label": "M01或Normal", "confidence": 0.0到1.0, "reasoning": "简要描述依据"}"""

SMOKING_VIDEOS = [
    "smoking_backward_1-1.mp4",
    "smoking_corner_1-1.mp4",
    "smoking_entrance_1-1.mp4",
    "smoking_forward_1-1.mp4",
]


def extract_and_label(
    video_path: Path,
    detector: Detector,
    tracker: Tracker,
    reviewer: VLMReviewer,
    sample_interval: int = 30,
    max_tracks: int = 2,
    workers: int = 4,
) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] 无法打开: {video_path.name}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"  {video_path.name}: {total} 帧, {fps:.0f} fps, 采样间隔={sample_interval}")

    frame_buffer: deque = deque(maxlen=3)
    fid_buffer: deque = deque(maxlen=3)
    samples = []
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        frame_buffer.append(frame.copy())
        fid_buffer.append(frame_count)

        if frame_count % sample_interval != 0:
            continue

        detections_raw = detector.infer(frame)
        tracks = tracker.update(detections_raw)
        sorted_tracks = sorted(tracks, key=lambda t: float(t.get("score", 0)), reverse=True)

        for track in sorted_tracks[:max_tracks]:
            tid = track["track_id"]
            bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
            x, y, w, h = [int(round(v)) for v in bbox]
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(fw, x + w), min(fh, y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            crops = []
            for bf in frame_buffer:
                c = bf[y1:y2, x1:x2].copy()
                if c.size > 0:
                    crops.append(c)
            if not crops:
                continue

            samples.append({
                "video": video_path.name,
                "frame_id": frame_count,
                "track_id": tid,
                "bbox": bbox,
                "crops": crops,
            })
    cap.release()
    print(f"  提取 {len(samples)} 个样本")

    def label_one(s):
        result = reviewer.review_clip(frames=s["crops"], track_id=s["track_id"], prompt=SMOKE_PROMPT)
        return {
            "video": s["video"],
            "frame_id": s["frame_id"],
            "track_id": s["track_id"],
            "bbox": s["bbox"],
            "label": result.get("label", "Normal"),
            "confidence": round(float(result.get("confidence", 0)), 4),
            "reasoning": result.get("reasoning", ""),
            "vlm_status": result.get("status", "error"),
            "frames_count": len(s["crops"]),
        }

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(label_one, s): s for s in samples}
        done = 0
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if r:
                    results.append(r)
            except Exception:
                pass
            done += 1
            if done % 20 == 0 or done == total:
                print(f"  [{done}/{total}] 已标注", flush=True)

    results.sort(key=lambda r: (r["video"], r["frame_id"]))
    return results


def main():
    output_dir = BASE_DIR / "results" / "vlm_smoking_relabel"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_yaml(str(BASE_DIR / "configs" / "model_config.yaml"))
    vlm_cfg = load_yaml(str(BASE_DIR / "configs" / "vlm_config.yaml"))
    vlm_cfg["use_vllm"] = True
    vlm_cfg["enabled"] = True

    detector_cfg = dict(model_cfg.get("detector", {}))
    detector_cfg["runtime"] = dict(model_cfg.get("runtime", {}))
    detector = Detector(detector_cfg)
    tracker = Tracker(model_cfg.get("tracker", {}))
    reviewer = VLMReviewer(vlm_cfg)

    data_dir = BASE_DIR.parent / "data"
    all_labels = []
    for video_name in SMOKING_VIDEOS:
        vp = data_dir / video_name
        if not vp.exists():
            print(f"[SKIP] {vp} 不存在")
            continue
        print(f"\n{'='*50}\n[PROCESS] {video_name}")
        samples = extract_and_label(vp, detector, tracker, reviewer, sample_interval=30, max_tracks=2, workers=4)
        all_labels.extend(samples)
        out = output_dir / f"{vp.stem}_relabel.json"
        out.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        from collections import Counter
        dist = Counter(r["label"] for r in samples)
        print(f"  → 保存: {out} | 分布: {dict(dist)}")

    all_out = output_dir / "all_relabel.json"
    all_out.write_text(json.dumps(all_labels, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    dist = Counter(r["label"] for r in all_labels)
    print(f"\n[DONE] 总计 {len(all_labels)} 样本 | 分布: {dict(dist)}")


if __name__ == "__main__":
    main()
