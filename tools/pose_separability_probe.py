from __future__ import annotations

"""姿态动力学可分性实验（ROADMAP Stage 2 姿态通道的前置验证）。

对 unknown_events 三段片段（wave1/wave2/throw）跑 检测+跟踪+ViTPose，
取每段帧覆盖最多的 track 的 32 帧姿态缓冲，提取动力学特征向量
（各身体组活动度 + 腕-肩距离序列统计矩），计算两两距离：
若 wave-wave 距离 < wave-throw 距离 → 姿态通道可分，值得接入原型库。
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.detector import Detector
from src.core.tracker import Tracker
from src.core.pose_estimator import PoseEstimator

ROOT = Path(__file__).resolve().parents[1]
UE = ROOT / "data_external" / "unknown_events"
GROUPS = {"头部": [0, 1, 2, 3, 4], "手臂": [5, 6, 7, 8, 9, 10], "下肢": [11, 12, 13, 14, 15, 16]}


def feature_for_clip(clip: Path) -> np.ndarray | None:
    cfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    det_cfg = dict(cfg.get("detector", {}))
    if det_cfg.get("runtime"):
        det_cfg["runtime"]["execution_provider"] = "CPU"
    detector = Detector(det_cfg)
    pose = PoseEstimator(cfg.get("pose", {}))
    tracker = Tracker(dict(cfg.get("tracker", {})))

    cap = cv2.VideoCapture(str(clip))
    fid = 0
    track_best = ""
    track_centers: dict[str, list[tuple[int, tuple[float, float]]]] = {}
    while fid < 120:
        ok, frame = cap.read()
        if not ok:
            break
        if fid % 2 == 0:
            tracks = tracker.update(detector.infer(frame))
            for t in tracks:
                if t["track_id"] not in track_centers:
                    track_centers[t["track_id"]] = []
                track_centers[t["track_id"]].append((fid, t["bbox"]))
        else:
            tracker.advance_without_detections()
        fid += 1
    cap.release()
    if not track_centers:
        return None
    track_best = max(track_centers, key=lambda k: len(track_centers[k]))
    boxes = dict(track_centers[track_best])
    print(f"  {clip.name[:40]}: 主 track={track_best} 覆盖 {len(boxes)} 处理帧")

    cap = cv2.VideoCapture(str(clip))
    pose_frames = []
    fid = 0
    while fid < 120:
        ok, frame = cap.read()
        if not ok:
            break
        if fid in boxes:
            x, y, w, h = [int(v) for v in boxes[fid]]
            crop = frame[max(0, y):y + h, max(0, x):x + w]
            if crop.size:
                kps = pose.infer(crop, {"bbox": [x, y, w, h]})
                if kps and len(kps) >= 17:
                    arr = np.array([[p[0], p[1], p[2]] for p in kps[:17]], np.float32)
                    diag = max(1.0, (w * w + h * h) ** 0.5)
                    arr[:, 0] /= diag
                    arr[:, 1] /= diag
                    pose_frames.append(arr)
        fid += 1
    cap.release()
    if len(pose_frames) < 8:
        print("    姿态帧不足:", len(pose_frames))
        return None
    arrs = pose_frames
    feats = []
    for g, idx in GROUPS.items():
        disp = [float(np.linalg.norm(b[idx, :2] - a[idx, :2], axis=1).mean()) for a, b in zip(arrs, arrs[1:])]
        feats.extend([np.mean(disp), np.std(disp)])
    # 腕-肩距离序列（右）：均值/ std / 峰次
    ws = [float(np.linalg.norm(b[10, :2] - b[6, :2])) for b in arrs]
    thr = np.mean(ws) + 0.5 * np.std(ws)
    peaks = sum(1 for i in range(1, len(ws) - 1) if ws[i] > thr and ws[i] >= ws[i - 1] and ws[i] >= ws[i + 1])
    feats.extend([np.mean(ws), np.std(ws), peaks / len(ws)])
    return np.array(feats, np.float32)


def main() -> int:
    clips = [("wave1", sorted(UE.glob("wave/*.avi"))[0]),
             ("wave2", sorted(UE.glob("wave/*.avi"))[1]),
             ("throw", sorted(UE.glob("throw/*.avi"))[0])]
    feats = {}
    for name, clip in clips:
        print(f"处理 {name}: {clip.name[:50]}")
        f = feature_for_clip(clip)
        if f is not None:
            feats[name] = f
    names = list(feats)
    print("\n两两欧氏距离（姿态动力学特征）:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = float(np.linalg.norm(feats[names[i]] - feats[names[j]]))
            print(f"  d({names[i]}, {names[j]}) = {d:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
