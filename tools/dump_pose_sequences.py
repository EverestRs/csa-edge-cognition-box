from __future__ import annotations

"""从真实视频提取原型类姿态序列（ROADMAP Stage 3 数据侧）。

对指定视频跑 检测+跟踪+ViTPose，取帧覆盖最多的 track，滑窗生成 32 帧姿态
序列（全帧像素坐标 + conf，与 FastPoseDataset 的 /1920,/1080 归一化约定配套），
输出 build_prototype_training_data 同格式的 class_NPZ。

用法:
  python tools/dump_pose_sequences.py --clip a.avi --clip b.avi \
      --label N01 --name 挥手 --proto proto_001 --out results/training_data/prototype_classes \
      [--stride 4] [--max-frames 150]
"""

import argparse
import json
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
T_DIM, V_DIM = 32, 133


def dump_clip(clip: Path, det: Detector, trk: Tracker, pose: PoseEstimator, max_frames: int):
    cap = cv2.VideoCapture(str(clip))
    fid = 0
    boxes: dict[str, dict[int, list[float]]] = {}
    while fid < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if fid % 2 == 0:
            for t in trk.update(det.infer(frame)):
                boxes.setdefault(t["track_id"], {})[fid] = t["bbox"]
        else:
            trk.advance_without_detections()
        fid += 1
    cap.release()
    if not boxes:
        return []
    main = max(boxes, key=lambda k: len(boxes[k]))
    per_frame = boxes[main]
    cap = cv2.VideoCapture(str(clip))
    fid = 0
    pose_seq: dict[int, np.ndarray] = {}
    while fid < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if fid in per_frame:
            x, y, w, h = [int(v) for v in per_frame[fid]]
            crop = frame[max(0, y):y + h, max(0, x):x + w]
            if crop.size:
                kps = pose.infer(crop, {"bbox": [x, y, w, h]})
                if kps and len(kps) >= 17:
                    arr = np.zeros((V_DIM, 3), np.float32)
                    for i, p in enumerate(kps[:17]):
                        arr[i] = [float(p[0]), float(p[1]), min(1.0, max(0.0, float(p[2])))]
                    pose_seq[fid] = arr
        fid += 1
    cap.release()
    fids = sorted(pose_seq)
    if len(fids) < T_DIM:
        return []
    windows = []
    for start in range(0, len(fids) - T_DIM + 1, 2):  # 滑窗步长2个姿态帧，尽量多样
        chunk = [pose_seq[f] for f in fids[start:start + T_DIM]]
        span = (fids[start + T_DIM - 1] - fids[start]) / 30.0
        if span < 0.4:  # 时间跨度过短的窗口（静止跳帧）跳过
            continue
        windows.append(np.stack(chunk))
    return windows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--proto", required=True)
    ap.add_argument("--out", default="results/training_data/prototype_classes")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--features", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    det_cfg = dict(cfg.get("detector", {}))
    if det_cfg.get("runtime"):
        det_cfg["runtime"]["execution_provider"] = "CPU"
    det = Detector(det_cfg)
    trk = Tracker(dict(cfg.get("tracker", {})))
    pose = PoseEstimator(cfg.get("pose", {}))

    all_windows = []
    for c in args.clip:
        wins = dump_clip(Path(c), det, trk, pose, args.max_frames)
        print(f"  {Path(c).name}: {len(wins)} 个窗口")
        all_windows.extend(wins)
    if len(all_windows) < 3:
        print("窗口不足 3，放弃")
        return 1
    arr = np.stack(all_windows).astype(np.float32)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / f"class_{args.label}.npz"
    np.savez_compressed(npz, pose_seq=arr)
    meta = {"classes": [{
        "label": args.label, "proto_id": args.proto, "name": args.name,
        "n_samples": int(arr.shape[0]), "npz": str(npz), "features": args.features,
        "source_clips": [Path(c).name for c in args.clip],
    }]}
    (out / "new_classes.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"落盘 {npz}: pose_seq {arr.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
