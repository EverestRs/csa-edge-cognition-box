from __future__ import annotations

"""当前管线训练数据重生成（根因 3 修复：训练/推理口径统一）。

背景：旧训练 NPZ（7-24）的坐标/conf 口径与当前引擎管线脱节（D1-D4 修复后未重生成），
导致部署正确模型后 PC 输入仍是分布外。本脚本用**当前管线**（检测+跟踪+ViTPose+归一化）
重新提取滑窗序列，并对每窗中心帧请 VLM 教师判定 11 类标签——数据与推理同口径，标签新鲜。

用法:
  python tools/regen_training_data.py [--videos all] [--step 60] [--max-frames 1400] \
      [--out results/training_data_v2] [--device cuda]
产出:
  <out>/npz/<video>.npz   {frame_id(窗中心), track_id, pose_seq(N,32,133,3), label, confidence}
  <out>/manifest.json     与 FastPoseDataset 兼容的样本清单
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.detector import Detector
from src.core.pose_estimator import PoseEstimator
from src.core.tracker import Tracker
from src.core.vlm_client import parse_json_loose, vlm_chat

ROOT = Path(__file__).resolve().parents[1]
T_DIM, V_DIM = 32, 133
CLASSES = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"]

LABEL_PROMPT = (
    "你是校园安防视频分析专家。以下是按时间顺序排列的三帧连续监控画面（间隔约0.3秒），"
    "请分析**红色框标注的主体**正在做的动作（若多帧中主体一致，以动作过程为准）。\n"
    "选项：Normal(正常行走/站立/坐姿), H01(打架斗殴), H02(校园霸凌), H03(跌倒/昏厥/躺倒在地), "
    "H04(疑似轻生/翻越高处), H05(破坏设施), M01(吸烟), M02(使用手机), M03(遮挡摄像头), "
    "S01(异常徘徊), S02(翻越围栏)。\n"
    "判定要领：手部靠近面部多为托腮/扶眼镜/喝水，不一定是吸烟；看手机（手持屏幕注视）是 M02。\n"
    '填空输出JSON（不要其他文字、不要数组）：{{"label": "<选项>", "confidence": <0到1>}}'
)


def label_center(vlm: dict, frames: list[np.ndarray]) -> tuple[str, float] | None:
    try:
        text = vlm_chat(
            base_url=vlm["url"], api_key=vlm["api_key"], model=vlm["model"],
            prompt=LABEL_PROMPT, frames=frames, max_tokens=48,
            timeout=60.0, max_image_px=512,
        )
        parsed = parse_json_loose(text) or {}
        label = str(parsed.get("label", "")).strip().upper()
        if label == "NORMAL":
            label = "Normal"  # 大小写归一（否则 conf 走回退分支被硬编码覆盖）
        for c in CLASSES:
            if label.startswith(c):
                conf = float(parsed.get("confidence", 0.0))
                return c, max(0.0, min(1.0, conf))
        import re
        m = re.search(r'"label"\s*:\s*"([A-Z]?\d{2}|Normal|NORMAL)', text or "")
        if m:
            lab = m.group(1).upper()
            return (lab if lab in CLASSES else "Normal"), 0.6
    except Exception:
        pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="all", help="逗号分隔的视频名过滤，all=全部")
    ap.add_argument("--step", type=int, default=60, help="采样步长（视频帧）")
    ap.add_argument("--max-frames", type=int, default=1400)
    ap.add_argument("--out", default="results/training_data_v2")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    vlm_cfg = yaml.safe_load((ROOT / "configs/vlm_config.yaml").read_text(encoding="utf-8"))
    vlm = {"url": vlm_cfg["vllm_url"], "api_key": "EMPTY", "model": vlm_cfg["model_name"]}
    model_cfg = yaml.safe_load((ROOT / "configs/model_config.yaml").read_text(encoding="utf-8"))
    det_cfg = dict(model_cfg.get("detector", {}))
    if det_cfg.get("runtime"):
        det_cfg["runtime"]["execution_provider"] = "CPU"  # 检测用 CPU，GPU 留给姿态+用户进程
    det = Detector(det_cfg)
    trk = Tracker(dict(model_cfg.get("tracker", {})))
    pose_cfg = dict(model_cfg.get("pose", {}))
    if args.device == "cuda":
        pose_cfg.setdefault("runtime", {})["execution_provider"] = "CUDA"
    pose = PoseEstimator(pose_cfg)

    data_dir = ROOT.parent / "data"  # data/ 在项目根
    videos = sorted(p.name for p in data_dir.glob("*.mp4"))
    if args.videos != "all":
        keep = set(args.videos.split(","))
        videos = [v for v in videos if v in keep]
    out_dir = Path(args.out)
    (out_dir / "npz").mkdir(parents=True, exist_ok=True)

    manifest = []
    t0 = time.time()
    for vi, video in enumerate(videos):
        cap = cv2.VideoCapture(str(data_dir / video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 处理帧步长 2（与在线口径一致），采样每 step 视频帧
        proc_step = 2
        fid = 0
        tracks: dict[str, dict] = {}
        buf: dict[str, list[np.ndarray]] = {}
        last_sampled: dict[str, int] = {}
        rows: list[dict] = []
        n_vlm = 0
        while fid < min(total, args.max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if fid % proc_step == 0:
                for t in trk.update(det.infer(frame)):
                    tracks[t["track_id"]] = {"bbox": t["bbox"]}
                    x, y, bw, bh = [int(v) for v in t["bbox"]]
                    crop = frame[max(0, y):y + bh, max(0, x):x + bw]
                    if crop.size:
                        kps = pose.infer(crop, {"bbox": [x, y, bw, bh]})
                        if kps and len(kps) >= 17:
                            arr = np.zeros((V_DIM, 3), np.float32)
                            for i, p in enumerate(kps[:V_DIM]):
                                arr[i] = [float(p[0]), float(p[1]), min(1.0, max(0.0, float(p[2])))]
                            buf.setdefault(t["track_id"], []).append(arr)
                            if len(buf[t["track_id"]]) > T_DIM:
                                buf[t["track_id"]].pop(0)
            else:
                trk.advance_without_detections()
            # 采样判定：buffer 满 + 距上次采样 ≥ step 视频帧
            for tid, b in buf.items():
                if len(b) < T_DIM:
                    continue
                if fid - last_sampled.get(tid, -10**9) < args.step:
                    continue
                last_sampled[tid] = fid
                seq = np.stack(b[-T_DIM:])
                center = fid - (T_DIM - 1)  # 窗口首帧附近
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, center - 8))
                f1 = cap.read()[1]
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, center))
                f2 = cap.read()[1]
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, center + 8))
                f3 = cap.read()[1]
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                trio = [f for f in (f1, f2, f3) if f is not None]
                # 在中心帧画面上画当前 track 的红框（教师聚焦主体）
                if trio and tid in tracks:
                    x, y, bw, bh = [int(v) for v in tracks[tid]["bbox"]]
                    for f in trio:
                        cv2.rectangle(f, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                r = label_center(vlm, trio) if trio else None
                n_vlm += 1
                if r is None:
                    continue
                label, conf = r
                if conf < 0.6:
                    continue
                rows.append({
                    "frame_id": center, "track_id": tid, "pose_seq": seq.astype(np.float32),
                    "label": label, "refined_confidence": conf,
                    "filename_label": video.rsplit("_", 1)[0].split("_")[0],
                })
            fid += 1
        cap.release()
        npz_path = out_dir / "npz" / f"{Path(video).stem}.npz"
        if rows:
            np.savez_compressed(
                npz_path,
                frame_id=np.array([r["frame_id"] for r in rows]),
                track_id=np.array([r["track_id"] for r in rows]),
                pose_seq=np.stack([r["pose_seq"] for r in rows]),
                label=np.array([r["label"] for r in rows]),
                refined_confidence=np.array([r["refined_confidence"] for r in rows]),
            )
            for r in rows:
                manifest.append({
                    "video": video, "frame_id": str(r["frame_id"]), "track_id": r["track_id"],
                    "filename_label": video.rsplit("_", 1)[0].split("_")[0],
                    "refined_label": r["label"], "refined_confidence": str(r["refined_confidence"]),
                    "source": "vlm_teacher_v2", "class_idx": str(CLASSES.index(r["label"])),
                })
        print(f"[{vi + 1}/{len(videos)}] {video}: {len(rows)} 样本（VLM {n_vlm} 次）"
              f" 累计耗时 {time.time() - t0:.0f}s", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    dist = Counter(m["refined_label"] for m in manifest)
    print(f"\n完成: {len(manifest)} 样本 → {out_dir / 'manifest.json'}")
    print(f"分布: {dict(sorted(dist.items(), key=lambda kv: -kv[1]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
