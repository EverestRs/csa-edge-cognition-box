from __future__ import annotations

"""H03 分歧帧 VLM 重标（WEAKNESS_AUDIT 复查#一-1/T2）。

manifest 中 filename_label=H03（录像脚本语义：laying 视频就是在拍躺地/跌倒）
而教师判 Normal 的 1546 条分歧帧，是"教师系统性漏标"的现成重标候选。

流程:
  1. 分歧帧按视频分组均匀采样（cap --max-samples）
  2. 每帧取 3 帧时序（f, f+8, f+16）送 VLM 判定"画面是否有人跌倒/躺倒在地（H03）"
     （填空式 prompt + 单对象约束，bnb 口径）
  3. accepted（true 且 conf≥0.6）→ 对其视频段跑 检测+跟踪+姿态 滑窗提取，
     窗口含 accepted 帧 → 标 H03，产出 class_H03_extra.npz（训练增量）
  4. 报告: 采样数/接受数/接受率 + 抽样人工核对清单

用法:
  python tools/relabel_h03.py [--max-samples 300] [--out results/training_data/h03_extra]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.vlm_client import parse_json_loose, vlm_chat

ROOT = Path(__file__).resolve().parents[1]
JUDGE_PROMPT = (
    "这是监控画面中按时间顺序的三帧。请判断画面中是否有人处于**跌倒后倒地/躺倒在地**的状态"
    "（H03：突然倒地且无法自行站起，或刻意躺卧在地面）。\n"
    "注意：坐在椅子上/凳子上不算；弯腰、下蹲不算；靠墙站立不算。\n"
    '填空输出JSON（不要其他文字）：{{"lying": <true或false>, "confidence": <0到1>}}'
)
T_DIM, V_DIM = 32, 133


def load_manifest() -> list[dict]:
    return json.loads((ROOT / "results/training_data/manifest.json").read_text(encoding="utf-8"))


def pick_disagreement(manifest: list[dict], max_samples: int, seed: int = 13) -> list[dict]:
    by_video = defaultdict(list)
    for s in manifest:
        if s.get("filename_label") == "H03" and s.get("refined_label") == "Normal":
            by_video[s["video"]].append(s)
    import random
    rng = random.Random(seed)
    per = max(1, max_samples // max(1, len(by_video)))
    picked = []
    for video in sorted(by_video):
        items = by_video[video][:]
        rng.shuffle(items)
        picked.extend(items[:per])
    return picked[:max_samples]


def judge_frames(vlm: dict, frames: list[np.ndarray]) -> tuple[bool, float] | None:
    try:
        text = vlm_chat(
            base_url=vlm["vllm_url"], api_key=vlm["vllm_api_key"], model=vlm["model_name"],
            prompt=JUDGE_PROMPT, frames=frames, max_tokens=48,
            timeout=vlm["max_latency_ms"] / 1000.0, max_image_px=512,
        )
        parsed = parse_json_loose(text) or {}
        if "lying" in parsed:
            return bool(parsed["lying"]), float(parsed.get("confidence", 0.0))
        import re
        m = re.search(r'"lying"\s*:\s*"?(true|false)"?', text or "", re.I)
        if m:
            return m.group(1).lower() == "true", 0.7
    except Exception:
        pass
    return None


def read_triplet(cap, fid: int, total: int) -> list[np.ndarray]:
    frames = []
    for f in (fid, min(total - 1, fid + 8), min(total - 1, fid + 16)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--min-accept", type=int, default=8, help="accepted 帧少于此数的视频不提取序列")
    ap.add_argument("--out", default="results/training_data/h03_extra")
    args = ap.parse_args()

    vlm_cfg = yaml.safe_load((ROOT / "configs" / "vlm_config.yaml").read_text(encoding="utf-8"))
    vlm = {"vllm_url": vlm_cfg["vllm_url"], "vllm_api_key": "EMPTY",
           "model_name": vlm_cfg["model_name"], "max_latency_ms": 60000}

    manifest = load_manifest()
    picked = pick_disagreement(manifest, args.max_samples)
    print(f"分歧帧候选: {sum(1 for s in manifest if s.get('filename_label')=='H03' and s.get('refined_label')=='Normal')} 条"
          f" → 采样 {len(picked)} 条")

    accepted: dict[str, list[int]] = defaultdict(list)
    judged = failed = 0
    video_cap: dict[str, cv2.VideoCapture] = {}
    try:
        for i, s in enumerate(picked):
            video = s["video"]
            if video not in video_cap:
                vp = ROOT / "data" / video
                if not vp.exists():
                    vp = ROOT.parent / "data" / video
                video_cap[video] = cv2.VideoCapture(str(vp))
            cap = video_cap[video]
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = read_triplet(cap, int(s["frame_id"]), total)
            if len(frames) < 2:
                failed += 1
                continue
            r = judge_frames(vlm, frames)
            judged += 1
            if r and r[0] and r[1] >= 0.6:
                accepted[video].append(int(s["frame_id"]))
            if (i + 1) % 50 == 0:
                print(f"  进度 {i + 1}/{len(picked)}: 接受 {sum(len(v) for v in accepted.values())}", flush=True)
    finally:
        for c in video_cap.values():
            c.release()
    n_acc = sum(len(v) for v in accepted.values())
    print(f"判定完成: judged={judged} failed={failed} accepted={n_acc}（接受率 {n_acc/max(1,judged):.0%}）")
    report = {"sampled": len(picked), "judged": judged, "accepted": n_acc,
              "accepted_rate": round(n_acc / max(1, judged), 3),
              "by_video": {k: len(v) for k, v in accepted.items()}}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "relabel_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    if n_acc < args.min_accept:
        print(f"accepted < {args.min_accept}，不提取序列（教师重标接受率过低，需人工核对报告）")
        return 1

    # accepted 视频段 → 姿态滑窗（复用 dump_pose_sequences 的检测+跟踪+姿态管线）
    sys.path.insert(0, str(ROOT))
    from tools.dump_pose_sequences import dump_clip  # noqa: E402
    model_cfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    det_cfg = dict(model_cfg.get("detector", {}))
    if det_cfg.get("runtime"):
        det_cfg["runtime"]["execution_provider"] = "CPU"
    from src.core.detector import Detector
    from src.core.tracker import Tracker
    from src.core.pose_estimator import PoseEstimator
    det = Detector(det_cfg)
    trk = Tracker(dict(model_cfg.get("tracker", {})))
    pose = PoseEstimator(model_cfg.get("pose", {}))

    seqs = []
    for video, fids in accepted.items():
        vp = ROOT / "data" / video
        if not vp.exists():
            vp = ROOT.parent / "data" / video
        wins = dump_clip(vp, det, trk, pose, max_frames=int(max(fids) + 64))
        for w in wins:
            seqs.append(w)  # 窗口来自含 accepted 帧的段，整窗标 H03
    if not seqs:
        print("未提取到序列")
        return 1
    arr = np.stack(seqs).astype(np.float32)
    np.savez_compressed(out_dir / "class_H03_extra.npz", pose_seq=arr)
    report["h03_sequences"] = int(arr.shape[0])
    (out_dir / "relabel_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"H03 增量: {arr.shape} → {out_dir / 'class_H03_extra.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
