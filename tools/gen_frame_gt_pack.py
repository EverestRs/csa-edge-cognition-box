from __future__ import annotations

"""帧级人工金标候选包（ROADMAP Stage 0b）。

消除"教师=考官"评估循环的人工锚点：从视频抽帧跑检测+跟踪，生成人员裁剪
候选页供人工标注，产出与伪真值评估同 schema 的帧级 GT。

三步工作流（与 tools/gen_event_candidates.py 的 candidates/page/apply 同风格）:
  1. pack  python tools/gen_frame_gt_pack.py pack <video> [--interval 30]
     → results/frame_gt/<video名>/{crops/*.jpg, page.html, gt_template.json}
  2. 人工  打开 page.html 对照裁剪图，在 gt_template.json 的 "label" 字段填
     11 类代码（Normal/H01..S02），保存为 gt_filled.json
  3. apply python tools/gen_frame_gt_pack.py apply <video> gt_filled.json
     → 校验合法性 → gt.json（评估器输入）

注意：第 2 步是人工环节，无法自动化。
"""

import argparse
import json
import shutil
import sys
from html import escape
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.detector import Detector
from src.core.tracker import Tracker

VALID_LABELS = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"]
PAGE_CSS = """
body{font-family:system-ui,sans-serif;background:#1a1a2e;color:#eee;margin:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.card{background:#16213e;border-radius:8px;padding:6px;font-size:11px}
.card img{width:100%;border-radius:4px}
.tag{color:#9ad;word-break:break-all}
"""


def _detector() -> Detector:
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "model_config.yaml", encoding="utf-8"))
    det_cfg = dict(cfg.get("detector", {}))
    if det_cfg.get("runtime"):
        det_cfg["runtime"]["execution_provider"] = "CPU"
    return Detector(det_cfg)


def _crop_person(frame, bbox, expand=1.35):
    x, y, w, h = [float(v) for v in bbox]
    cx, cy = x + w / 2, y + h / 2
    ew, eh = w * expand, h * expand
    x1, y1 = max(0, int(cx - ew / 2)), max(0, int(cy - eh / 2))
    x2, y2 = int(cx + ew / 2), int(cy + eh / 2)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def run_pack(video: str, interval: int, max_frames: int) -> Path:
    out_dir = Path("results") / "frame_gt" / Path(video).stem
    crops_dir = out_dir / "crops"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    crops_dir.mkdir(parents=True)

    detector = _detector()
    tracker = Tracker({"match_threshold": 0.6})
    cap = cv2.VideoCapture(video)
    entries = []
    fid = 0
    while fid < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if fid % 2 == 0:  # 处理帧口径与 realtime 档一致（frame_stride≈2-4）
            tracks = tracker.update(detector.infer(frame))
        else:
            tracks = tracker.advance_without_detections()
        if fid % interval == 0:
            for t in tracks:
                crop = _crop_person(frame, t["bbox"])
                if crop is None or crop.shape[0] < 24 or crop.shape[1] < 24:
                    continue
                name = f"{fid:06d}_{t['track_id']}.jpg"
                cv2.imwrite(str(crops_dir / name), crop)
                entries.append({
                    "video": Path(video).name,
                    "frame_id": fid,
                    "track_id": t["track_id"],
                    "bbox": [float(v) for v in t["bbox"]],
                    "crop": f"crops/{name}",
                    "label": "",
                })
        fid += 1
    cap.release()

    template = out_dir / "gt_template.json"
    template.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")

    cards = []
    for e in entries:
        cards.append(
            f'<div class="card"><img src="{escape(e["crop"])}">'
            f'<div class="tag">{e["frame_id"]} / {e["track_id"]}</div>'
            f'<div>label: <b>____</b></div></div>'
        )
    (out_dir / "page.html").write_text(
        f"<html><head><meta charset='utf-8'><style>{PAGE_CSS}</style></head>"
        f"<body><h2>帧级金标候选 — {escape(Path(video).name)}</h2>"
        f"<p>共 {len(entries)} 个候选。对照下方裁剪图，在 gt_template.json 中按 "
        f"frame_id/track_id 填 label（{'/'.join(VALID_LABELS)}），填完另存为 gt_filled.json 后执行 apply。</p>"
        f"<div class='grid'>{''.join(cards)}</div></body></html>",
        encoding="utf-8",
    )
    print(f"候选包完成：{len(entries)} 个候选（每 {interval} 处理帧采样，共读 {fid} 帧）")
    print(f"  裁剪图: {crops_dir}")
    print(f"  候选页: {out_dir / 'page.html'}")
    print(f"  待填模板: {template}")
    return out_dir


def run_apply(video: str, gt_filled: str) -> Path:
    out_dir = Path("results") / "frame_gt" / Path(video).stem
    entries = json.loads(Path(gt_filled).read_text(encoding="utf-8"))
    bad = [e for e in entries if e.get("label") not in VALID_LABELS]
    if bad:
        raise SystemExit(f"{len(bad)} 条 label 非法（须为 {'/'.join(VALID_LABELS)}），例如 frame "
                         f"{bad[0].get('frame_id')} label={bad[0].get('label')!r}")
    gt_path = out_dir / "gt.json"
    gt_path.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    dist = Counter(e["label"] for e in entries)
    print(f"金标已落盘: {gt_path}（{len(entries)} 条）")
    print(f"  类别分布: {dict(sorted(dist.items()))}")
    return gt_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack", "apply"])
    ap.add_argument("video")
    ap.add_argument("gt_filled", nargs="?")
    ap.add_argument("--interval", type=int, default=30, help="每 N 个处理帧采样一次（C5 密度口径）")
    ap.add_argument("--frames", type=int, default=1500, help="最多读取的处理帧数")
    args = ap.parse_args()
    if args.cmd == "pack":
        run_pack(args.video, args.interval, args.frames)
    else:
        if not args.gt_filled:
            raise SystemExit("apply 需要 gt_filled.json 路径")
        run_apply(args.video, args.gt_filled)


if __name__ == "__main__":
    main()
