#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T2.5：crop 多分类器训练管道（build → train → export，另附 selftest）。

数据源（全部现成，无需新标注）：
  - 标签包: results/vlm_labels_v2/*.json、results/vlm_smoking_relabel/*.json
    （{video, frame_id, track_id, bbox:[x,y,w,h], label, confidence}——直接按 bbox 裁剪，
      与 VLM 教师所见一致；vlm_labels_test 为评估伪真值，默认排除）
  - HMDB51 演示类: data_external/hmdb51/shards/*.parquet（HF 格式，video bytes 内嵌）
    → 解码采样帧 → 中心裁剪（可选 --use-detector 走项目 Detector person 框）

预处理与推理严格对齐 edge_migration/crop_classifier._preprocess：
letterbox(pad=114) → BGR→RGB → /255，训练侧仅额外做翻转/色彩/仿射增强。

用法:
  python tools/train_crop_classifier.py build   --labels "results/vlm_labels_v2/*_vlm_labels.json" ...
  python tools/train_crop_classifier.py train   --data results/crop_dataset --epochs 30
  python tools/train_crop_classifier.py export  --ckpt results/crop_dataset/best.pth --out models/crop/crop_v1.onnx
  python tools/train_crop_classifier.py selftest
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CLASSES = ["Normal", "M01", "M02", "H03", "Wave", "Throw", "Handshake"]
SIZE = 224
PAD = 114  # 与 crop_classifier._preprocess 一致

# HMDB51 动作词表（文件名尾部段 + 多词子串双通道匹配；本地 HF 子集无 clap，用 handshake 替代）
_HMDB_ACTIONS = {"wave", "clap", "throw", "cartwheel", "climb", "fall_floor", "hug", "kiss",
                 "pick", "pour", "pushup", "ride_bike", "run", "shake_hands", "situp",
                 "smile", "smoke", "fencing", "golf", "brush_hair", "catch", "chew",
                 "dive", "drink", "eat", "fall", "hit", "jump", "kick", "pullup",
                 "punch", "push", "ride_horse", "shoot_ball", "somersault", "stand",
                 "swing_baseball", "sword", "turn", "handshake"}
_HMDB_MULTI = ("shake_hands", "fall_floor", "ride_bike", "brush_hair",
               "ride_horse", "shoot_ball", "swing_baseball")


def _hmdb_action_name(filename: str) -> str | None:
    name = filename.replace(".avi", "").lower()
    hit = next((s for s in reversed(name.split("_")) if s in _HMDB_ACTIONS), None)
    if hit:
        return hit
    return next((m for m in _HMDB_MULTI if m in name), None)


# --------------------------------------------------------------------------- #
# build：标签包 / HMDB51 → crop 图像 + manifest
# --------------------------------------------------------------------------- #
def letterbox_canvas(crop_bgr: np.ndarray, size: int = SIZE) -> np.ndarray:
    """与 edge_migration.crop_classifier._preprocess 相同的 letterbox（pad=114）。"""
    ch, cw = crop_bgr.shape[:2]
    scale = min(size / cw, size / ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD, dtype=np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _resolve_video(video_name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        p = root / video_name
        if p.exists():
            return p
        hit = list(root.rglob(video_name))[:1]
        if hit:
            return hit[0]
    return None


def build(args: argparse.Namespace) -> int:
    out = Path(args.out)
    crops_dir = out / "crops"
    if out.exists() and not args.append:
        shutil.rmtree(out)
    (crops_dir).mkdir(parents=True, exist_ok=True)
    classes = list(args.classes)
    video_roots = [Path(r) for r in args.video_root] + [ROOT.parent / "data", ROOT.parent / "test"]

    # 1) 收集标签 → 按视频分组（顺序读帧，避免逐条 seek）
    records: list[dict] = []
    for pattern in args.labels:
        for f in sorted(glob.glob(str(ROOT / pattern))):
            if any(x in f for x in args.exclude):
                continue
            records.extend(json.load(open(f, encoding="utf-8")))
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        lb = str(r.get("label", "Normal"))
        if lb not in classes:
            continue
        if float(r.get("confidence", 0) or 0) < args.min_conf:
            continue
        by_video[str(r["video"])].append(r)
    print(f"[build] 标签 {len(records)} 条 → 类内有效 {sum(len(v) for v in by_video.values())} 条"
          f"（classes={classes}, min_conf={args.min_conf}）")

    # 2) 跨视频全局类额底下采样（均匀步长选取，保持时间分布）
    quota = {c: args.max_per_class for c in classes}
    per_cls_rows: dict[str, list[dict]] = {c: [] for c in classes}
    for rows in by_video.values():
        for r in rows:
            per_cls_rows[r["label"]].append(r)
    chosen: list[dict] = []
    for c in classes:
        rows = sorted(per_cls_rows[c], key=lambda r: (r["video"], r["frame_id"]))
        if len(rows) > quota[c]:
            step = len(rows) / quota[c]
            rows = [rows[int(i * step)] for i in range(quota[c])]
        chosen.extend(rows)
    chosen_by_video: dict[str, list[dict]] = defaultdict(list)
    for r in chosen:
        chosen_by_video[r["video"]].append(r)

    # 3) 逐视频顺序抽帧裁剪
    manifest: list[dict] = []
    stats_fail = 0
    per_class: Counter = Counter()
    t0 = time.time()
    for vid, rows in sorted(chosen_by_video.items()):
        rows.sort(key=lambda r: r["frame_id"])
        vpath = _resolve_video(vid, video_roots)
        if vpath is None:
            print(f"[build] 跳过（视频缺失）: {vid}")
            stats_fail += len(rows)
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            stats_fail += len(rows)
            continue
        cur_fid, cur_frame = -1, None
        for r in rows:
            fid = int(r["frame_id"])
            if fid != cur_fid:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ok, cur_frame = cap.read()
                cur_fid = fid if ok else -1
            if cur_frame is None:
                stats_fail += 1
                continue
            x, y, w, h = [float(v) for v in r["bbox"]]
            ex = (args.crop_expand - 1.0) / 2.0
            H, W = cur_frame.shape[:2]
            x1, y1 = max(0, int(x - w * ex)), max(0, int(y - h * ex))
            x2, y2 = min(W, int(x + w * (1 + ex))), min(H, int(y + h * (1 + ex)))
            if x2 - x1 < 8 or y2 - y1 < 8:
                stats_fail += 1
                continue
            crop = letterbox_canvas(cur_frame[y1:y2, x1:x2])
            cls = r["label"]
            per_class[cls] += 1
            rel = f"crops/{cls}/{vid.replace('.mp4','')}_f{fid}_{r['track_id']}.jpg"
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)  # cv2.imwrite 不自动建目录
            cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            manifest.append({"image": rel, "cls": cls, "source": "labelpack", "video": vid,
                             "frame_id": fid})
        cap.release()
    print(f"[build] 标签包裁剪 {len(manifest)} 张（失败 {stats_fail}）{time.time()-t0:.0f}s")

    # 4) HMDB51 演示类（parquet → 解码 → 采样 → 中心裁剪/检测器）
    if args.hmdb:
        manifest.extend(_build_hmdb(args, out, classes, per_class))

    # 5) 分层 split + 落盘
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for m in manifest:
        by_cls[m["cls"]].append(m)
    train, val = [], []
    rng = random.Random(args.seed)
    for c, rows in sorted(by_cls.items()):
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * args.val_ratio)) if len(rows) >= 4 else 0
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    for split, rows in (("train", train), ("val", val)):
        with (out / f"manifest_{split}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in rows)
    dist = {c: {"train": sum(1 for m in train if m["cls"] == c),
                "val": sum(1 for m in val if m["cls"] == c)} for c in sorted(by_cls)}
    json.dump({"classes": classes, "distribution": dist, "total": len(manifest),
               "build_args": {k: str(v) for k, v in vars(args).items()}},
              open(out / "dataset_stats.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    (out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    print("[build] 分布:", json.dumps(dist, ensure_ascii=False))
    print(f"[build] -> {out}（train {len(train)} / val {len(val)}）")
    return 0


def _build_hmdb(args: argparse.Namespace, out: Path, classes: list[str],
                per_class: Counter) -> list[dict]:
    """HMDB51 parquet 分片 → 解码视频 → 采样帧 → crop。类名映射: 数据集名:项目类名。"""

    import pyarrow.parquet as pq

    mapping: dict[str, str] = {}
    for item in args.hmdb_classes:
        ds_name, proj_cls = item.split(":", 1)
        mapping[ds_name.lower()] = proj_cls
    per_clip_frames = args.hmdb_frames_per_clip
    manifest: list[dict] = []
    files = sorted(glob.glob(str(ROOT / args.hmdb)))
    print(f"[build][hmdb] 分片 {len(files)} 个，映射 {mapping}")
    for pf in files:
        try:
            table = pq.read_table(pf, columns=["video"])
        except Exception as exc:
            print(f"[build][hmdb] 跳过分片 {pf}: {exc}")
            continue
        for i in range(table.num_rows):
            item = table.column("video")[i].as_py()
            path = str(item.get("path", ""))
            ds_cls = _hmdb_action_name(path)
            proj_cls = mapping.get(ds_cls or "", None)
            if proj_cls not in classes:
                continue
            if per_class[proj_cls] >= args.max_per_class:
                continue
            tmp = None
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=Path(path).suffix or ".avi",
                                                  delete=False)
                tmp.write(item["bytes"])
                tmp.close()
                cap = cv2.VideoCapture(tmp.name)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if n <= 0:
                    continue
                ids = np.linspace(0, n - 1, per_clip_frames).astype(int)
                for fid in ids:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    H, W = frame.shape[:2]
                    if args.use_detector:
                        boxes = _detect_person(frame)
                        if boxes:
                            x, y, w, h = boxes[0]
                        else:
                            x, y, w, h = W * 0.2, H * 0.1, W * 0.6, H * 0.8
                    else:
                        x, y, w, h = W * 0.2, H * 0.1, W * 0.6, H * 0.8
                    crop = letterbox_canvas(frame[int(y):int(y + h), int(x):int(x + w)])
                    per_class[proj_cls] += 1
                    rel = f"crops/{proj_cls}/hmdb_{ds_cls}_{Path(path).stem}_f{int(fid)}.jpg"
                    dst = out / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    manifest.append({"image": rel, "cls": proj_cls,
                                     "source": f"hmdb:{ds_cls}", "video": path,
                                     "frame_id": int(fid)})
                cap.release()
            except Exception as exc:
                print(f"[build][hmdb] 解码失败 {path}: {exc}")
            finally:
                if tmp is not None:
                    Path(tmp.name).unlink(missing_ok=True)
    return manifest


_DETECTOR = None


def _detect_person(frame: np.ndarray) -> list[list[float]]:
    """可选：项目 Detector person 框（xywh），置信度降序。"""
    global _DETECTOR
    try:
        if _DETECTOR is None:
            from src.core.detector import Detector
            _DETECTOR = Detector({"active_onnx_path": "models/detector/yolov8n_int8.onnx",
                                  "confidence_threshold": 0.35})
        dets = _DETECTOR.infer(frame)
        persons = [d for d in dets if d.get("class_id") == 0]
        persons.sort(key=lambda d: -float(d.get("score", 0)))
        return [[float(v) for v in d["bbox"]] for d in persons]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# train：MobileNetV3-Small
# --------------------------------------------------------------------------- #
def _load_manifest_split(data_dir: Path, split: str) -> list[dict]:
    f = data_dir / f"manifest_{split}.jsonl"
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def train(args: argparse.Namespace) -> int:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    data_dir = Path(args.data)
    classes = json.loads((data_dir / "classes.json").read_text(encoding="utf-8"))
    cls2idx = {c: i for i, c in enumerate(classes)}
    rows_tr = _load_manifest_split(data_dir, "train")
    rows_va = _load_manifest_split(data_dir, "val")
    print(f"[train] classes={classes} train={len(rows_tr)} val={len(rows_va)}")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tf_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.15, 0.02),
        transforms.RandomAffine(degrees=6, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    tf_eval = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                  transforms.Normalize(mean, std)])

    class CropDS(Dataset):
        def __init__(self, rows, tf):
            self.rows, self.tf = rows, tf

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            img = cv2.imread(str(data_dir / r["image"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.tf(img), cls2idx[r["cls"]]

    counts = Counter(r["cls"] for r in rows_tr)
    weights = [1.0 / max(counts[c], 1) for c in classes]
    sampler_w = [weights[cls2idx[r["cls"]]] for r in rows_tr]
    g = torch.Generator().manual_seed(args.seed)
    loader_tr = DataLoader(CropDS(rows_tr, tf_train), batch_size=args.batch, shuffle=False,
                           sampler=torch.utils.data.WeightedRandomSampler(
                               sampler_w, num_samples=len(rows_tr), replacement=True,
                               generator=g), num_workers=args.workers, pin_memory=(device == "cuda"))
    loader_va = DataLoader(CropDS(rows_va, tf_eval), batch_size=args.batch,
                           shuffle=False, num_workers=2, pin_memory=(device == "cuda"))

    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        pretrained = "imagenet1k"
    except Exception as exc:
        print(f"[train] 预训练权重不可用（离线?）→ 随机初始化: {exc}")
        model = models.mobilenet_v3_small(weights=None)
        pretrained = "scratch"
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.to(device)

    freq = np.array([max(counts[c], 1) for c in classes], dtype=np.float32)
    w = torch.tensor(freq.sum() / (len(classes) * freq), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, tn, tc = 0.0, 0, 0
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tl += float(loss) * len(yb)
            tn += len(yb)
            tc += int((logits.argmax(1) == yb).sum())
        sched.step()
        model.eval()
        correct = Counter()
        total = Counter()
        conf_pair: Counter = Counter()
        with torch.no_grad():
            for xb, yb in loader_va:
                pred = model(xb.to(device)).argmax(1).cpu()
                for p, y in zip(pred.tolist(), yb.tolist()):
                    total[y] += 1
                    correct[y] += int(p == y)
                    if p != y:
                        conf_pair[f"{classes[y]}->{classes[p]}"] += 1
        val_acc = sum(correct.values()) / max(sum(total.values()), 1)
        history.append({"epoch": epoch, "train_loss": round(tl / max(tn, 1), 4),
                        "train_acc": round(tc / max(tn, 1), 4), "val_acc": round(val_acc, 4)})
        print(f"[train] epoch {epoch}/{args.epochs} loss={history[-1]['train_loss']}"
              f" val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": classes,
                        "val_acc": val_acc, "epoch": epoch}, data_dir / "best.pth")
    per_class = {classes[c]: {"n": total[c], "recall": round(correct[c] / max(total[c], 1), 3)}
                 for c in range(len(classes)) if total[c]}
    metrics = {"best_val_acc": round(best_acc, 4), "pretrained": pretrained,
               "per_class_recall": per_class,
               "top_confusions": dict(conf_pair.most_common(8)), "history": history}
    json.dump(metrics, open(data_dir / "train_metrics.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[train] best_val_acc={best_acc:.4f} -> {data_dir}/best.pth")
    return 0


# --------------------------------------------------------------------------- #
# export：ONNX + parity
# --------------------------------------------------------------------------- #
def export(args: argparse.Namespace) -> int:
    import torch
    from torchvision import models

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 3, SIZE, SIZE)
    torch.onnx.export(model, x, str(out_path), opset_version=17, do_constant_folding=True,
                      input_names=["input"], output_names=["logits"])
    # parity：ORT vs torch
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    rel = []
    with torch.no_grad():
        for _ in range(3):
            xi = torch.randn(1, 3, SIZE, SIZE)
            o_t = model(xi).numpy()
            o_o = sess.run(None, {"input": xi.numpy()})[0]
            rel.append(float(np.abs(o_t - o_o).max() / max(float(np.abs(o_t).max()), 1e-6)))
    (out_path.with_suffix(".json")).write_text(json.dumps(
        {"classes": classes, "val_acc": ckpt.get("val_acc"),
         "max_rel_err_vs_torch": round(max(rel), 6), "opset": 17,
         "size_mb": round(out_path.stat().st_size / 1e6, 1)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[export] {out_path} classes={classes} parity_max_rel={max(rel):.5f}"
          f"（盒上换 .om 后用 g4 探针复测时延）")
    return 0


# --------------------------------------------------------------------------- #
# selftest：合成数据端到端（build 内存版 → train 2 epoch → export → parity）
# --------------------------------------------------------------------------- #
def selftest(args: argparse.Namespace) -> int:
    tmp = Path(args.out or "results/crop_selftest")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "crops").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows_tr, rows_va = [], []
    for ci, cls in enumerate(DEFAULT_CLASSES):
        (tmp / "crops" / cls).mkdir(parents=True, exist_ok=True)
        for i in range(24):
            img = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
            img[:, :, ci % 3] = (img[:, :, ci % 3].astype(int) + 60).clip(0, 255).astype(np.uint8)
            rel = f"crops/{cls}/synth_{i}.jpg"
            cv2.imwrite(str(tmp / rel), img)
            (rows_va if i < 6 else rows_tr).append({"image": rel, "cls": cls})
    for split, rows in (("train", rows_tr), ("val", rows_va)):
        with (tmp / f"manifest_{split}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(m) + "\n" for m in rows)
    (tmp / "classes.json").write_text(json.dumps(DEFAULT_CLASSES), encoding="utf-8")
    print(f"[selftest] 合成数据 {len(rows_tr)}/{len(rows_va)}")

    train_args = argparse.Namespace(data=str(tmp), epochs=2, batch=32, lr=2e-3, device="cpu",
                                    workers=0, seed=0)
    train(train_args)
    export_args = argparse.Namespace(ckpt=str(tmp / "best.pth"), out=str(tmp / "crop_selftest.onnx"))
    export(export_args)
    ok = (tmp / "crop_selftest.onnx").exists() and (tmp / "train_metrics.json").exists()
    print(f"[selftest] {'PASS' if ok else 'FAIL'} -> {tmp}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="标签包/HMDB51 → crop 数据集")
    b.add_argument("--labels", nargs="+", default=[
        "results/vlm_labels_v2/*_vlm_labels.json",
        "results/vlm_smoking_relabel/*_relabel.json"])
    b.add_argument("--exclude", nargs="*", default=["vlm_labels_test", "_test"])
    b.add_argument("--hmdb", default="data_external/hmdb51/shards/*.parquet")
    b.add_argument("--hmdb-classes", nargs="*",
                   default=["wave:Wave", "throw:Throw", "shake_hands:Handshake"])
    b.add_argument("--hmdb-frames-per-clip", type=int, default=6)
    b.add_argument("--use-detector", action="store_true", help="HMDB 用项目检测器取 person 框")
    b.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    b.add_argument("--video-root", nargs="*", default=[])
    b.add_argument("--out", default="results/crop_dataset")
    b.add_argument("--max-per-class", type=int, default=4000)
    b.add_argument("--min-conf", type=float, default=0.55)
    b.add_argument("--crop-expand", type=float, default=1.25)
    b.add_argument("--val-ratio", type=float, default=0.15)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--append", action="store_true")
    b.set_defaults(fn=build)

    t = sub.add_parser("train", help="MobileNetV3-Small 训练")
    t.add_argument("--data", default="results/crop_dataset")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--device", default="auto")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=train)

    e = sub.add_parser("export", help="导出 ONNX + parity 校验")
    e.add_argument("--ckpt", default="results/crop_dataset/best.pth")
    e.add_argument("--out", default="models/crop/crop_v1.onnx")
    e.set_defaults(fn=export)

    s = sub.add_parser("selftest", help="合成数据端到端冒烟")
    s.add_argument("--out", default="results/crop_selftest")
    s.set_defaults(fn=selftest)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
