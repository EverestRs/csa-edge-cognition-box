#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""crop 仲裁阈值调优（T2.5 配套）：在 val 上扫每类阈值，输出可直接贴进
vlm_config.crop_review.class_thresholds 的 JSON。

策略：对每个非 Normal 类，取满足 precision ≥ 目标（默认 0.80）的最低阈值——
阈值越低召回越高；若全段都达不到目标精度，输出该类可达到的最高精度并标注
（提示需要补数据而非调阈值，parity 剩余分歧的正规修法）。

用法:
  python tools/tune_crop_thresholds.py --ckpt results/crop_dataset/best.pth \
      --data results/crop_dataset --target-precision 0.80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="results/crop_dataset/best.pth")
    ap.add_argument("--data", default="results/crop_dataset")
    ap.add_argument("--target-precision", type=float, default=0.80)
    ap.add_argument("--normal-rebate-sweep", action="store_true",
                    help="同时扫 normal_rebate_conf（Normal 置信分位数建议）")
    ap.add_argument("--out", default="results/crop_dataset/thresholds_tuned.json")
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import models, transforms

    data_dir = Path(a.data)
    classes = json.loads((data_dir / "classes.json").read_text(encoding="utf-8"))
    cls2idx = {c: i for i, c in enumerate(classes)}
    rows = [json.loads(l) for l in (data_dir / "manifest_val.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[tune] val={len(rows)} classes={classes}")

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                             transforms.Normalize(mean, std)])

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            img = cv2.imread(str(data_dir / rows[i]["image"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return tf(img), cls2idx[rows[i]["cls"]]

    loader = DataLoader(DS(), batch_size=64, num_workers=4)
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.softmax(model(xb), dim=1))
            ys.append(yb)
    P = torch.cat(probs).numpy()
    Y = torch.cat(ys).numpy()

    grid = [round(0.40 + 0.05 * i, 2) for i in range(11)]  # 0.40..0.90
    thresholds: dict[str, float] = {}
    report: dict[str, dict] = {}
    for c in classes:
        if c == "Normal":
            continue
        ci = cls2idx[c]
        p_c, y_c = P[:, ci], (Y == ci)
        best = None
        for t in grid:
            pred_pos = p_c >= t
            tp = int((pred_pos & y_c).sum())
            fp = int((pred_pos & ~y_c).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(int(y_c.sum()), 1)
            if prec >= a.target_precision and (best is None or t < best["threshold"]):
                best = {"threshold": t, "precision": round(prec, 3),
                        "recall": round(rec, 3), "val_pos": int(y_c.sum())}
            if t == grid[-1] and best is None:
                best = {"threshold": None, "precision": round(prec, 3), "recall": round(rec, 3),
                        "val_pos": int(y_c.sum()),
                        "note": f"全段精度未达 {a.target_precision}，需补数据（非阈值可解）"}
        if best and best.get("threshold"):
            thresholds[c] = best["threshold"]
        report[c] = best or {}
        print(f"  {c}: {report[c]}")

    normal_conf = P[:, cls2idx["Normal"]]
    rebate = round(float(np.quantile(normal_conf, 0.90)), 2) if a.normal_rebate_sweep else None

    out = {"target_precision": a.target_precision,
           "class_thresholds": thresholds,
           "normal_rebate_conf_suggestion": rebate,
           "per_class": report,
           "note": "贴入 vlm_config.crop_review.class_thresholds；normal_rebate_conf 取 "
                   "Normal 置信 P90（翻案需强证据）"}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[tune] -> {a.out}")
    print(json.dumps({"class_thresholds": thresholds, "normal_rebate_conf_suggestion": rebate},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
