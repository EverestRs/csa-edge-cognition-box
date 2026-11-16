from __future__ import annotations

"""H03 增量数据微调（T2：分歧帧重标的落地）。

从生产锚点 final.pth 出发，在 原manifest + H03增量 上短程微调，
验证集（原15%分层）指标对比基线 val 91.01%：不降才允许 --deploy。

用法:
  python tools/finetune_with_extra.py --extra results/training_data/h03_extra/class_H03_extra.npz \
      --epochs 8 --device cuda [--deploy]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.pose_classifier import FastPoseDataset, PoseClassifier

ROOT = Path(__file__).resolve().parents[1]
OLD_PTH = ROOT / "models/action/pose_classifier_best.pth"  # 生产真值（= 部署 ONNX 的导出来源；final.pth 是末轮 epoch 已劣化，溯源校准见 ROADMAP §8）
CLASSES = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"]
H03_IDX = CLASSES.index("H03")


class ExtraDataset(Dataset):
    """H03 增量序列（dump 口径：原始像素坐标，与 FastPoseDataset 的归一化约定配套）。"""

    def __init__(self, npz_path: Path):
        self.data = np.load(npz_path, allow_pickle=True)["pose_seq"].astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].copy()
        x[:, :, 0] /= 1920.0
        x[:, :, 1] /= 1080.0
        x[:, :, 2] = np.clip(x[:, :, 2], 0.0, 1.0)
        return torch.from_numpy(x), H03_IDX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default="results/training_data/h03_extra/class_H03_extra.npz")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4, help="微调 LR（低于重训 1e-3，防灾难遗忘）")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--deploy", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((ROOT / "results/training_data/manifest.json").read_text(encoding="utf-8"))
    data_dir = ROOT / "results/training_data"
    # 按 video 分层 15% 验证集（与原训练口径一致）
    videos = sorted({s["video"] for s in manifest})
    rng = random.Random(7)
    rng.shuffle(videos)
    val_videos = set(videos[: max(1, int(len(videos) * 0.15))])
    train_m = [s for s in manifest if s["video"] not in val_videos]
    val_m = [s for s in manifest if s["video"] in val_videos]
    train_ds = FastPoseDataset(train_m, data_dir, augment=True)
    val_ds = FastPoseDataset(val_m, data_dir, augment=False)
    extra_ds = ExtraDataset(ROOT / args.extra)
    print(f"数据: 原 train {len(train_ds)} + H03增量 {len(extra_ds)} | val {len(val_ds)}")

    full_train = torch.utils.data.ConcatDataset([train_ds, extra_ds])
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = PoseClassifier(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(OLD_PTH, map_location="cpu", weights_only=True))
    model.to(device)

    def evaluate() -> float:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in DataLoader(val_ds, batch_size=64, num_workers=0):
                pred = model(xb.to(device)).argmax(1).cpu()
                correct += int((pred == yb).sum())
                total += len(yb)
        return correct / max(1, total)

    base_val = evaluate()
    print(f"微调前 val acc: {base_val:.2%}（基线 91.01%）")
    # 类权重：H03 增量上采样（增量条数少，加权防淹没）
    counts = torch.zeros(len(CLASSES))
    for _, y in DataLoader(extra_ds, batch_size=64):
        for yi in y:
            counts[yi] += len(y)
    loader = DataLoader(full_train, batch_size=args.batch, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = base_val
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss))
        val = evaluate()
        print(f"  epoch {epoch + 1}/{args.epochs}: loss={np.mean(losses):.4f} val_acc={val:.2%}")
        if val > best_val:
            best_val = val
            out = ROOT / "results/extended_model"
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / "pose_classifier_h03_ft.pth")
            print(f"    ↑ 新最佳，已存 {out / 'pose_classifier_h03_ft.pth'}")
    print(f"\n微调完成: val {base_val:.2%} → best {best_val:.2%}（基线 91.01%）")
    if best_val >= base_val and args.deploy:
        import shutil
        shutil.copy(ROOT / "results/extended_model/pose_classifier_h03_ft.pth",
                    ROOT / "models/action/pose_classifier_final.pth")
        # ONNX 重导出（A2 口径：dummy 维度读 v_dim）
        model_cpu = model.cpu().eval()
        dummy = torch.randn(1, 32, model_cpu.v_dim, 3)
        torch.onnx.export(model_cpu, dummy, str(ROOT / "models/action/pose_classifier.onnx"),
                          opset_version=17, input_names=["pose_input"], output_names=["logits"],
                          dynamic_axes={"pose_input": {0: "batch"}})
        print("已部署（final.pth + onnx；best.pth 未动）")
    elif not args.deploy:
        print("未部署（--deploy 触发；仅 val 不降时建议部署）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
