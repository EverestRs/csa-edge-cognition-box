from __future__ import annotations

"""当前口径训练（根因 3 修复的落地）：regen_training_data 的新数据 + H03 增量重训。

与旧训练的差异（可复现性修正）：
  - 划分用 sorted(videos) + random.Random(seed).shuffle（旧代码 list(set(...)) 受
    字符串哈希随机化影响，划分不可复现——"91.01% vs 29.21%"的调查发现之一）
  - 逐视频 val 报告（暴露视频间方差）
用法:
  python tools/train_v2.py [--epochs 100] [--device cuda] [--deploy]
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

from src.training.pose_classifier import PoseClassifier

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}


class V2Dataset(Dataset):
    """当前口径序列（regen 产出，坐标已 /W /H 归一化——与推理 append_pose 同口径）。"""

    def __init__(self, samples: list[dict], npz_dir: Path, augment: bool = False):
        self.augment = augment
        self._seqs: list[np.ndarray] = []
        self._ys: list[int] = []
        cache: dict[str, tuple] = {}
        for s in samples:
            video = s["video"]
            if video not in cache:
                npz = npz_dir / f"{Path(video).stem}.npz"
                d = np.load(npz, allow_pickle=True)
                cache[video] = (d["frame_id"].astype(int), d["pose_seq"].astype(np.float32))
            fids, seqs = cache[video]
            key = np.where(fids == int(s["frame_id"]))[0]
            if not len(key):
                continue
            x = seqs[int(key[0])].copy()
            # NPZ 存原始像素（dump 口径），训练归一化与 FastPoseDataset/推理 append_pose 对齐
            x[..., 0] /= 1920.0
            x[..., 1] /= 1080.0
            x[..., 2] = np.clip(x[..., 2], 0.0, 1.0)
            self._seqs.append(x)
            self._ys.append(CLS_IDX[s["refined_label"]])

    def __len__(self):
        return len(self._seqs)

    def __getitem__(self, idx):
        x = self._seqs[idx].copy()
        if self.augment:
            x[..., :2] += np.random.randn(*x[..., :2].shape).astype(np.float32) * 0.003
        return torch.from_numpy(x), self._ys[idx]


def evaluate(model, ds, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in DataLoader(ds, batch_size=64, num_workers=0):
            pred = model(xb.to(device)).argmax(1).cpu()
            correct += int((pred == yb).sum())
            total += len(yb)
    return correct / max(1, total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="results/training_data_v2")
    ap.add_argument("--extra", action="append", default=[],
                    help="附加 NPZ（如 H03 增量），整批标 H03")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--deploy", action="store_true")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    npz_dir = data_dir / "npz"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    videos = sorted({m["video"] for m in manifest})
    random.Random(args.seed).shuffle(videos)
    n_val = max(1, int(len(videos) * 0.15))
    val_videos = set(videos[:n_val])
    print(f"验证视频（seed={args.seed}，可复现）: {sorted(val_videos)}")

    train_m = [m for m in manifest if m["video"] not in val_videos]
    val_m = [m for m in manifest if m["video"] in val_videos]
    train_ds = V2Dataset(train_m, npz_dir, augment=True)
    val_ds = V2Dataset(val_m, npz_dir, augment=False)

    # 附加数据（--extra，整批同一标签，如 H03 增量）：真正接入训练
    # （原实现 extra_sets/ExtraDS 从未喂入 loader——死代码，--extra 是摆设）
    extra_ds_list = []
    for extra in args.extra:
        p = ROOT / extra
        arr = np.load(p, allow_pickle=True)["pose_seq"].astype(np.float32)
        rows = []
        for i in range(len(arr)):
            x = arr[i].copy()
            x[..., 0] /= 1920.0
            x[..., 1] /= 1080.0
            x[..., 2] = np.clip(x[..., 2], 0.0, 1.0)
            rows.append((x, CLS_IDX["H03"]))
        extra_ds_list.append(rows)
        print(f"附加数据: {extra} ({len(arr)} 条 H03)")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = PoseClassifier(num_classes=len(CLASSES)).to(device)

    class ExtraDataset(Dataset):
        """预归一化增量样本（H03 增量为 dump 口径原始像素，此处统一归一化）。"""

        def __init__(self, rows: list):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            x, y = self.rows[i]
            return torch.from_numpy(x), y

    datasets = [train_ds] + [ExtraDataset(rows) for rows in extra_ds_list]
    loader = DataLoader(torch.utils.data.ConcatDataset(datasets),
                        batch_size=args.batch, shuffle=True, num_workers=0)
    n_extra = sum(len(rows) for rows in extra_ds_list)
    print(f"训练集: v2 {len(train_ds)} + 附加 {n_extra}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val, best_state = 0.0, None
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss))
        sched.step()
        val = evaluate(model, val_ds, device)
        tag = ""
        if val > best_val:
            best_val = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tag = " ↑"
        print(f"  epoch {epoch + 1}/{args.epochs}: loss={np.mean(losses):.4f} val_acc={val:.2%}{tag}", flush=True)
    print(f"\n训练完成: best val_acc={best_val:.2%}（新数据口径，无历史基线可比）")
    if best_state is not None:
        model.load_state_dict(best_state)
        out = ROOT / "results/extended_model"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "pose_classifier_v2.pth")
        model_cpu = model.cpu().eval()
        dummy = torch.randn(1, 32, model_cpu.v_dim, 3)
        torch.onnx.export(model_cpu, dummy, str(out / "pose_classifier_v2.onnx"),
                          opset_version=17, input_names=["pose_input"], output_names=["logits"],
                          dynamic_axes={"pose_input": {0: "batch"}})
        print(f"产物: {out / 'pose_classifier_v2.{pth,onnx}'}")
        if args.deploy:
            import shutil
            shutil.copy(out / "pose_classifier_v2.onnx", ROOT / "models/action/pose_classifier.onnx")
            torch.save(model.state_dict(), ROOT / "models/action/pose_classifier_best.pth")
            print("已部署（onnx + best.pth；生产切换）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
