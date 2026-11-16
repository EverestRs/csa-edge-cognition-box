"""优化版 PoseClassifier 训练脚本 - 使用索引加速数据加载"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CLASSES = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


class FastPoseDataset(Dataset):
    """使用预构建索引的快速姿态序列数据集。"""

    def __init__(self, manifest: list[dict], data_dir: Path, augment: bool = False):
        self.samples = manifest
        self.data_dir = data_dir
        self.augment = augment

        # 预加载所有 NPZ 数据并构建索引
        self._pose_cache: dict[str, np.ndarray] = {}  # video -> pose_seq
        self._index: list[tuple[str, int]] = []  # (video, row_idx) for each sample

        # 按视频分组构建索引
        video_samples = defaultdict(list)
        for i, s in enumerate(self.samples):
            video_samples[s["video"]].append(i)

        print(f"[INFO] 预加载 NPZ 数据并构建索引...")
        for video, indices in video_samples.items():
            stem = Path(video).stem
            npz_path = data_dir / f"{stem}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"NPZ not found: {npz_path}")

            data = np.load(str(npz_path), allow_pickle=True)
            frame_ids = data["frame_id"]
            track_ids = data["track_id"]
            pose_seq = data["pose_seq"]
            self._pose_cache[video] = pose_seq

            # 为每个样本构建 (frame_id, track_id) -> row_idx 映射
            lookup = {}
            for row_idx in range(len(frame_ids)):
                key = (int(frame_ids[row_idx]), str(track_ids[row_idx]))
                lookup[key] = row_idx

            # 为这批样本建立索引
            for idx in indices:
                s = self.samples[idx]
                key = (s["frame_id"], s["track_id"])
                if key in lookup:
                    self._index.append((video, lookup[key]))
                else:
                    # 找最近的 frame_id
                    all_fids = frame_ids
                    nearest = int(np.argmin(np.abs(all_fids - s["frame_id"])))
                    self._index.append((video, int(nearest)))

        print(f"[INFO] 索引构建完成，共 {len(self._index)} 个样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        s = self.samples[idx]
        video, row_idx = self._index[idx]
        class_idx = int(s["class_idx"])

        pose_seq = self._pose_cache[video][row_idx].astype(np.float32)

        if self.augment:
            pose_seq = self._augment(pose_seq)

        return torch.from_numpy(pose_seq), class_idx

    @staticmethod
    def _augment(pose: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*pose.shape).astype(np.float32) * 0.005
        pose = pose + noise
        if np.random.rand() > 0.5:
            pose[..., 0] += np.random.randn() * 2.0
        return np.clip(pose, -1.0, 1.0)


class PoseClassifier(nn.Module):
    def __init__(self, num_classes: int = 11, t_dim: int = 32, v_dim: int = 133):
        super().__init__()
        self.t_dim = t_dim
        self.v_dim = v_dim
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=128 * v_dim, hidden_size=256, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, t, v, c = x.shape
        x_flat = x.reshape(-1, c)
        x_encoded = self.spatial_encoder(x_flat)
        x_encoded = x_encoded.reshape(batch_size, t, v * 128)
        x_gru, _ = self.gru(x_encoded)
        x_last = x_gru[:, -1, :]
        logits = self.classifier(x_last)
        return logits


def collate_fn(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), torch.tensor(labels, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="快速训练 PoseClassifier")
    parser.add_argument("--data-dir", default=str(BASE_DIR / "results" / "training_data"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "models" / "action_v2"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 设备: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest_path = data_dir / "manifest.json"
    with open(manifest_path) as f:
        all_samples = json.load(f)

    print(f"[INFO] 总样本数: {len(all_samples)}")

    # 按视频分层划分
    videos = list(set(s["video"] for s in all_samples))
    np.random.shuffle(videos)
    n_val = max(1, int(len(videos) * args.val_ratio))
    val_videos = set(videos[:n_val])
    train_videos = set(videos[n_val:])

    train_samples = [s for s in all_samples if s["video"] in train_videos]
    val_samples = [s for s in all_samples if s["video"] in val_videos]

    print(f"[INFO] 训练视频: {len(train_videos)}, 验证视频: {len(val_videos)}")
    print(f"[INFO] 训练样本: {len(train_samples)}, 验证样本: {len(val_samples)}")

    # 计算类别权重
    train_labels = [s["refined_label"] for s in train_samples]
    label_counts = {l: train_labels.count(l) for l in set(train_labels)}
    max_count = max(label_counts.values()) if label_counts else 1
    class_weights = np.ones(11, dtype=np.float32)
    for label, count in label_counts.items():
        idx = CLASS_TO_IDX.get(label, 0)
        class_weights[idx] = max_count / max(1, count)
    class_weights = np.clip(class_weights, 0.5, 10.0)
    class_weights = class_weights / class_weights.mean()
    print(f"[INFO] 类别权重: {dict(zip(CLASSES, class_weights.round(2)))}")

    # 创建数据集
    train_dataset = FastPoseDataset(train_samples, data_dir, augment=True)
    val_dataset = FastPoseDataset(val_samples, data_dir, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )

    # 自动检测关键点数量
    sample_data = np.load(str(data_dir / f"{Path(train_samples[0]['video']).stem}.npz"), allow_pickle=True)
    v_dim = int(sample_data["pose_seq"].shape[2])
    print(f"[INFO] 关键点数量: {v_dim}")

    model = PoseClassifier(num_classes=11, v_dim=v_dim).to(device)

    weight_tensor = torch.from_numpy(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n[INFO] 开始训练 ({args.epochs} epochs)...")
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/max(1,total):.2f}%")

        train_acc = 100. * correct / max(1, total)

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100. * val_correct / max(1, val_total)
        print(f"\nEpoch {epoch}/{args.epochs} | Train Loss: {total_loss/len(train_loader):.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(output_dir / "pose_classifier_best.pth"))
            print(f"  [SAVE] 最佳模型 (val_acc={val_acc:.2f}%)")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), str(output_dir / f"pose_classifier_epoch{epoch}.pth"))

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")

    # 导出 ONNX
    print(f"\n[INFO] 导出 ONNX 模型...")
    model.load_state_dict(torch.load(str(output_dir / "pose_classifier_best.pth"), map_location="cpu"))
    model.eval()
    model.cpu()
    dummy = torch.randn(1, 32, 17, 3)
    torch.onnx.export(model, dummy, str(output_dir / "pose_classifier.onnx"),
                      input_names=["pose_input"], output_names=["logits"],
                      opset_version=17, dynamic_axes={"pose_input": {0: "batch"}, "logits": {0: "batch"}})
    print(f"[OK] ONNX: {output_dir / 'pose_classifier.onnx'}")


if __name__ == "__main__":
    main()