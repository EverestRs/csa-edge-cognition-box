from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 项目 11 类
CLASSES = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


class PoseSequenceDataset(Dataset):
    """从 NPZ 文件中加载姿态序列 + 标签的数据集。"""

    def __init__(self, manifest: list[dict], data_dir: Path, augment: bool = False):
        self.samples = manifest
        self.data_dir = data_dir
        self.augment = augment

        # 按视频分组，方便按需加载 NPZ
        self.video_samples: dict[str, list[dict]] = {}
        for s in self.samples:
            self.video_samples.setdefault(s["video"], []).append(s)

        # 缓存已加载的 NPZ 数据
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def _load_video_data(self, video: str) -> dict[str, np.ndarray]:
        if video in self._cache:
            return self._cache[video]
        stem = Path(video).stem
        npz_path = self.data_dir / f"{stem}.npz"
        if npz_path.exists():
            data = np.load(str(npz_path), allow_pickle=True)
            self._cache[video] = data
            return data
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        s = self.samples[idx]
        video = s["video"]
        frame_id = s["frame_id"]
        track_id = s["track_id"]
        class_idx = int(s["class_idx"])

        data = self._load_video_data(video)
        # 在 NPZ 中找到对应帧的 pose 序列
        frame_ids = data["frame_id"]
        track_ids = data["track_id"]
        pose_seqs = data["pose_seq"]  # (N, T, V, 3)

        # 查找匹配的行
        mask = np.ones(len(frame_ids), dtype=bool)
        for i in range(len(frame_ids)):
            if int(frame_ids[i]) != frame_id:
                mask[i] = False
            elif str(track_ids[i]) != track_id:
                mask[i] = False

        match_indices = np.where(mask)[0]
        if len(match_indices) == 0:
            # 回退：找最近的 frame_id
            match_idx = int(np.argmin(np.abs(frame_ids - frame_id)))
        else:
            match_idx = int(match_indices[0])

        pose_seq = pose_seqs[match_idx].astype(np.float32)  # (T, V, 3)

        # 数据增强
        if self.augment:
            pose_seq = self._augment(pose_seq)

        tensor = torch.from_numpy(pose_seq)  # (T, V, 3)
        return tensor, class_idx

    @staticmethod
    def _augment(pose: np.ndarray) -> np.ndarray:
        """简单的姿态数据增强。"""
        # 轻微随机噪声
        noise = np.random.randn(*pose.shape).astype(np.float32) * 0.005
        pose = pose + noise
        # 随机水平翻转关键点
        if np.random.rand() > 0.5:
            # 翻转 x 坐标（相对于图像中心，归一化后以 0.5 为轴）
            # 由于坐标是绝对像素值，这里不做真正翻转，只加小抖动
            pose[..., 0] += np.random.randn() * 2.0
        return np.clip(pose, -1.0, 1.0)


class PoseClassifier(nn.Module):
    """基于姿态序列的轻量级行为分类器。

    输入: (batch, T, V, 3)  — T=32帧, V=133关键点(ViTPose WholeBody), 3=(x,y,conf)
    输出: (batch, 11)       — 11 类行为分数
    """

    def __init__(self, num_classes: int = 11, t_dim: int = 32, v_dim: int = 133):
        super().__init__()
        self.t_dim = t_dim
        self.v_dim = v_dim

        # 空间编码：每个关键点独立编码
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # 时序编码：双向 GRU
        self.gru = nn.GRU(
            input_size=128 * v_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, V, 3)
        batch_size, t, v, c = x.shape

        # 空间编码: (batch*T*V, 3) -> (batch*T*V, 128)
        x_flat = x.reshape(-1, c)
        x_encoded = self.spatial_encoder(x_flat)  # (batch*T*V, 128)

        # 重塑为 (batch, T, V*128)
        x_encoded = x_encoded.reshape(batch_size, t, v * 128)

        # GRU: (batch, T, V*128) -> (batch, T, 512)
        x_gru, _ = self.gru(x_encoded)  # bidirectional: hidden*2=512

        # 取最后一帧的输出
        x_last = x_gru[:, -1, :]  # (batch, 512)

        # 分类头
        logits = self.classifier(x_last)  # (batch, 11)
        return logits


def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """自定义 collate，处理可能的不同长度。"""
    tensors = []
    labels = []
    for t, l in batch:
        tensors.append(t)
        labels.append(l)
    return torch.stack(tensors, dim=0), torch.tensor(labels, dtype=torch.long)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)  # (batch, T, V, 3)
        targets = targets.to(device)

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
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f} | Acc: {100. * correct / max(1, total):.2f}%")

    avg_loss = total_loss / max(1, len(loader))
    accuracy = 100. * correct / max(1, total)
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    accuracy = 100. * correct / max(1, total)
    confusion = np.zeros((11, 11), dtype=np.int32)
    for p, t in zip(all_preds, all_targets):
        confusion[t, p] += 1
    return avg_loss, accuracy, confusion


def main() -> None:
    parser = argparse.ArgumentParser(description="微调姿态序列行为分类器")
    parser.add_argument("--data-dir", default=str(BASE_DIR / "results" / "training_data"), help="训练数据目录")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "models" / "action"), help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", type=str, default="", help="从 checkpoint 恢复训练")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] CUDA 可用: {torch.cuda.is_available()}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 加载 manifest
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] 未找到 manifest: {manifest_path}")
        print(f"        请先运行 extract_action_training_data.py")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        all_samples = json.load(f)

    print(f"[INFO] 总样本数: {len(all_samples)}")

    # 统计标签分布
    label_counts: dict[str, int] = {}
    for s in all_samples:
        label = s["refined_label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"[INFO] 标签分布:")
    for label in sorted(label_counts.keys()):
        print(f"  {label}: {label_counts[label]}")

    # 按视频分层划分训练/验证集
    videos = list(set(s["video"] for s in all_samples))
    np.random.shuffle(videos)
    n_val = max(1, int(len(videos) * args.val_ratio))
    val_videos = set(videos[:n_val])
    train_videos = set(videos[n_val:])

    train_samples = [s for s in all_samples if s["video"] in train_videos]
    val_samples = [s for s in all_samples if s["video"] in val_videos]

    print(f"\n[INFO] 训练视频: {len(train_videos)}, 验证视频: {len(val_videos)}")
    print(f"[INFO] 训练样本: {len(train_samples)}, 验证样本: {len(val_samples)}")

    # 计算类别权重（处理不平衡）
    train_labels = [s["refined_label"] for s in train_samples]
    label_counts_train = {l: train_labels.count(l) for l in set(train_labels)}
    max_count = max(label_counts_train.values()) if label_counts_train else 1
    class_weights = np.ones(11, dtype=np.float32)
    for label, count in label_counts_train.items():
        idx = CLASS_TO_IDX.get(label, 0)
        class_weights[idx] = max_count / max(1, count)
    # 限制权重范围
    class_weights = np.clip(class_weights, 0.5, 10.0)
    class_weights = class_weights / class_weights.mean()
    print(f"[INFO] 类别权重: {dict(zip(CLASSES, class_weights.round(2)))}")

    # 创建数据集
    train_dataset = PoseSequenceDataset(train_samples, data_dir, augment=True)
    val_dataset = PoseSequenceDataset(val_samples, data_dir, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 创建模型
    # 自动检测关键点数量
    sample = all_samples[0]
    sample_video = sample["video"]
    sample_npz = data_dir / f"{Path(sample_video).stem}.npz"
    sample_data = np.load(str(sample_npz), allow_pickle=True)
    v_dim = int(sample_data["pose_seq"].shape[2])  # 从数据中获取关键点数量
    print(f"[INFO] 关键点数量: {v_dim}")

    model = PoseClassifier(num_classes=11, v_dim=v_dim)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"[INFO] 从 checkpoint 恢复: {args.resume}")

    model = model.to(device)

    # 损失函数（带类别权重）
    weight_tensor = torch.from_numpy(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    print(f"\n[INFO] 开始训练 ({args.epochs} epochs)...")
    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc, confusion = evaluate(model, val_loader, criterion, device)

        # 打印混淆矩阵摘要
        per_class_acc = []
        for i in range(11):
            total_cls = confusion[i].sum()
            if total_cls > 0:
                per_class_acc.append((CLASSES[i], confusion[i, i] / total_cls * 100))
        per_class_str = ", ".join(f"{c}: {a:.1f}%" for c, a in per_class_acc if a > 0)

        print(f"\nEpoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        print(f"  各类准确率: {per_class_str}")

        scheduler.step()

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), str(output_dir / "pose_classifier_best.pth"))
            print(f"  [SAVE] 最佳模型 (val_acc={val_acc:.2f}%)")

        # 每 10 轮保存 checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), str(output_dir / f"pose_classifier_epoch{epoch}.pth"))

    print(f"\n{'='*60}")
    print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}% (epoch {best_epoch})")
    best_model_path = output_dir / "pose_classifier_best.pth"

    # 导出 ONNX
    print(f"\n[INFO] 导出 ONNX 模型...")
    try:
        model.load_state_dict(torch.load(str(best_model_path), map_location="cpu"))
        model.eval()
        model.cpu()

        dummy_input = torch.randn(1, 32, 17, 3, dtype=torch.float32)
        onnx_path = output_dir / "pose_classifier.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["pose_input"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={
                "pose_input": {0: "batch"},
                "logits": {0: "batch"},
            },
        )
        print(f"[OK] ONNX 模型已导出: {onnx_path}")

        # 验证
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider", "CUDAExecutionProvider"])
        print(f"[INFO] ONNX 输入: {[(i.name, i.shape) for i in sess.get_inputs()]}")
        print(f"[INFO] ONNX 输出: {[(o.name, o.shape) for o in sess.get_outputs()]}")
    except Exception as e:
        print(f"[WARN] ONNX 导出失败: {e}")

    # 保存标签映射
    labels_path = output_dir / "pose_classifier_labels.json"
    labels_path.write_text(
        json.dumps({"classes": CLASSES, "class_to_idx": CLASS_TO_IDX}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 标签映射: {labels_path}")

    # 保存最终模型
    final_path = output_dir / "pose_classifier_final.pth"
    torch.save(model.state_dict(), str(final_path))
    print(f"[OK] 最终模型: {final_path}")

    print(f"\n下一步:")
    print(f"  1. 运行 export_action_onnx.py 导出 ONNX")
    print(f"  2. 修改 model_config.yaml 中的 action.onnx_path 指向新模型")
    print(f"  3. 重新启动后端服务")


if __name__ == "__main__":
    main()