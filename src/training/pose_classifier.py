from __future__ import annotations

"""PoseClassifier 训练（VLM-as-Teacher 弱监督）。

从 tools/train_pose_classifier_v2.py 收敛而来的规范实现，供三个入口共用：
  - python train.py（命令行）
  - POST /api/v1/train + GET /api/v1/train/{train_id}（REST 后台任务）
  - tools/train_pose_classifier_v2.py（历史脚本，保留兼容）

输入: results/training_data/manifest.json + *.npz（ViTPose 关键点序列）
输出: models/action_v2/pose_classifier_best.pth + pose_classifier.onnx + train_metrics.json
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

CLASSES = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


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
        logits = self.classifier(x_gru[:, -1, :])
        return logits


class FastPoseDataset(Dataset):
    """使用预构建索引的快速姿态序列数据集。"""

    def __init__(self, manifest: list[dict], data_dir: Path, augment: bool = False):
        self.samples = manifest
        self.data_dir = data_dir
        self.augment = augment

        self._pose_cache: dict[str, np.ndarray] = {}
        self._index: list[tuple[str, int]] = []

        video_samples: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(self.samples):
            video_samples[s["video"]].append(i)

        for video, indices in video_samples.items():
            stem = Path(video).stem
            npz_path = data_dir / f"{stem}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"NPZ not found: {npz_path}")

            data = np.load(str(npz_path), allow_pickle=True)
            frame_ids = data["frame_id"]
            track_ids = data["track_id"]
            self._pose_cache[video] = data["pose_seq"]

            lookup = {}
            for row_idx in range(len(frame_ids)):
                lookup[(int(frame_ids[row_idx]), str(track_ids[row_idx]))] = row_idx

            for idx in indices:
                s = self.samples[idx]
                key = (s["frame_id"], s["track_id"])
                if key in lookup:
                    self._index.append((video, lookup[key]))
                else:
                    nearest = int(np.argmin(np.abs(frame_ids - s["frame_id"])))
                    self._index.append((video, int(nearest)))

    def __len__(self) -> int:
        return len(self.samples)

    NORM_W = 1920.0
    NORM_H = 1080.0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        s = self.samples[idx]
        video, row_idx = self._index[idx]
        class_idx = int(s["class_idx"])
        pose_seq = self._pose_cache[video][row_idx].astype(np.float32)
        pose_seq[:, :, 0] /= self.NORM_W
        pose_seq[:, :, 1] /= self.NORM_H
        pose_seq[:, :, 2] = np.clip(pose_seq[:, :, 2], 0.0, 1.0)
        if self.augment:
            pose_seq = self._augment(pose_seq)
        return torch.from_numpy(pose_seq), class_idx

    @staticmethod
    def _augment(pose: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*pose.shape).astype(np.float32) * 0.003
        pose = pose + noise
        if np.random.rand() > 0.5:
            pose[..., 0] += np.random.randn() * 0.01
        return np.clip(pose, 0.0, 1.0)


def collate_fn(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), torch.tensor(labels, dtype=torch.long)


def train_pose_classifier(
    data_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.15,
    seed: int = 42,
    device: str = "auto",
    progress_cb: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """训练 PoseClassifier 并导出 ONNX。返回训练指标摘要。"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    manifest_path = data_dir / "manifest.json"
    with open(manifest_path) as f:
        all_samples = json.load(f)

    videos = list(set(s["video"] for s in all_samples))
    np.random.shuffle(videos)
    n_val = max(1, int(len(videos) * val_ratio))
    val_videos = set(videos[:n_val])
    train_videos = set(videos[n_val:])

    train_samples = [s for s in all_samples if s["video"] in train_videos]
    val_samples = [s for s in all_samples if s["video"] in val_videos]

    train_labels = [s["refined_label"] for s in train_samples]
    label_counts = {label: train_labels.count(label) for label in set(train_labels)}
    max_count = max(label_counts.values()) if label_counts else 1
    class_weights = np.ones(len(CLASSES), dtype=np.float32)
    for label, count in label_counts.items():
        class_weights[CLASS_TO_IDX.get(label, 0)] = max_count / max(1, count)
    class_weights = np.clip(class_weights, 0.5, 10.0)
    class_weights = class_weights / class_weights.mean()

    train_dataset = FastPoseDataset(train_samples, data_dir, augment=True)
    val_dataset = FastPoseDataset(val_samples, data_dir, augment=False)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )

    sample_data = np.load(str(data_dir / f"{Path(train_samples[0]['video']).stem}.npz"), allow_pickle=True)
    v_dim = int(sample_data["pose_seq"].shape[2])

    model = PoseClassifier(num_classes=len(CLASSES), v_dim=v_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: list[dict] = []
    best_val_acc = 0.0

    def report(payload: dict) -> None:
        if progress_cb is not None:
            try:
                progress_cb(payload)
            except Exception:
                pass

    report({
        "stage": "data_loaded",
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_videos": len(train_videos),
        "val_videos": len(val_videos),
        "v_dim": v_dim,
        "device": str(device),
    })

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
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
        train_acc = 100.0 * correct / max(1, total)
        train_loss = total_loss / max(1, len(train_loader))

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
        val_acc = 100.0 * val_correct / max(1, val_total)
        val_loss_avg = val_loss / max(1, len(val_loader))

        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss_avg, 4),
            "val_acc": round(val_acc, 2),
        })
        report({"stage": "epoch", **history[-1]})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(output_dir / "pose_classifier_best.pth"))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), str(output_dir / f"pose_classifier_epoch{epoch}.pth"))

    torch.save(model.state_dict(), str(output_dir / "pose_classifier_final.pth"))

    # 导出 ONNX（CPU 推理）
    model.load_state_dict(torch.load(str(output_dir / "pose_classifier_best.pth"), map_location="cpu"))
    model.eval().cpu()
    dummy = torch.randn(1, 32, model.v_dim, 3)
    onnx_path = output_dir / "pose_classifier.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["pose_input"], output_names=["logits"],
        opset_version=17,
        dynamic_axes={"pose_input": {0: "batch"}, "logits": {0: "batch"}},
    )

    summary = {
        "best_val_acc": round(best_val_acc, 2),
        "epochs": epochs,
        "total_samples": len(all_samples),
        "v_dim": v_dim,
        "device": str(device),
        "onnx_path": str(onnx_path),
        "best_pth_path": str(output_dir / "pose_classifier_best.pth"),
        "history": history,
    }
    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report({"stage": "done", "best_val_acc": summary["best_val_acc"], "onnx_path": str(onnx_path)})
    return summary
