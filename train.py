from __future__ import annotations

"""PoseClassifier 训练入口（收敛自 tools/train_pose_classifier_v2.py）。

用法:
    python train.py                                  # 默认参数训练
    python train.py --epochs 50 --batch-size 32      # 自定义参数
    python train.py --device cpu                     # 指定设备
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.training.pose_classifier import train_pose_classifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="PoseClassifier 训练（VLM-as-Teacher 伪标签）")
    parser.add_argument("--data-dir", default=str(BASE_DIR / "results" / "training_data"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "models" / "action_v2"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    def on_progress(payload: dict) -> None:
        if payload.get("stage") == "epoch":
            print(
                f"Epoch {payload['epoch']}/{args.epochs} | "
                f"train_loss={payload['train_loss']:.4f} acc={payload['train_acc']:.2f}% | "
                f"val_loss={payload['val_loss']:.4f} acc={payload['val_acc']:.2f}%"
            )
        elif payload.get("stage") == "data_loaded":
            print(
                f"[DATA] 总样本 {payload['total_samples']} | "
                f"训练 {payload['train_samples']} / 验证 {payload['val_samples']} | "
                f"v_dim={payload['v_dim']} | device={payload['device']}"
            )

    summary = train_pose_classifier(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        progress_cb=on_progress,
    )
    print(f"\n训练完成！最佳验证准确率: {summary['best_val_acc']:.2f}%")
    print(f"ONNX: {summary['onnx_path']}")


if __name__ == "__main__":
    main()
