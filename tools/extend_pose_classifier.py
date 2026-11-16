from __future__ import annotations

"""PoseClassifier 分类头扩展（ROADMAP Stage 3 第二步）。

从 pose_classifier_best.pth 出发：分类头 11 → 11+K（旧行权重保留、新行初始化），
冻结骨干微调头（默认），导出 ONNX（dummy 维度读 v_dim，A2 口径），产出扩展标签表。

用法:
  python tools/extend_pose_classifier.py --new-data results/training_data/prototype_classes
      [--epochs 8] [--batch 32] [--device cpu|cuda] [--full-finetune] [--deploy]

产物（默认写到 results/extended_model/，--deploy 才覆盖 models/action/ 部署文件）:
  pose_classifier_extended.pth / .onnx / extended_labels.json
真机微调建议 GPU（V100 上 batch 32 约占 2-3GB，可与用户进程共存）；本工具全流程可在 CPU 用合成小数据机械验证。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.training.pose_classifier import PoseClassifier

OLD_CLASSES = ["Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"]
CLS_IDX = {c: i for i, c in enumerate(OLD_CLASSES)}
OLD_PTH = Path("models/action/pose_classifier_best.pth")
# 溯源校准（E2E 发现）：部署 ONNX 与 final.pth 逐位一致（差 0.0000），
# best.pth 是另一轮训练的遗留——扩展必须锚定生产真值


class NewClassDataset(Dataset):
    def __init__(self, npz_path: str, class_idx: int):
        self.data = np.load(npz_path)["pose_seq"].astype(np.float32)
        self.class_idx = class_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].copy()
        x[:, :, 0] /= 1920.0   # 与训练侧 FastPoseDataset 同口径（NPZ 存原始像素）
        x[:, :, 1] /= 1080.0
        x[:, :, 2] = np.clip(x[:, :, 2], 0.0, 1.0)
        return torch.from_numpy(x), self.class_idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-data", required=True, help="build_prototype_training_data.py 输出目录")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--full-finetune", action="store_true", help="全参微调（默认只训新头，防遗忘）")
    ap.add_argument("--deploy", action="store_true", help="覆盖 models/action/ 部署文件（默认只写 results/extended_model/）")
    ap.add_argument("--rehearsal-v2", action="store_true", default=True, help="用 training_data_v2（当前口径）做回放")
    args = ap.parse_args()
    args_rehearsal_augment = [False]

    new_dir = Path(args.new_data)
    classes_meta = json.loads((new_dir / "new_classes.json").read_text(encoding="utf-8"))["classes"]
    new_labels = [c["label"] for c in classes_meta]
    num_classes = len(OLD_CLASSES) + len(new_labels)
    print(f"扩展: {len(OLD_CLASSES)} + {len(new_labels)} ({new_labels}) = {num_classes} 类")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = PoseClassifier(num_classes=num_classes)
    state = torch.load(OLD_PTH, map_location="cpu", weights_only=True)
    # 剔除旧头（形状不匹配，strict=False 也不容忍 size mismatch），骨干直接加载
    head_keys = {k for k in state if k.startswith("classifier.4")}
    partial = {k: v for k, v in state.items() if k not in head_keys}
    missing, unexpected = model.load_state_dict(partial, strict=False)
    # 旧头权重拷入扩展头前 11 行，新行零初始化
    with torch.no_grad():
        model.classifier[4].weight[: len(OLD_CLASSES)].copy_(state["classifier.4.weight"])
        model.classifier[4].bias[: len(OLD_CLASSES)].copy_(state["classifier.4.bias"])
        nn.init.zeros_(model.classifier[4].bias[len(OLD_CLASSES):])
        nn.init.kaiming_uniform_(model.classifier[4].weight[len(OLD_CLASSES):])
    print(f"加载旧权重: missing={len(missing)} unexpected={len(unexpected)}（应仅为扩展头行）")

    datasets = [NewClassDataset(c["npz"], class_idx=len(OLD_CLASSES) + i) for i, c in enumerate(classes_meta)]
    # 旧类回放（rehearsal）：单类微调会把 softmax 其它类全压掉（E2E 实测 Normal 全误报 N01），
    # 混入旧 manifest 数据共同参与 loss，保旧类 logits
    rehearsal_n = 0
    if args.rehearsal_v2:
        # v2 口径回放（默认）：regen 数据与扩展 base（v2）同口径，NPZ 已归一化（regen 后
        # V2Dataset 口径）不再除 1920/1080——旧 manifest 回放是旧口径，会重蹈根因 3
        v2_manifest = json.loads(Path("results/training_data_v2/manifest.json").read_text(encoding="utf-8"))
        v2_npz = Path("results/training_data_v2/npz")
        rows = []
        cache: dict[str, tuple] = {}
        for m in v2_manifest:
            video = m["video"]
            if video not in cache:
                d = np.load(v2_npz / f"{Path(video).stem}.npz", allow_pickle=True)
                cache[video] = (d["frame_id"].astype(int), d["pose_seq"].astype(np.float32))
            fids, seqs = cache[video]
            key = np.where(fids == int(m["frame_id"]))[0]
            if not len(key):
                continue
            x = seqs[int(key[0])].copy()
            x[..., 2] = np.clip(x[..., 2], 0.0, 1.0)
            rows.append((x, CLS_IDX[m["refined_label"]]))
        import random
        rng = random.Random(7)
        rng.shuffle(rows)
        rows = rows[:1500]

        class V2Rehearsal(Dataset):
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                x, y = self.items[i]
                if args_rehearsal_augment[0]:
                    x = x.copy()
                    x[..., :2] += np.random.randn(*x[..., :2].shape).astype(np.float32) * 0.003
                return torch.from_numpy(x), y

        args_rehearsal_augment[0] = True
        if rows:
            datasets.insert(0, V2Rehearsal(rows))
            rehearsal_n = len(rows)
    else:
        rehearsal_n = 0
        manifest_path = Path("results/training_data/manifest.json")
        if manifest_path.exists():
            from src.training.pose_classifier import FastPoseDataset
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            data_dir = manifest_path.parent
            existing = {p.stem for p in data_dir.glob("*.npz")}
            usable = [s for s in manifest if Path(s["video"]).stem in existing]
            by_cls: dict[int, list[dict]] = {}
            for s in usable:
                by_cls.setdefault(int(s["class_idx"]), []).append(s)
            import random
            rng = random.Random(7)
            subset = []
            for idx, items in sorted(by_cls.items()):
                rng.shuffle(items)
                subset.extend(items[:400])  # 每旧类封顶 400
            if subset:
                old_ds = FastPoseDataset(subset, data_dir, augment=True)
                datasets.insert(0, old_ds)
                rehearsal_n = len(old_ds)
    loader = DataLoader(torch.utils.data.ConcatDataset(datasets),
                        batch_size=min(args.batch, min(len(d) for d in datasets)),
                        shuffle=True, num_workers=0)
    print(f"训练集: 旧类回放 {rehearsal_n} + 新类 {sum(len(d) for d in datasets[1:] if isinstance(d, NewClassDataset))}")
    if not args.full_finetune:
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False
    model.to(device)
    # 全程 eval：冻结所有 BN running statistics 与 dropout（否则新数据分布搬动旧类 logits）
    model.eval()
    # ROADMAP Stage 3 最终方案：冻结特征上逐新类拟合各自行（加权 logistic）——
    # 旧头行 bit 级不动，旧类输出与原模型一致；每个新行学习"本类 vs 其余全部"。
    # （逐 batch 联合训练在少量新样本下要么欠拟合要么经 BN/softmax 漂移旧类，均已实测否决）
    feats, ys = [], []
    hook_out = {}
    handle = model.classifier[3].register_forward_hook(lambda m, i, o: hook_out.__setitem__("f", o.detach()))
    with torch.no_grad():
        for xb, yb in loader:
            model(xb.to(device))
            feats.append(hook_out["f"].cpu())
            ys.append(yb)
    handle.remove()
    F_all = torch.cat(feats)
    Y_all = torch.cat(ys)
    k_base = len(OLD_CLASSES)
    # 多新类联合拟合：新类样本上新类行间 softmax 互分（校准行间尺度——独立 one-vs-rest
    # 会因各自行 pos_weight 不同导致 logit 尺度失衡，E2E 实测全被 N01 抢走）；
    # 旧类样本上所有新行压负（保旧类 argmax 不被抢）。
    rows_w, rows_b = [], []
    for k_new in range(k_base, num_classes):
        rows_w.append(model.classifier[4].weight[k_new].detach().clone().requires_grad_(True))
        rows_b.append(model.classifier[4].bias[k_new].detach().clone().requires_grad_(True))
    if num_classes - k_base == 1:
        # 单新类：历史验证路径——BCE(pos_weight) 大权重正例，行向量充分对齐正类特征方向
        # （该路径曾验收：留出 100% 命中 / 旧类 0% 误报 / 新旧一致 100%）
        W, B = torch.stack(rows_w), torch.stack(rows_b)
        pos_weight = torch.tensor(float((Y_all < k_base).sum()) / max(1.0, float((Y_all == k_base).sum())))
        opt = torch.optim.AdamW(rows_w + rows_b, lr=5e-3, weight_decay=1e-3)
        t_all = (Y_all == k_base).float()
        for ep in range(400):
            opt.zero_grad()
            z = (F_all.to(device) @ W.T + B).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                z, t_all.to(device), pos_weight=pos_weight)
            loss.backward()
            opt.step()
            if (ep + 1) % 100 == 0:
                with torch.no_grad():
                    acc = float(((z[0] > 0).float().cpu() == t_all).float().mean())
                print(f"  单行拟合 epoch {ep + 1}/400: bce={float(loss):.4f} acc={acc:.2%}")
        with torch.no_grad():
            model.classifier[4].weight[k_base] = rows_w[0]
            model.classifier[4].bias[k_base] = rows_b[0]
        print("  单新类行拟合完成（BCE pos_weight 路径）")
        out_dir = Path("results/extended_model")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "pose_classifier_extended.pth")
        model_cpu = model.cpu().eval()
        dummy = torch.randn(1, 32, model_cpu.v_dim, 3)
        torch.onnx.export(model_cpu, dummy, str(out_dir / "pose_classifier_extended.onnx"),
                          opset_version=17, input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}})
        labels = {"classes": OLD_CLASSES + [classes_meta[0]["label"]],
                  "class_to_idx": {c: i for i, c in enumerate(OLD_CLASSES + [classes_meta[0]["label"]])},
                  "new_classes": classes_meta, "base_checkpoint": str(OLD_PTH)}
        (out_dir / "extended_labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
        import onnxruntime as _ort
        sess = _ort.InferenceSession(str(out_dir / "pose_classifier_extended.onnx"), providers=["CPUExecutionProvider"])
        o = sess.run(None, {"input": np.zeros((1, 32, model_cpu.v_dim, 3), np.float32)})[0]
        assert o.shape == (1, num_classes), o.shape
        print(f"ONNX 导出验证 OK: 输出 (1, {num_classes})")
        if args.deploy:
            import shutil as _sh
            _sh.copy(out_dir / "pose_classifier_extended.onnx", "models/action/pose_classifier.onnx")
            _sh.copy(out_dir / "pose_classifier_extended.pth", "models/action/pose_classifier_best.pth")
            _sh.copy(out_dir / "extended_labels.json", "models/action/pose_classifier_labels.json")
            print("已部署到 models/action/（重启 API 服务生效）")
        else:
            print(f"产物在 {out_dir}（未部署）")
        return 0
    W = torch.stack(rows_w)
    B = torch.stack(rows_b)
    for ep in range(400):
        opt.zero_grad()
        logits_new = F_all.to(device) @ W.T + B  # (N, K_new)
        is_new = (Y_all >= k_base).to(device)
        loss_new = torch.tensor(0.0, device=device)
        if is_new.any():
            targets = (Y_all.to(device)[is_new] - k_base)
            # 类不均衡权重：新类正例（7-19条）远少于回放负例（1526），无权重则正例梯度被
            # 稀释+weight_decay 回收行向量（E2E 实测留出命中掉 0%）
            cls_count = torch.stack([(Y_all == k_base + i).sum() for i in range(num_classes - k_base)]).float().to(device)
            w_ce = cls_count.sum() / cls_count.clamp(min=1.0)
            loss_new = nn.functional.cross_entropy(logits_new[is_new], targets, weight=w_ce)
        loss_old = torch.tensor(0.0, device=device)
        if (~is_new).any():
            loss_old = nn.functional.binary_cross_entropy_with_logits(
                logits_new[~is_new], torch.zeros_like(logits_new[~is_new]))
        loss = 5.0 * loss_new + 0.5 * loss_old
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            with torch.no_grad():
                pred_new = logits_new.argmax(1)
                acc_new = float((pred_new[is_new] == (Y_all.to(device)[is_new] - k_base)).float().mean()) if is_new.any() else 0.0
                fp_old = float((logits_new[~is_new] > 0).any(1).float().mean()) if (~is_new).any() else 0.0
            print(f"  新类联合拟合 epoch {ep + 1}/400: loss={float(loss):.4f} 新类互分acc={acc_new:.2%} 旧类误触发率={fp_old:.2%}")
    with torch.no_grad():
        for i in range(len(rows_w)):
            model.classifier[4].weight[k_base + i] = rows_w[i]
            model.classifier[4].bias[k_base + i] = rows_b[i]
    print(f"  {num_classes - k_base} 个新类行联合拟合完成")

    out_dir = Path("results/extended_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "pose_classifier_extended.pth")
    model_cpu = model.cpu().eval()
    dummy = torch.randn(1, 32, model_cpu.v_dim, 3)
    torch.onnx.export(model_cpu, dummy, str(out_dir / "pose_classifier_extended.onnx"),
                      opset_version=17, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}})
    labels = {"classes": OLD_CLASSES + new_labels,
              "class_to_idx": {c: i for i, c in enumerate(OLD_CLASSES + new_labels)},
              "new_classes": classes_meta, "base_checkpoint": str(OLD_PTH)}
    (out_dir / "extended_labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")

    # ONNX 冒烟：输出形状 (1, num_classes)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_dir / "pose_classifier_extended.onnx"), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": np.zeros((1, 32, model_cpu.v_dim, 3), np.float32)})[0]
    assert out.shape == (1, num_classes), out.shape
    print(f"ONNX 导出验证 OK: 输出 (1, {num_classes})")
    if args.deploy:
        shutil.copy(out_dir / "pose_classifier_extended.onnx", "models/action/pose_classifier.onnx")
        shutil.copy(out_dir / "pose_classifier_extended.pth", "models/action/pose_classifier_best.pth")
        shutil.copy(out_dir / "extended_labels.json", "models/action/pose_classifier_labels.json")
        print("已部署到 models/action/（重启 API 服务生效）")
    else:
        print(f"产物在 {out_dir}（未部署；--deploy 覆盖 models/action/）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
