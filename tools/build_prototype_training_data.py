from __future__ import annotations

"""原型簇 → 训练数据（ROADMAP Stage 3 第一步）。

把已批准原型（status=active）簇内备忘记录的 pose_seq（Stage 1 落盘的 NPZ）
汇编为扩展类训练数据：
  results/training_data/prototype_classes/
    ├── new_classes.json      {label: "N01", proto_id, name, n_samples}
    └── class_N01.npz         pose_seq (M, 32, 133, 3)，整批同一新类

用法:
  python tools/build_prototype_training_data.py [--prototypes results/prototypes/prototypes.json]
前置：原型已人工批准（tools/review_prototypes.py apply）；成员备忘含 pose_seq_file。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

T_DIM, V_DIM = 32, 133


def _load_pose_seq(npz_path: str) -> np.ndarray | None:
    try:
        d = np.load(npz_path)
        seq = d["pose_seq"].astype(np.float32)
    except Exception:
        return None
    # 布局 (T, V, C)：V∈{17,133}，C=3 (x,y,conf)
    if seq.ndim != 3 or seq.shape[2] != 3 or seq.shape[1] not in (17, 133) or seq.shape[0] < 8:
        return None
    if seq.shape[1] == 17:  # 17 点补零对齐 133（与 D3 口径一致）
        seq = np.pad(seq, ((0, 0), (0, 133 - 17), (0, 0)))
    if seq.shape[0] < T_DIM:  # 不足 32 帧：重复末帧补齐（短片段）
        pad = np.repeat(seq[-1:], T_DIM - seq.shape[0], axis=0)
        seq = np.concatenate([seq, pad], axis=0)
    return seq[:T_DIM]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prototypes", default="results/prototypes/prototypes.json")
    ap.add_argument("--novel-root", default="results/novel")
    ap.add_argument("--out", default="results/training_data/prototype_classes")
    args = ap.parse_args()

    store = json.loads(Path(args.prototypes).read_text(encoding="utf-8"))
    active = [p for p in store.get("prototypes", []) if p.get("status") == "active" and p.get("name")]
    if not active:
        print("无已批准原型——先执行 tools/review_prototypes.py name/apply")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, proto in enumerate(active):
        label = f"N{i + 1:02d}"
        seqs = []
        missing = []
        for key in proto.get("member_memo_keys", []):
            memo_path = Path(args.novel_root) / key
            try:
                rec = json.loads(memo_path.read_text(encoding="utf-8"))
            except Exception:
                missing.append(key)
                continue
            seq_file = rec.get("pose_seq_file")
            if not seq_file or not Path(seq_file).exists():
                missing.append(key)
                continue
            seq = _load_pose_seq(seq_file)
            if seq is not None:
                seqs.append(seq)
        if len(seqs) < 3:
            print(f"  跳过 {proto['proto_id']}（{proto['name']}）：有效姿态序列 {len(seqs)} < 3（缺失 {len(missing)}）")
            continue
        arr = np.stack(seqs)
        npz = out_dir / f"class_{label}.npz"
        np.savez_compressed(npz, pose_seq=arr)
        manifest.append({
            "label": label, "proto_id": proto["proto_id"], "name": proto["name"],
            "n_samples": int(arr.shape[0]), "npz": str(npz),
            "features": proto.get("features", ""),
        })
        print(f"  {label} ← {proto['name']}（{proto['proto_id']}）: {arr.shape[0]} 条序列")
    if not manifest:
        print("没有任何原型具备足够姿态序列，未生成数据")
        return 1
    (out_dir / "new_classes.json").write_text(json.dumps({"classes": manifest}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"扩展类清单: {out_dir / 'new_classes.json'}（{len(manifest)} 个新类，共 {sum(m['n_samples'] for m in manifest)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
