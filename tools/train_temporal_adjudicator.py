from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(1e-8, np.sum(exp, axis=1, keepdims=True))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def build_dataset(rows: list[dict[str, Any]], class_order: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    filtered = [r for r in rows if str(r.get("target_class", "")).strip()]
    if not filtered:
        raise RuntimeError("no labeled rows with target_class found in dataset")

    feature_names = [str(x) for x in filtered[0].get("feature_names", [])]
    X = np.array([r.get("feature_vector", []) for r in filtered], dtype=np.float32)
    labels = [str(r.get("target_class", "")).strip() for r in filtered]

    label_set = set(labels)
    classes = [c for c in class_order if c in label_set]
    remaining = sorted([c for c in label_set if c not in set(classes)])
    classes.extend(remaining)

    if len(classes) < 2:
        raise RuntimeError(f"need at least 2 classes, got {classes}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in labels], dtype=np.int64)
    return X, y, feature_names, classes


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    if n <= 1:
        return X, y, X[:0], y[:0]

    ratio = float(min(0.5, max(0.0, val_ratio)))
    if ratio <= 0.0:
        return X, y, X[:0], y[:0]

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for cls in sorted(set(int(v) for v in y.tolist())):
        cls_idx = np.where(y == cls)[0]
        cls_idx = rng.permutation(cls_idx)
        m = len(cls_idx)
        if m <= 1:
            train_idx.extend(int(i) for i in cls_idx)
            continue
        m_val = int(round(m * ratio))
        m_val = max(1, min(m - 1, m_val))
        val_idx.extend(int(i) for i in cls_idx[:m_val])
        train_idx.extend(int(i) for i in cls_idx[m_val:])

    train_idx_np = np.array(train_idx, dtype=np.int64)
    val_idx_np = np.array(val_idx, dtype=np.int64)
    return X[train_idx_np], y[train_idx_np], X[val_idx_np], y[val_idx_np]


def compute_class_weights(y: np.ndarray, num_classes: int, mode: str) -> np.ndarray:
    mode = str(mode or "none").strip().lower()
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)

    if mode == "none":
        return np.ones((num_classes,), dtype=np.float32)

    eps = 1e-6
    if mode == "inverse_sqrt":
        inv = 1.0 / np.sqrt(np.maximum(eps, counts))
    else:
        inv = 1.0 / np.maximum(eps, counts)

    inv *= float(num_classes) / max(eps, float(np.sum(inv)))
    return inv.astype(np.float32)


def train_multiclass_logreg(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    class_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n, d = X.shape
    W = np.zeros((num_classes, d), dtype=np.float32)
    b = np.zeros((num_classes,), dtype=np.float32)

    Y = np.zeros((n, num_classes), dtype=np.float32)
    Y[np.arange(n), y] = 1.0
    sample_weights = class_weights[y].reshape(-1, 1)
    sw_sum = float(max(1e-6, np.sum(sample_weights)))

    for _ in range(max(1, epochs)):
        logits = X @ W.T + b
        probs = softmax(logits)

        grad_logits = (probs - Y) * sample_weights / sw_sum
        grad_W = grad_logits.T @ X + weight_decay * W
        grad_b = np.sum(grad_logits, axis=0)

        W -= lr * grad_W
        b -= lr * grad_b

    return W, b


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    if len(y) <= 0:
        return {"samples": 0}

    n = int(len(y))
    logits = X @ W.T + b
    probs = softmax(logits)
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y))

    eps = 1e-8
    nll = -np.sum(np.log(np.maximum(eps, probs[np.arange(n), y]))) / float(n)

    k = len(classes)
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y.tolist(), pred.tolist()):
        cm[int(t), int(p)] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for i, cls_name in enumerate(classes):
        tp = float(cm[i, i])
        fp = float(np.sum(cm[:, i]) - tp)
        fn = float(np.sum(cm[i, :]) - tp)
        support = int(np.sum(cm[i, :]))
        precision = tp / max(1e-8, tp + fp)
        recall = tp / max(1e-8, tp + fn)
        f1 = (2.0 * precision * recall) / max(1e-8, precision + recall)
        f1_values.append(float(f1))
        per_class[cls_name] = {
            "support": support,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        }

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    return {
        "samples": int(len(y)),
        "accuracy": round(acc, 4),
        "nll": round(float(nll), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight temporal adjudicator linear model")
    parser.add_argument("--dataset", required=True, help="jsonl dataset exported by export_temporal_adjudicator_dataset.py")
    parser.add_argument("--output", required=True, help="output model json")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight-mode", choices=["none", "inverse_freq", "inverse_sqrt"], default="inverse_freq")
    parser.add_argument("--class-order", default="direct_fall,risk_then_fall,non_fall")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset))
    class_order = [item.strip() for item in str(args.class_order).split(",") if item.strip()]
    X, y, feature_names, classes = build_dataset(rows, class_order=class_order)
    X_train, y_train, X_val, y_val = stratified_split(
        X=X,
        y=y,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    class_weights = compute_class_weights(
        y=y_train,
        num_classes=len(classes),
        mode=str(args.class_weight_mode),
    )

    W, b = train_multiclass_logreg(
        X=X_train,
        y=y_train,
        num_classes=len(classes),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        class_weights=class_weights,
    )

    train_metrics = evaluate_model(X_train, y_train, W, b, classes)
    val_metrics = evaluate_model(X_val, y_val, W, b, classes)
    class_weight_dict = {cls: round(float(class_weights[i]), 6) for i, cls in enumerate(classes)}

    output = {
        "model_type": "multiclass_logistic_regression",
        "classes": classes,
        "feature_names": feature_names,
        "weights": W.tolist(),
        "bias": b.tolist(),
        "temperature": float(max(1e-6, args.temperature)),
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "class_weights": class_weight_dict,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "output": str(out_path),
        "classes": classes,
        "train_samples": int(train_metrics.get("samples", 0)),
        "val_samples": int(val_metrics.get("samples", 0)),
        "train_accuracy": train_metrics.get("accuracy"),
        "val_accuracy": val_metrics.get("accuracy"),
        "train_macro_f1": train_metrics.get("macro_f1"),
        "val_macro_f1": val_metrics.get("macro_f1"),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
