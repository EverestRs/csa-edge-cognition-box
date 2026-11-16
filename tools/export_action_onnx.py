from __future__ import annotations

import argparse
from pathlib import Path

import torch


class SlowFastWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, slow_pathway: torch.Tensor, fast_pathway: torch.Tensor) -> torch.Tensor:
        return self.model([slow_pathway, fast_pathway])


def infer_pathway_frames(model: torch.nn.Module) -> tuple[int, int]:
    for module in model.modules():
        if module.__class__.__name__ != "PoolConcatPathway":
            continue
        pool = getattr(module, "pool", None)
        if pool is None or len(pool) < 2:
            continue

        slow_kernel = getattr(pool[0], "kernel_size", None)
        fast_kernel = getattr(pool[1], "kernel_size", None)
        if not slow_kernel or not fast_kernel:
            continue

        slow_t = int(slow_kernel[0])
        fast_t = int(fast_kernel[0])
        if slow_t > 0 and fast_t > 0:
            return slow_t, fast_t

    return 8, 32


def variant_name(slow_t: int, fast_t: int) -> str:
    if (slow_t, fast_t) == (4, 16):
        return "4x16"
    if (slow_t, fast_t) == (8, 32):
        return "8x8"
    return f"unknown(slow={slow_t},fast={fast_t})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SlowFast .pth model to ONNX")
    parser.add_argument("--pth", default="models/action/slowfast.pth", help="input .pth file path")
    parser.add_argument("--onnx", default="models/action/slowfast.onnx", help="output .onnx file path")
    parser.add_argument("--height", type=int, default=224, help="input height")
    parser.add_argument("--width", type=int, default=224, help="input width")
    parser.add_argument("--batch", type=int, default=1, help="batch size for export")
    parser.add_argument("--opset", type=int, default=17, help="onnx opset version")
    args = parser.parse_args()

    pth_path = Path(args.pth)
    onnx_path = Path(args.onnx)

    if not pth_path.exists():
        raise SystemExit(f"[ERROR] pth not found: {pth_path}")

    model_obj = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    if isinstance(model_obj, dict):
        raise SystemExit(
            "[ERROR] Loaded object is a state_dict-like dict. This script expects a serialized model object."
        )

    model = model_obj.eval()
    slow_t, fast_t = infer_pathway_frames(model)
    print(f"[INFO] inferred pathway frames: slow={slow_t}, fast={fast_t}")
    print(f"[INFO] inferred variant: {variant_name(slow_t, fast_t)}")

    wrapper = SlowFastWrapper(model).eval()

    slow = torch.randn(args.batch, 3, slow_t, args.height, args.width, dtype=torch.float32)
    fast = torch.randn(args.batch, 3, fast_t, args.height, args.width, dtype=torch.float32)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (slow, fast),
        str(onnx_path),
        input_names=["slow_pathway", "fast_pathway"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={
            "slow_pathway": {0: "batch"},
            "fast_pathway": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    print(f"[OK] exported: {onnx_path}")

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        print("[INFO] onnx inputs:", [(item.name, item.shape, item.type) for item in sess.get_inputs()])
        print("[INFO] onnx outputs:", [(item.name, item.shape, item.type) for item in sess.get_outputs()])
    except Exception as exc:
        print(f"[WARN] onnx validation skipped/failed: {exc}")


if __name__ == "__main__":
    main()
