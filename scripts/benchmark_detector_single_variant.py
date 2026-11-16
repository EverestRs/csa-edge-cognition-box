from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.detector import Detector
from src.utils.config_loader import load_yaml
from src.utils.preflight import resolve_profile


def signature_of_detections(dets: list[dict]) -> str:
    normalized = []
    for d in dets:
        bbox = d.get("bbox", [0, 0, 0, 0])
        normalized.append(
            {
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "score": round(float(d.get("score", 0.0)), 6),
                "class": str(d.get("class", "person")),
            }
        )
    normalized.sort(key=lambda x: (x["bbox"][0], x["bbox"][1], x["bbox"][2], x["bbox"][3], -x["score"]))
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round(q * (len(sorted_values) - 1)))
    return float(sorted_values[index])


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def merge_cfg(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_cfg(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one detector variant over sampled frames")
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", default="realtime")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--name", required=True)

    parser.add_argument("--onnx-path", default="")
    parser.add_argument("--preprocess-optimized", default="true")
    parser.add_argument("--confidence-threshold", type=float, default=-1.0)
    parser.add_argument("--iou-threshold", type=float, default=-1.0)
    parser.add_argument("--cv-threads", type=int, default=1)

    parser.add_argument("--execution-provider", default="CPUExecutionProvider")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--graph-opt", default="ORT_ENABLE_ALL")
    parser.add_argument("--intra-threads", type=int, default=4)
    parser.add_argument("--inter-threads", type=int, default=1)
    parser.add_argument("--enable-mem-pattern", default="true")
    parser.add_argument("--enable-cpu-arena", default="true")

    parser.add_argument("--baseline-signatures", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(max(1, int(args.cv_threads)))

    model_cfg = load_yaml(PROJECT_ROOT / args.model_config)
    merged_cfg, _ = resolve_profile(model_cfg, args.profile)
    detector_cfg = merged_cfg.get("detector", {})

    patch = {
        "preprocess_optimized": parse_bool(args.preprocess_optimized),
        "runtime": {
            "device": str(args.device),
            "execution_provider": str(args.execution_provider),
            "graph_optimization_level": str(args.graph_opt),
            "intra_op_num_threads": int(max(1, args.intra_threads)),
            "inter_op_num_threads": int(max(1, args.inter_threads)),
            "enable_mem_pattern": parse_bool(args.enable_mem_pattern),
            "enable_cpu_mem_arena": parse_bool(args.enable_cpu_arena),
        },
    }
    if args.onnx_path:
        patch["active_onnx_path"] = args.onnx_path
    if args.confidence_threshold >= 0:
        patch["confidence_threshold"] = float(args.confidence_threshold)
    if args.iou_threshold >= 0:
        patch["iou_threshold"] = float(args.iou_threshold)

    cfg = merge_cfg(detector_cfg, patch)
    detector = Detector(cfg)

    result = {
        "name": args.name,
        "init_ok": detector.session is not None,
        "session_error": detector.session_init_error,
        "sampled_frames": 0,
        "avg_ms": 0.0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "fps": 0.0,
        "providers": [],
        "signatures": [],
    }

    if detector.session is None:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
        return

    try:
        result["providers"] = detector.session.get_providers()
    except Exception:
        result["providers"] = []

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")

    frame_id = 0
    kept = 0
    warmup_done = False
    times_ms: list[float] = []
    signatures: list[str] = []

    while kept < max(2, int(args.frames)):
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % max(1, int(args.sample_step)) != 0:
            continue

        if not warmup_done:
            _ = detector.infer(frame)
            warmup_done = True
            kept += 1
            continue

        t0 = time.perf_counter()
        dets = detector.infer(frame)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(float(dt_ms))
        signatures.append(signature_of_detections(dets))
        kept += 1

    cap.release()

    avg_ms = statistics.mean(times_ms) if times_ms else 0.0
    result["sampled_frames"] = len(times_ms)
    result["avg_ms"] = round(avg_ms, 3)
    result["p50_ms"] = round(statistics.median(times_ms) if times_ms else 0.0, 3)
    result["p95_ms"] = round(percentile(times_ms, 0.95), 3)
    result["fps"] = round(1000.0 / max(1e-6, avg_ms), 3)
    result["signatures"] = signatures

    if args.baseline_signatures:
        baseline_path = Path(args.baseline_signatures)
        if not baseline_path.is_absolute():
            baseline_path = PROJECT_ROOT / baseline_path
        baseline_obj = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_sigs = baseline_obj.get("signatures", [])
        compare_len = min(len(baseline_sigs), len(signatures))
        mismatch = sum(1 for i in range(compare_len) if baseline_sigs[i] != signatures[i])
        result["signature_mismatch_frames"] = int(mismatch)
        result["output_identical_to_baseline"] = bool(mismatch == 0 and compare_len > 0)
        base_avg = float(baseline_obj.get("avg_ms", 0.0))
        if base_avg > 0:
            result["improvement_vs_baseline_pct"] = round((base_avg - avg_ms) * 100.0 / base_avg, 2)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print without heavy signature list.
    lightweight = {k: v for k, v in result.items() if k != "signatures"}
    print(json.dumps(lightweight, ensure_ascii=False))


if __name__ == "__main__":
    main()
