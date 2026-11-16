from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
import sys

import cv2
import numpy as np

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


def collect_sample_frame_indices(video: str, max_frames: int, sample_step: int) -> list[int]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")

    frame_indices: list[int] = []
    frame_id = 0
    while len(frame_indices) < max_frames:
        ok, _ = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % sample_step != 0:
            continue
        frame_indices.append(frame_id)

    cap.release()
    if not frame_indices:
        raise RuntimeError("no frames sampled")
    return frame_indices


def run_variant(video: str, frame_indices: list[int], detector_cfg: dict, cv_threads: int) -> dict:
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(int(max(1, cv_threads)))

    try:
        detector = Detector(detector_cfg)
    except Exception as exc:
        return {
            "init_ok": False,
            "session_error": f"detector_init_exception:{type(exc).__name__}",
            "providers": [],
            "times_ms": [],
            "signatures": [],
        }
    if detector.session is None:
        return {
            "init_ok": False,
            "session_error": detector.session_init_error or "session_not_ready",
            "providers": [],
            "times_ms": [],
            "signatures": [],
        }

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return {
            "init_ok": False,
            "session_error": "video_open_failed",
            "providers": [],
            "times_ms": [],
            "signatures": [],
        }

    times_ms: list[float] = []
    signatures: list[str] = []
    target_set = set(frame_indices)
    max_target = frame_indices[-1]
    frame_id = 0
    warmed_up = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id > max_target:
                break
            if frame_id not in target_set:
                continue

            if not warmed_up:
                _ = detector.infer(frame)
                warmed_up = True
                continue

            t0 = time.perf_counter()
            dets = detector.infer(frame)
            elapsed = (time.perf_counter() - t0) * 1000.0
            times_ms.append(float(elapsed))
            signatures.append(signature_of_detections(dets))

            if len(times_ms) >= max(0, len(frame_indices) - 1):
                break
    except Exception as exc:
        cap.release()
        return {
            "init_ok": False,
            "session_error": f"infer_exception:{type(exc).__name__}",
            "providers": [],
            "times_ms": [],
            "signatures": [],
        }

    cap.release()

    providers = []
    try:
        providers = detector.session.get_providers()
    except Exception:
        providers = []

    return {
        "init_ok": True,
        "session_error": "",
        "providers": providers,
        "times_ms": times_ms,
        "signatures": signatures,
    }


def summarize(name: str, result: dict, base_signatures: list[str]) -> dict:
    if not result.get("init_ok", False):
        return {
            "name": name,
            "init_ok": False,
            "session_error": result.get("session_error", "session_not_ready"),
            "providers": result.get("providers", []),
            "sampled_frames": 0,
        }

    times = result["times_ms"]
    signatures = result["signatures"]
    compare_len = min(len(base_signatures), len(signatures))
    mismatches = sum(1 for i in range(compare_len) if base_signatures[i] != signatures[i])
    avg_ms = statistics.mean(times) if times else 0.0
    return {
        "name": name,
        "init_ok": True,
        "session_error": "",
        "providers": result.get("providers", []),
        "sampled_frames": len(times),
        "avg_ms": round(avg_ms, 3),
        "p50_ms": round(statistics.median(times) if times else 0.0, 3),
        "p95_ms": round(percentile(times, 0.95), 3),
        "fps": round(1000.0 / max(1e-6, avg_ms), 3),
        "signature_mismatch_frames": int(mismatches),
        "output_identical_to_baseline": bool(mismatches == 0),
    }


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Detector optimization matrix benchmark with streaming frame sampling")
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", default="realtime")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--output", default="results/detector_300f_matrix.json")
    args = parser.parse_args()

    model_cfg = load_yaml(PROJECT_ROOT / args.model_config)
    merged_cfg, _ = resolve_profile(model_cfg, args.profile)
    base_detector_cfg = merged_cfg.get("detector", {})

    frame_indices = collect_sample_frame_indices(args.video, max_frames=max(2, int(args.frames)), sample_step=max(1, int(args.sample_step)))
    model_dir = PROJECT_ROOT / "models" / "detector"
    yolov8n_fp32 = "models/detector/yolov8n.onnx"
    yolov8n_int8 = "models/detector/yolov8n_int8.onnx"
    has_yolov8n_fp32 = (model_dir / "yolov8n.onnx").exists()
    has_yolov8n_int8 = (model_dir / "yolov8n_int8.onnx").exists()

    variants: list[tuple[str, dict, int]] = [
        (
            "baseline_cpu_opt_preprocess_on",
            deep_merge(
                base_detector_cfg,
                {
                    "preprocess_optimized": True,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_ALL",
                        "intra_op_num_threads": 4,
                        "inter_op_num_threads": 2,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": True,
                    },
                },
            ),
            1,
        ),
        (
            "group1_preprocess_off",
            deep_merge(
                base_detector_cfg,
                {
                    "preprocess_optimized": False,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_ALL",
                        "intra_op_num_threads": 4,
                        "inter_op_num_threads": 2,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": True,
                    },
                },
            ),
            1,
        ),
        (
            "group1_preprocess_on",
            deep_merge(
                base_detector_cfg,
                {
                    "preprocess_optimized": True,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_ALL",
                        "intra_op_num_threads": 4,
                        "inter_op_num_threads": 2,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": True,
                    },
                },
            ),
            1,
        ),
        (
            "group2_threads_2x1_opt_all",
            deep_merge(
                base_detector_cfg,
                {
                    "preprocess_optimized": True,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_ALL",
                        "intra_op_num_threads": 2,
                        "inter_op_num_threads": 1,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": True,
                    },
                },
            ),
            2,
        ),
        (
            "group2_threads_4x1_opt_all",
            deep_merge(
                base_detector_cfg,
                {
                    "preprocess_optimized": True,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_ALL",
                        "intra_op_num_threads": 4,
                        "inter_op_num_threads": 1,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": True,
                    },
                },
            ),
            2,
        ),
        (
            "group2_threads_6x1_opt_all",
            deep_merge(
                base_detector_cfg,
                {
                    "preprocess_optimized": True,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_ALL",
                        "intra_op_num_threads": 6,
                        "inter_op_num_threads": 1,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": True,
                    },
                },
            ),
            2,
        ),
        (
            "group2_opt_basic_threads_4x1",
            deep_merge(
                base_detector_cfg,
                {
                    "preprocess_optimized": True,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_BASIC",
                        "intra_op_num_threads": 4,
                        "inter_op_num_threads": 1,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": True,
                    },
                },
            ),
            2,
        ),
        (
            "group2_opt_extended_threads_4x1",
            deep_merge(
                base_detector_cfg,
                {
                    "preprocess_optimized": True,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_EXTENDED",
                        "intra_op_num_threads": 4,
                        "inter_op_num_threads": 1,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": True,
                    },
                },
            ),
            2,
        ),
        (
            "group2_no_mem_arena_threads_4x1",
            deep_merge(
                base_detector_cfg,
                {
                    "preprocess_optimized": True,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_ALL",
                        "intra_op_num_threads": 4,
                        "inter_op_num_threads": 1,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": False,
                    },
                },
            ),
            2,
        ),
        (
            "group3_conf_040",
            deep_merge(
                base_detector_cfg,
                {
                    "confidence_threshold": 0.40,
                    "preprocess_optimized": True,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_ALL",
                        "intra_op_num_threads": 4,
                        "inter_op_num_threads": 1,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": True,
                    },
                },
            ),
            1,
        ),
        (
            "group3_iou_045",
            deep_merge(
                base_detector_cfg,
                {
                    "iou_threshold": 0.45,
                    "preprocess_optimized": True,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_ALL",
                        "intra_op_num_threads": 4,
                        "inter_op_num_threads": 1,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": True,
                    },
                },
            ),
            1,
        ),
        (
            "group3_cv_threads_1",
            deep_merge(
                base_detector_cfg,
                {
                    "preprocess_optimized": True,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_ALL",
                        "intra_op_num_threads": 4,
                        "inter_op_num_threads": 1,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": True,
                    },
                },
            ),
            1,
        ),
        (
            "group3_cv_threads_2",
            deep_merge(
                base_detector_cfg,
                {
                    "preprocess_optimized": True,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_ALL",
                        "intra_op_num_threads": 4,
                        "inter_op_num_threads": 1,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": True,
                    },
                },
            ),
            2,
        ),
        (
            "group3_cv_threads_4",
            deep_merge(
                base_detector_cfg,
                {
                    "preprocess_optimized": True,
                    "runtime": {
                        "device": "cpu",
                        "execution_provider": "CPUExecutionProvider",
                        "graph_optimization_level": "ORT_ENABLE_ALL",
                        "intra_op_num_threads": 4,
                        "inter_op_num_threads": 1,
                        "enable_mem_pattern": True,
                        "enable_cpu_mem_arena": True,
                    },
                },
            ),
            4,
        ),
    ]

    if has_yolov8n_fp32:
        variants.append(
            (
                "group2_model_yolov8n_fp32_threads_4x1",
                deep_merge(
                    base_detector_cfg,
                    {
                        "active_onnx_path": yolov8n_fp32,
                        "preprocess_optimized": True,
                        "runtime": {
                            "device": "cpu",
                            "execution_provider": "CPUExecutionProvider",
                            "graph_optimization_level": "ORT_ENABLE_ALL",
                            "intra_op_num_threads": 4,
                            "inter_op_num_threads": 1,
                            "enable_mem_pattern": True,
                            "enable_cpu_mem_arena": True,
                        },
                    },
                ),
                2,
            )
        )

    if has_yolov8n_int8:
        variants.append(
            (
                "group2_model_yolov8n_int8_threads_4x1",
                deep_merge(
                    base_detector_cfg,
                    {
                        "active_onnx_path": yolov8n_int8,
                        "preprocess_optimized": True,
                        "runtime": {
                            "device": "cpu",
                            "execution_provider": "CPUExecutionProvider",
                            "graph_optimization_level": "ORT_ENABLE_ALL",
                            "intra_op_num_threads": 4,
                            "inter_op_num_threads": 1,
                            "enable_mem_pattern": True,
                            "enable_cpu_mem_arena": True,
                        },
                    },
                ),
                2,
            )
        )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_results: dict[str, dict] = {}
    for index, (name, cfg, cv_threads) in enumerate(variants, start=1):
        print(f"[{index}/{len(variants)}] running {name} ...", flush=True)
        raw_results[name] = run_variant(args.video, frame_indices, cfg, cv_threads=cv_threads)

        partial = {
            "video": args.video,
            "profile": args.profile,
            "sample_step": int(args.sample_step),
            "sampled_frames_per_variant": max(0, len(frame_indices) - 1),
            "baseline": "baseline_cpu_opt_preprocess_on",
            "progress": {"completed_variants": index, "total_variants": len(variants)},
            "raw_results_keys": list(raw_results.keys()),
        }
        output_path.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_key = "baseline_cpu_opt_preprocess_on"
    baseline = raw_results[baseline_key]
    baseline_signatures = baseline.get("signatures", []) if baseline.get("init_ok") else []

    summary_rows = [summarize(name, result, baseline_signatures) for name, result in raw_results.items()]

    baseline_avg = 0.0
    for row in summary_rows:
        if row["name"] == baseline_key and row.get("init_ok"):
            baseline_avg = float(row.get("avg_ms", 0.0))
            break

    for row in summary_rows:
        if row.get("init_ok") and baseline_avg > 0.0:
            current = float(row.get("avg_ms", 0.0))
            row["improvement_vs_baseline_pct"] = round((baseline_avg - current) * 100.0 / baseline_avg, 2)

    output = {
        "video": args.video,
        "profile": args.profile,
        "sample_step": int(args.sample_step),
        "sampled_frames_per_variant": max(0, len(frame_indices) - 1),
        "baseline": baseline_key,
        "rows": summary_rows,
    }

    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
