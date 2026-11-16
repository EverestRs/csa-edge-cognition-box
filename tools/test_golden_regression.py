from __future__ import annotations

"""Golden 回归护栏（WEAKNESS_AUDIT 复查#一-4：规则引擎复杂度止血第一步）。

在 VLM 关闭的确定性配置下跑固定视频段，逐帧记录
{frame_id, detections: [{track_id, bbox, label, confidence}]}，作为 golden 基线。
未来对推理引擎的状态机归一/重构（五套复核状态、规则精化、平滑投票）必须通过
本护栏：check 模式逐帧比对（conf 容差 ±0.02，bbox 容差 ±2px）。

用法:
  python tools/test_golden_regression.py capture   # 生成/更新基线
  python tools/test_golden_regression.py check     # 回归比对（退出码 0=通过）
基线: results/golden/main_path_corner_1_200.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.inference_engine import InferenceEngine

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "results/golden/main_path_corner_1_200.json"
CONF_TOL = 0.02
BBOX_TOL = 2.0


def run_baseline(frames: int = 200, start: int = 250) -> list[dict]:
    model_cfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    vlm_cfg = yaml.safe_load((ROOT / "configs" / "vlm_config.yaml").read_text(encoding="utf-8"))
    vlm_cfg["use_vllm"] = False  # 确定性：VLM 关，纯主路径（PC→三级决策→SlowFast 回退→规则）
    engine = InferenceEngine(model_cfg, vlm_cfg, output_dir="results/golden_tmp")
    engine.begin_task("golden")
    cap = cv2.VideoCapture(str(ROOT / "../data/whole_process_corner_1.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)  # 跳过空场景头段，从有人活动段起
    records = []
    fid = start
    end = start + frames
    while fid < end:
        ok, frame = cap.read()
        if not ok:
            break
        res = engine.process(frame, fid)
        records.append({
            "frame_id": fid,
            "detections": [
                {"track_id": d.track_id, "bbox": [int(v) for v in d.bbox],
                 "label": d.behavior.type, "confidence": round(float(d.behavior.confidence), 3)}
                for d in res.detections
            ],
        })
        fid += 1
    cap.release()
    engine.novel_behavior_memo.stop()
    return records


def compare(base: list[dict], cur: list[dict]) -> list[str]:
    diffs = []
    if len(base) != len(cur):
        return [f"帧数不一致: golden={len(base)} current={len(cur)}"]
    for b, c in zip(base, cur):
        if b["frame_id"] != c["frame_id"]:
            diffs.append(f"frame {b['frame_id']}: frame_id 不一致")
            continue
        bd = {d["track_id"]: d for d in b["detections"]}
        cd = {d["track_id"]: d for d in c["detections"]}
        if set(bd) != set(cd):
            diffs.append(f"frame {b['frame_id']}: track 集合不一致 golden={sorted(bd)} current={sorted(cd)}")
            continue
        for tid in bd:
            g, k = bd[tid], cd[tid]
            if g["label"] != k["label"]:
                diffs.append(f"frame {b['frame_id']} {tid}: label {g['label']}→{k['label']}")
            if abs(g["confidence"] - k["confidence"]) > CONF_TOL:
                diffs.append(f"frame {b['frame_id']} {tid}: conf {g['confidence']}→{k['confidence']}")
            if any(abs(a - b2) > BBOX_TOL for a, b2 in zip(g["bbox"], k["bbox"])):
                diffs.append(f"frame {b['frame_id']} {tid}: bbox {g['bbox']}→{k['bbox']}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "check"])
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()
    records = run_baseline(args.frames)
    if args.mode == "capture":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps({"video": "whole_process_corner_1.mp4", "frames": args.frames,
                                      "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
        n_det = sum(len(r["detections"]) for r in records)
        print(f"golden 基线已写入 {GOLDEN}（{len(records)} 帧 / {n_det} 条检测）")
        return 0
    if not GOLDEN.exists():
        print("golden 基线不存在，先运行 capture")
        return 2
    base = json.loads(GOLDEN.read_text(encoding="utf-8"))["records"]
    diffs = compare(base, records)
    if diffs:
        print(f"✗ 回归失败（{len(diffs)} 处差异）:")
        for d in diffs[:20]:
            print("  -", d)
        return 1
    print(f"✓ golden 回归通过（{len(records)} 帧逐帧一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
