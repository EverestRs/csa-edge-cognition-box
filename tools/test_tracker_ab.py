from __future__ import annotations

"""跟踪器真机 A/B（ROADMAP Stage 0a 验证）。

对同一视频跑 YOLOv8 检测，分别喂给"旧版纯 IoU 算法"（脚本内置基线实现）
与新版 Tracker（运动预测 + 两阶段匹配），统计 ID 碎片化指标：
  - track 创建总数（越少越连贯）
  - 平均轨迹年龄（越长越连贯）
  - C1 口径 ID 切换次数（同 track 相邻匹配帧质心跳变 > 2×对角线）

用法:
  python tools/test_tracker_ab.py --video ../data/whole_process_corner_1.mp4 --frames 600
"""

import argparse
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.detector import Detector
from src.core.tracker import Tracker


class LegacyIouTracker:
    """旧版纯 IoU 贪心实现（修复前基线，行为与 git 历史一致）。"""

    def __init__(self, match_threshold: float = 0.8, max_missed: int = 30):
        self.match_threshold = match_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[str, dict] = {}

    @staticmethod
    def _iou_xywh(a, b):
        ax, ay, aw, ah = [float(v) for v in a]
        bx, by, bw, bh = [float(v) for v in b]
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections):
        for track in self._tracks.values():
            track["matched"] = False
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            score = float(det.get("score", 0.0))
            best_id, best_iou = "", 0.0
            for tid, tr in self._tracks.items():
                if tr.get("matched"):
                    continue
                iou = self._iou_xywh(tr["bbox"], bbox)
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id and best_iou >= self.match_threshold:
                t = self._tracks[best_id]
                t.update(bbox=[int(v) for v in bbox], score=score, missed=0, matched=True)
                t["age"] = int(t.get("age", 0)) + 1
            else:
                tid = f"P{self._next_id:03d}"
                self._next_id += 1
                self._tracks[tid] = {"track_id": tid, "bbox": [int(v) for v in bbox], "score": score,
                                     "missed": 0, "age": 1, "matched": True}
        stale = [tid for tid, t in self._tracks.items()
                 if not t.get("matched") and int(t.get("missed", 0)) + 1 > self.max_missed]
        for tid, t in self._tracks.items():
            if not t.get("matched"):
                t["missed"] = int(t.get("missed", 0)) + 1
        for tid in stale:
            self._tracks.pop(tid, None)
        active = []
        for t in self._tracks.values():
            if t.get("matched"):
                active.append({"track_id": t["track_id"], "bbox": list(t["bbox"]),
                               "score": float(t["score"]), "age": int(t.get("age", 1))})
        return active


def count_switches(history: dict[str, list[list[float]]]) -> int:
    """C1 口径：同 track 相邻匹配帧质心跳变 > 2×对角线 记一次切换。"""
    switches = 0
    for boxes in history.values():
        for prev, cur in zip(boxes, boxes[1:]):
            pcx, pcy = prev[0] + prev[2] / 2, prev[1] + prev[3] / 2
            ccx, ccy = cur[0] + cur[2] / 2, cur[1] + cur[3] / 2
            diag = (cur[2] ** 2 + cur[3] ** 2) ** 0.5 or 1.0
            if ((ccx - pcx) ** 2 + (ccy - pcy) ** 2) ** 0.5 > 2.0 * diag:
                switches += 1
    return switches


def run(tracker, detections_stream):
    history: dict[str, list[list[float]]] = {}
    ages = []
    for dets in detections_stream:
        active = tracker.update(dets)
        for t in active:
            history.setdefault(t["track_id"], []).append(t["bbox"])
            ages.append(t["age"])
    return {
        "n_tracks": len(history),
        "mean_age": sum(ages) / max(1, len(ages)),
        "id_switches": count_switches(history),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--detector-stride", type=int, default=1)
    parser.add_argument("--cpu", action="store_true", help="强制 ORT CPU（不占 GPU）")
    args = parser.parse_args()

    model_cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "model_config.yaml", encoding="utf-8"))
    det_cfg = dict(model_cfg.get("detector", {}))
    if args.cpu and det_cfg.get("runtime"):
        det_cfg["runtime"]["execution_provider"] = "CPU"
    detector = Detector(det_cfg)

    cap = cv2.VideoCapture(args.video)
    stream = []
    fid = 0
    while fid < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        if fid % args.detector_stride == 0:
            dets = [d for d in detector.infer(frame) if d.get("class_id", 0) == 0 or d.get("label", "person") == "person"]
            stream.append(dets)
        fid += 1
    cap.release()
    print(f"视频 {args.video}：读取 {len(stream)} 个检测帧")

    legacy = run(LegacyIouTracker(match_threshold=0.8), [list(d) for d in stream])
    modern = run(Tracker(dict(model_cfg.get("tracker", {}))), [list(d) for d in stream])

    print(f"\n{'指标':<16}{'旧版(纯IoU 0.8)':>18}{'新版(预测+两阶段)':>20}")
    for key, label in [("n_tracks", "track 创建数"), ("mean_age", "平均轨迹年龄"), ("id_switches", "C1 口径切换次数")]:
        print(f"{label:<16}{legacy[key]:>18.1f}{modern[key]:>20.1f}")
    ratio = legacy["n_tracks"] / max(1, modern["n_tracks"])
    print(f"\ntrack 碎片化下降 {ratio:.1f}x")
    out = Path(__file__).resolve().parents[1] / "results" / "tracker_ab_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps({"video": args.video, "frames": len(stream), "legacy": legacy, "modern": modern}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
