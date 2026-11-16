from __future__ import annotations

"""评估入口：对测试视频运行推理流水线，与 VLM 伪真值对比，输出量化指标。

用法:
    # 主路径评估（PoseClassifier 主路径 + 场景上下文复核）
    python eval.py --video ../test/whole_process_corner_3.mp4 \
        --ground-truth results/eval/gt/whole_process_corner_3.json

    # 对照组（禁用 PoseClassifier，走 SlowFast 关键词映射基线）
    python eval.py --video ../test/whole_process_corner_3.mp4 \
        --ground-truth results/eval/gt/whole_process_corner_3.json --baseline

    # 消融组（关闭场景上下文注入）
    python eval.py --video ../test/whole_process_corner_3.mp4 \
        --ground-truth results/eval/gt/whole_process_corner_3.json --no-scene-context

说明: 伪真值由 VLM 教师生成（与被评主路径同源，存在乐观偏差，报告需声明）；
     326 帧人工抽检可作为独立锚点。
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluation.evaluator import run_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="系统评估（VLM 伪真值）")
    parser.add_argument("--video", required=True, help="测试视频路径")
    parser.add_argument("--ground-truth", required=True, help="伪真值 JSON 路径")
    parser.add_argument("--profile", default="realtime")
    parser.add_argument("--baseline", action="store_true", help="对照组：禁用 PoseClassifier 主路径")
    parser.add_argument("--no-scene-context", action="store_true", help="消融组：关闭场景上下文注入")
    parser.add_argument("--frame-stride", type=int, default=None, help="处理帧步长（默认取档位配置）")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理到第 N 帧（调试用）")
    parser.add_argument("--output-dir", default="results/eval")
    args = parser.parse_args()

    def on_progress(payload: dict) -> None:
        if payload.get("stage") == "processing":
            print(
                f"\r  处理中 {payload['frame_id']}/{payload['stop_frame']} "
                f"({payload['progress'] * 100:.1f}%)",
                end="", flush=True,
            )

    report = run_evaluation(
        video_path=args.video,
        ground_truth_path=args.ground_truth,
        profile=args.profile,
        baseline=args.baseline,
        use_scene_context=not args.no_scene_context,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        progress_cb=on_progress,
    )
    print()
    metrics = report["metrics"]
    print(f"准确率: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"异常检出率(any): {metrics['abnormal_recall_any']:.4f} | 漏报率: {metrics['false_negative_rate']:.4f} | 误报率: {metrics['false_positive_rate']:.4f}")
    print(f"延迟: mean={report['latency_ms']['mean']}ms p95={report['latency_ms']['p95']}ms")
    print(f"报告: {report['report_file']}")


if __name__ == "__main__":
    main()
