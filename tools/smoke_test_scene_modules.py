"""场景理解三模块端到端冒烟测试（依赖本地 vLLM 服务运行中）。

用法:
    conda run -n VdAnalyst python tools/smoke_test_scene_modules.py \
        --video ../data/laying_corner_1-1.mp4

验证内容:
  1. vlm_client 图像/文本调用连通性
  2. FrameRingBuffer 采样 + SceneNarrator 周期叙述 + JSONL 落盘
  3. AlertAdjudicator 告警异步裁决（描述→裁决两趟 + 事件回写）
  4. SceneIndex BM25/向量检索
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.alert_adjudicator import AlertAdjudicator  # noqa: E402
from src.core.scene_index import SceneIndex  # noqa: E402
from src.core.scene_narrator import SceneNarrator  # noqa: E402
from src.pipeline.frame_buffer import FrameRingBuffer  # noqa: E402
from src.utils.config_loader import load_yaml  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="场景理解模块冒烟测试")
    parser.add_argument("--video", default=str(BASE_DIR.parent / "data" / "laying_corner_1-1.mp4"))
    parser.add_argument("--max-frames", type=int, default=400, help="送入的帧数上限")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "results" / "smoke_scene"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vlm_cfg = load_yaml(str(BASE_DIR / "configs" / "vlm_config.yaml"))

    # ── Step 0: vLLM 连通性 ──────────────────────────────────
    print("\n[Step 0] vLLM 连通性测试")
    from src.core.vlm_client import vlm_chat

    text = vlm_chat(
        base_url=vlm_cfg["vllm_url"],
        api_key=vlm_cfg.get("vllm_api_key", "EMPTY"),
        model=vlm_cfg["model_name"],
        prompt="请仅回复四个字：服务正常",
        max_tokens=16,
        timeout=30,
    )
    print(f"  纯文本调用返回: {text}")
    assert "正常" in text or len(text) > 0

    # ── Step 1: 环形缓冲 + 场景叙述器 ─────────────────────────
    print("\n[Step 1] SceneNarrator 场景叙述测试（每4秒视频时间一次）")
    buffer = FrameRingBuffer(max_seconds=30, samples_per_second=2, target_width=640)
    narrator = SceneNarrator(vlm_cfg, output_dir=str(output_dir))
    narrator.interval_seconds = 4  # 测试用短间隔
    narrator.attach_buffer(buffer)
    narrator.set_source_fps(25.0)

    cap = cv2.VideoCapture(args.video)
    narrated = 0
    frame_id = 0
    while frame_id < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        buffer.push(frame, frame_id)
        narrator.on_frame(frame_id, person_count=1)
        if narrator.status()["narration_count"] > narrated:
            narrated = narrator.status()["narration_count"]
            print(f"  frame {frame_id}: 第{narrated}条叙述已提交")
        time.sleep(0.004)  # 模拟解码节奏，不压满CPU

    # 等待最后一条叙述完成
    for _ in range(120):
        if not narrator.status()["inflight"]:
            break
        time.sleep(0.5)
    print(f"  叙述总数: {narrator.status()['narration_count']}, 错误: {narrator.status()['last_error'] or '无'}")
    log = narrator.get_log(limit=5)
    if log:
        latest = log[-1]
        print(f"  最新叙述 (frame {latest['frame_id']}): {latest['scene_summary'][:100]}")
        print(f"  值得注意活动: {latest['notable_activities']}")
        print(f"  环境: {latest['environment']}")
    else:
        print("  [WARN] 无叙述产出")

    # ── Step 2: 告警裁决器 ───────────────────────────────────
    print("\n[Step 2] AlertAdjudicator 告警裁决测试")
    adjudicator = AlertAdjudicator(vlm_cfg, output_dir=str(output_dir))
    adjudicator.attach_buffer(buffer)
    adjudicator.attach_narrator(narrator)
    adjudicator.set_source_fps(25.0)

    alert = {
        "triggered": True,
        "type": "h03_detected",
        "source": "base_model",
        "final_decision": "pending",
        "snapshot_url": "",
        "clip_path": "",
    }
    alert_frame_id = frame_id
    adjudicator.submit(alert, alert_frame_id)
    # 继续喂数据直到裁决窗口收满（after_seconds=5 → 125帧）
    for _ in range(200):
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        buffer.push(frame, frame_id)
        adjudicator.on_frame(frame_id)
        time.sleep(0.004)
    for _ in range(240):
        if not adjudicator.status()["inflight"]:
            break
        time.sleep(0.5)
    print(f"  裁决状态: {adjudicator.status()}")
    push = adjudicator.consume_pending_push()
    if push:
        print(f"  裁决结果: decision={push.get('decision')} conf={push.get('confidence')}")
        print(f"  事件描述: {str(push.get('description', ''))[:100]}")
        print(f"  理由: {str(push.get('reasoning', ''))[:100]}")
    else:
        print("  [WARN] 无裁决产出")

    narrator.stop()
    adjudicator.stop()

    # ── Step 3: 场景索引检索 ─────────────────────────────────
    print("\n[Step 3] SceneIndex 检索测试")
    index = SceneIndex(vlm_cfg, results_dir=str(output_dir))
    # 构造一条告警事件供索引
    event = {
        "frame_id": alert_frame_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "detections": [{"behavior": {"type": "H03", "confidence": 0.7, "role": "victim"}}],
        "alert": {**alert, "adjudication": push or {}},
    }
    (output_dir / f"event_{alert_frame_id}.json").write_text(
        json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    index.rebuild_if_stale()
    print(f"  索引状态: {index.status()}")
    for query in ["跌倒", "有人躺在地上", "正常行走"]:
        result = index.search(query, top_k=3)
        print(f"  查询[{query}] 模式={result['mode']} 命中{len(result['hits'])}条")
        for hit in result["hits"][:2]:
            print(f"    - [{hit['doc_type']}] {str(hit['highlight'])[:60]} (score={hit['score']})")

    print("\n[SMOKE DONE] 全部步骤执行完毕")


def narated_count(n: int) -> str:
    return str(n)


if __name__ == "__main__":
    main()
