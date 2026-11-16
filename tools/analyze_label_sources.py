"""归因分析：M01 标签的来源构成（配合 eval_dump 的 sources 字段）。

用法: python tools/analyze_label_sources.py <dump.jsonl> [fps]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def main() -> None:
    path = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0

    m01_src = Counter()          # M01 帧级来源
    all_src = Counter()          # 全部帧级来源（对照）
    m01_by_minute: dict[int, Counter] = defaultdict(Counter)
    m01_tracks: dict[str, Counter] = defaultdict(Counter)

    for line in open(path, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        fid = int(rec.get("frame_id", 0))
        minute = int(fid / fps // 60)
        sources = rec.get("sources", {}) or {}
        for tid, label in rec.get("labels", []):
            if not label:
                continue
            src = sources.get(tid, "unknown")
            all_src[src] += 1
            if label == "M01":
                m01_src[src] += 1
                m01_by_minute[minute][src] += 1
                m01_tracks[tid][src] += 1

    total_m01 = sum(m01_src.values())
    total_all = sum(all_src.values())
    print(f"=== M01 帧级来源构成（共 {total_m01} 帧·人次）===")
    for src, n in m01_src.most_common():
        print(f"  {src:>12}: {n:6d} ({100*n/max(1,total_m01):5.1f}%)")
    print(f"\n=== 全部标签来源构成（共 {total_all}，对照）===")
    for src, n in all_src.most_common():
        print(f"  {src:>12}: {n:6d} ({100*n/max(1,total_all):5.1f}%)")

    print("\n=== M01 按分钟 × 来源 ===")
    for m in sorted(m01_by_minute):
        parts = " ".join(f"{s}:{n}" for s, n in m01_by_minute[m].most_common())
        print(f"  {m:3d}min: {parts}")

    print(f"\n=== M01 track 数: {len(m01_tracks)}（前 10）===")
    for tid, cnt in sorted(m01_tracks.items(), key=lambda x: -sum(x[1].values()))[:10]:
        parts = " ".join(f"{s}:{n}" for s, n in cnt.most_common())
        print(f"  {tid}: {parts}")


if __name__ == "__main__":
    main()
