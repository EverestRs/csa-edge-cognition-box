#!/usr/bin/env bash
# G1 探针：确认 Atlas 内存版本（4GB/8GB）→ 决定 VLM 档位建议（EDGE_MIGRATION §3.4 G1）
# 用法: bash edge_migration/scripts/g1_mem.sh [输出json路径]
# 退出码: 0=成功（含 4GB/8GB 判定），非0=读取失败
set -euo pipefail

OUT="${1:-results/edge_m0/g1_mem.json}"
mkdir -p "$(dirname "$OUT")"

KB=$(awk '/MemTotal/{print $2; exit}' /proc/meminfo)
if [ -z "$KB" ]; then echo "MemTotal 读取失败" >&2; exit 1; fi
GB=$(awk "BEGIN{printf \"%.1f\", $KB/1048576}")

if awk "BEGIN{exit !($GB < 5.5)}"; then
    VARIANT="4GB"; STRETCH="false"
    TIER="文本LLM常驻需实测G1净余（可能降0.5B）；stretch VL 取消；zram 建议开启"
elif awk "BEGIN{exit !($GB < 10)}"; then
    VARIANT="8GB"; STRETCH="true"
    TIER="Qwen2.5-1.5B 文本常驻（EDGE v3.0 §5 预算）；stretch VL-2B 交替档可评估"
else
    VARIANT=">=10GB(疑似非Atlas/双盒环境)"; STRETCH="true"
    TIER="按实测内存规划；本方案默认仍按单盒走"
fi

printf '{\n  "probe": "g1_mem",\n  "ts": %s,\n  "mem_total_gb": %s,\n  "variant": "%s",\n  "vlm_stretch_eligible": %s,\n  "tier_advice": "%s"\n}\n' \
    "$(date +%s)" "$GB" "$VARIANT" "$STRETCH" "$TIER" > "$OUT"
echo "[g1] MemTotal=${GB}GB variant=${VARIANT} -> $OUT"
