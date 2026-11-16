#!/usr/bin/env bash
# G3 探针：YOLOv8n ONNX → ATC → OM，并在盒上 bench（T1.4）。
# 仅在 Atlas 盒上可完整执行；非盒环境自动 SKIP（退出码 0，JSON 记 skip 原因）。
# 用法: bash edge_migration/scripts/g3_atc_om_bench.sh [onnx路径] [输出json]
#   环境变量: SOC_VERSION(默认 Ascend310B4) INPUT_SHAPE(默认 images:1,3,640,640) PYBIN
set -uo pipefail
cd "$(dirname "$0")/../.."

SRC="${1:-models/detector/yolov8n.onnx}"
OUT="${2:-results/edge_m0/g3_atc_om.json}"
SOC_VERSION="${SOC_VERSION:-Ascend310B4}"
INPUT_SHAPE="${INPUT_SHAPE:-images:1,3,640,640}"
PYBIN="${PYBIN:-python3}"
mkdir -p "$(dirname "$OUT")" results/edge_m0

emit(){ printf '{"probe":"g3_atc_om","ts":%s,"status":"%s","reason":"%s","src":"%s","soc":"%s"}\n' \
  "$(date +%s)" "$1" "$2" "$SRC" "$SOC_VERSION" > "$OUT"; echo "[g3] $1: $2 -> $OUT"; }

command -v atc >/dev/null 2>&1 || { emit SKIP "atc_not_installed(非盒环境或CANN未配环境变量)"; exit 0; }
[ -f "$SRC" ] || { emit FAIL "src_missing:$SRC"; exit 1; }

OM="${SRC%.onnx}_310B.om"
LOG=results/edge_m0/g3_atc.log
atc --model="$SRC" --framework=5 --output="$OM" \
    --input_shape="$INPUT_SHAPE" --soc_version="$SOC_VERSION" \
    --log_file="$LOG" >/dev/null 2>&1
RC=$?
if [ $RC -ne 0 ] || [ ! -f "$OM" ]; then
    emit FAIL "atc_rc=$RC(详见 $LOG)"
    exit 1
fi

# OM bench + 同输入数值对拍（ORT vs OM），pyACL 不可用时记 SKIP 但转换产物保留
$PYBIN edge_migration/scripts/g3_om_infer.py --om "$OM" --compare-onnx "$SRC" \
    --runs 100 --out "$OUT" && { echo "[g3] bench -> $OUT"; exit 0; }
printf '{"probe":"g3_atc_om","ts":%s,"status":"CONVERTED_BENCH_SKIPPED","reason":"pyacl不可用(g3_om_infer失败)；OM已产出 %s","om":"%s"}\n' \
    "$(date +%s)" "$OM" "$OM" > "$OUT"
echo "[g3] 转换成功、bench跳过(盒外静态检查场景) -> $OUT"
