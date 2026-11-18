#!/usr/bin/env bash
# T2.4 前置：ONNX → OM 批量转换（detector/pose/crop）+ 逐个 bench（盒上执行）
# 用法: bash tools/export_om_atlas.sh [soc_version]
#   环境变量: PYBIN、SKIP_BENCH=1
set -uo pipefail
cd "$(dirname "$0")/.."

SOC="${1:-${SOC_VERSION:-Ascend310B4}}"
PYBIN="${PYBIN:-python3}"
OUT=results/edge_m0/export_om.json
mkdir -p results/edge_m0 models/crop

command -v atc >/dev/null 2>&1 || { echo "[export_om] atc 不可用（非盒环境）"; printf '{"status":"SKIP"}\n' > "$OUT"; exit 0; }

convert(){  # src onnx → 同目录 .om（失败仅记录，继续其余）
  # 第 3 参可选：输出基名（缺省=源文件名去 .onnx），用于 FP32 源产出 _fp16_bs1 命名的 OM
  local SRC="$1" SHAPE="$2" OUTBASE="${3:-${SRC%.onnx}}"
  local OM="${OUTBASE}.om"
  [ -f "$OM" ] && { echo "[export_om] 已存在跳过: $OM"; return 0; }
  echo "[export_om] atc: $SRC"
  # D1: atc 自动补 .om 后缀 → --output 传去后缀路径（否则产出 .om.om）；新版无 --log_file，用重定向
  atc --model="$SRC" --framework=5 --output="$OUTBASE" --input_shape="$SHAPE" \
      --soc_version="$SOC" >results/edge_m0/atc_$(basename "$OUTBASE").log 2>&1
  [ -f "$OM" ] && echo "[export_om] OK: $OM" || { echo "[export_om] FAIL: $SRC（见 log）"; return 1; }
}

FAILS=0
# D7 配套：优先转 int8 检测模型（ONNX 回退同为 int8，CPU 兜底也快一倍）
convert models/detector/yolov8n_int8.onnx "images:1,3,640,640" || FAILS=$((FAILS+1))
convert models/detector/yolov8n.onnx        "images:1,3,640,640" || FAILS=$((FAILS+1))
# 姿态：edge 档加载的是 FP32 源转出的 FP16 batch=1 OM（vitpose_wholebody_s_fp16_bs1.om）。
#   int8 QLinear 版在 CANN 6.2 ATC 无 parser（E19010 QuantizeLinear/QLinearConv），转不动，故不再尝试；
#   逐 track batch=1 由 PoseEstimator._is_om 路径适配。FP32 ONNX 保留作 CPU 回退。
convert models/pose/vitpose_wholebody_s.onnx "input:1,3,256,192" \
        models/pose/vitpose_wholebody_s_fp16_bs1 || FAILS=$((FAILS+1))
convert models/crop/crop_v1.onnx            "input:1,3,224,224"    || FAILS=$((FAILS+1))

if [ "${SKIP_BENCH:-0}" != "1" ]; then
  # 会商项A：先跑约定探针（子进程隔离，segfault 安全），再 bench 对拍
  [ -f models/detector/yolov8n_int8.om ] && "$PYBIN" edge_migration/scripts/g3_pyacl_probe.py \
      --om models/detector/yolov8n_int8.om || true
  [ -f models/detector/yolov8n_int8.om ] && "$PYBIN" edge_migration/scripts/g3_om_infer.py \
      --om models/detector/yolov8n_int8.om --compare-onnx models/detector/yolov8n_int8.onnx \
      --runs 100 --out results/edge_m0/g3_om_bench.json || true
  [ -f models/crop/crop_v1.om ] && "$PYBIN" edge_migration/scripts/g3_om_infer.py \
      --om models/crop/crop_v1.om --runs 100 --shape 1,3,224,224 \
      --out results/edge_m0/g3_om_bench_crop.json || true
fi

printf '{"status":"DONE","fails":%s,"soc":"%s","ts":%s}\n' "$FAILS" "$SOC" "$(date +%s)" > "$OUT"
echo "[export_om] 完成（失败 $FAILS）-> $OUT"
exit 0
