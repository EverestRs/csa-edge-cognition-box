#!/usr/bin/env bash
# 拉取超出 git 体积限制的大权重（GitHub Release）与第三方公开权重（ModelScope）。
# 用法：bash scripts/download_models.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# GitHub Release 归属仓库；同名 Tag=v1.0.0-models 下已上传 4 个权重文件
OWNER="EverestRs/csa-edge-cognition-box"
TAG="v1.0.0-models"
BASE="https://github.com/${OWNER}/releases/download/${TAG}"

mkdir -p models/detector models/pose models/action_v2 models/embedding

echo "[1/4] 自研/导出中间权重（GitHub Release）"
curl -fL --retry 3 -o models/pose/vitpose_wholebody_s.onnx        "${BASE}/vitpose_wholebody_s.onnx"
curl -fL --retry 3 -o models/pose/vitpose_wholebody_s_fp16_bs1.om "${BASE}/vitpose_wholebody_s_fp16_bs1.om"
curl -fL --retry 3 -o models/action_v2/pose_classifier.onnx        "${BASE}/pose_classifier.onnx"
curl -fL --retry 3 -o models/detector/yolov8n.onnx                 "${BASE}/yolov8n.onnx"

echo "[2/4] 场景检索嵌入 bge-small-zh-v1.5（ModelScope，MIT）"
pip install -q modelscope
modelscope download --model BAAI/bge-small-zh-v1.5 --local_dir models/embedding/bge-small-zh-v1.5

echo "[3/4] 语言层 Qwen2.5-0.5B GGUF（ModelScope，Apache-2.0）→ ~/models/llm/"
mkdir -p ~/models/llm
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  --include '*q4_0*' --local_dir ~/models/llm

echo "[4/4] 校验 OM 到位"
# detector/crop 两个小 OM 随 git 提交；pose FP16 bs1 OM 由本脚本从 GitHub Release 拉取
ls -l models/detector/yolov8n.om models/crop/crop_v1.om models/pose/vitpose_wholebody_s_fp16_bs1.om

echo "完成。OM 缺失或需重编译时：tools/export_om_atlas.sh（需 CANN 6.2.T2）"
