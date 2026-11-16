#!/usr/bin/env bash
# 启动本地 Qwen2.5-VL-7B vLLM 服务（OpenAI 兼容 API）
#
# 用途:
#   - VLM 灰区复核 (src/core/vlm_reviewer.py, vlm_config.yaml: vllm_url)
#   - 场景叙述器 SceneNarrator (src/core/scene_narrator.py)
#   - 离线伪标注 (tools/vlm_label_pipeline.py)
#   - 告警事件裁决
#
# 用法:
#   bash scripts/start_vllm.sh            # 前台启动
#   nohup bash scripts/start_vllm.sh &    # 后台启动, 日志: results/vllm_serve.log
#
# 停止: pkill -f "vllm serve"  或 kill $(cat results/vllm_serve.pid)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${VLLM_MODEL_PATH:-$PROJECT_ROOT/../Qwen2.5-VL-7B-Instruct}"
PORT="${VLLM_PORT:-8001}"
SERVED_NAME="${VLLM_SERVED_NAME:-qwen2.5-vl-7b-instruct}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.60}"  # 需与 ORT CUDA 推理共存于同一块 GPU，预留显存
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
LOG_FILE="${VLLM_LOG:-$PROJECT_ROOT/results/vllm_serve.log}"

# 低显存模式（显存优化）：VLLM_QUANT=bnb 切换 bitsandbytes NF4 量化——
# 语言层权重 15.2GB→约4.5GB，配 VLLM_GPU_MEM_UTIL=0.30 时 vLLM 实测总占用 9.3GB（fp16 默认 17.6GB，-47%）。
# 注意：V100=SM70，vLLM 的 AWQ/AWQ-marlin/fp8/marlin 均要求 SM75+ 不可用，Qwen 官方也未发布
# 2.5-VL 的 GPTQ 版；bnb（min_SM=70）是本卡唯一 int4 路径。能力验收（语义精判/命名/多图判定
# 三项与 fp16 基线一致）与已知退化（多图输出数组化，解析器已容错+prompt 加单对象约束）
# 记录见 UNSEEN_BEHAVIOR_ROADMAP.md §8。
QUANT_ARGS=()
if [ "${VLLM_QUANT:-}" = "bnb" ]; then
    MODEL_PATH="${VLLM_BNB_PATH:-$PROJECT_ROOT/../Qwen2.5-VL-7B-Instruct}"
    QUANT_ARGS=(--quantization bitsandbytes --load-format bitsandbytes)
    
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "[start_vllm] 模型目录不存在: $MODEL_PATH" >&2
    exit 1
fi

# 防重复启动：双实例会瓜分显存，导致多图请求图像编码失败（退化为模板回显）乃至 OOM
if curl -s --connect-timeout 2 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[start_vllm] 端口 ${PORT} 已有 vLLM 服务在运行，如需重启请先: pkill -9 -f vllm.entrypoints" >&2
    exit 0
fi
if pgrep -f "vllm.entrypoints" >/dev/null 2>&1; then
    echo "[start_vllm] 检测到残留 vLLM 进程（未监听端口），先终止:" >&2
    pgrep -fa "vllm.entrypoints" >&2
    pkill -9 -f "vllm.entrypoints" || true
    sleep 3
fi

mkdir -p "$(dirname "$LOG_FILE")"

echo "[start_vllm] model=$MODEL_PATH port=$PORT gpu_mem_util=$GPU_MEM_UTIL max_model_len=$MAX_MODEL_LEN"
echo "[start_vllm] log=$LOG_FILE"

# conda 环境
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "VdAnalyst" ]; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate VdAnalyst 2>/dev/null || true
fi

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --dtype half \
    --limit-mm-per-prompt image=8 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --disable-log-requests \
    "${QUANT_ARGS[@]}" \
    "$@"
