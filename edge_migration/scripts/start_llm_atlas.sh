#!/usr/bin/env bash
# 盒上文本 LLM 服务启动（llama.cpp llama-server，OpenAI 兼容 :8001，01 §2.5a）
# 仿 scripts/start_vllm.sh：防重复启动 + 日志落盘 + 健康检查约定
# 用法:
#   LLM_MODEL_PATH=models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf bash edge_migration/scripts/start_llm_atlas.sh
# 停止: kill $(cat results/llm_serve.pid)
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${LLM_MODEL_PATH:-models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
PORT="${LLM_PORT:-8001}"
CTX="${LLM_CTX_SIZE:-2048}"          # 小 KV 预算（EDGE §5）
THREADS="${LLM_THREADS:-3}"          # 给感知编排(A55)留核：不占满 4 核
MODEL_NAME="${LLM_SERVED_NAME:-qwen2.5-1.5b-instruct}"
LOG="${LLM_LOG:-results/llm_serve.log}"
PIDF="results/llm_serve.pid"
mkdir -p "$(dirname "$LOG")"

if command -v llama-server >/dev/null 2>&1; then LLAMA=llama-server; else
    LLAMA="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
    [ -x "$LLAMA" ] || { echo "[start_llm] llama-server 不存在：先按 ENV_LOCK 编译或设 LLAMA_BIN" >&2; exit 1; }
fi

# 防重复：端口已有服务则复用（避免双实例抢内存——同 start_vllm.sh 血泪教训）
if curl -s --connect-timeout 2 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[start_llm] 端口 ${PORT} 已有 OpenAI 兼容服务，直接复用"
    curl -s "http://localhost:${PORT}/v1/models" | head -c 300; echo
    exit 0
fi
[ -f "$MODEL" ] || { echo "[start_llm] 模型不存在: $MODEL（GGUF 需预先下载，禁云端 API 指推理期而非获取期）" >&2; exit 1; }

MODEL_ARGS=()
# D6: 新版 llama.cpp 移除 --model-name（served 名取文件/别名）——探测后决定是否传
if "$LLAMA" --help 2>&1 | grep -q -- '--model-name'; then
    MODEL_ARGS+=(--model-name "$MODEL_NAME")
fi

nohup "$LLAMA" -m "$MODEL" "${MODEL_ARGS[@]}" \
    --host 127.0.0.1 --port "$PORT" --ctx-size "$CTX" --threads "$THREADS" \
    --temp 0.1 --top-p 0.9 >> "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "[start_llm] pid=$(cat "$PIDF") log=$LOG"
for i in $(seq 1 30); do
    sleep 2
    if curl -s --connect-timeout 1 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[start_llm] 健康检查通过：http://127.0.0.1:${PORT}/v1/chat/completions"
        echo "  vlm_config.yaml 设: vllm_url: http://127.0.0.1:${PORT}/v1/chat/completions, model_name: ${MODEL_NAME}"
        exit 0
    fi
done
echo "[start_llm] 60s 内未就绪，查看 $LOG" >&2
exit 1
