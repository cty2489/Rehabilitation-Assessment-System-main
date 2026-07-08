#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/rehab_project
APP=$BASE/Rehabilitation-Assessment-System-main
LOG_FILE=$BASE/gguf_server.log
PORT=${LLM_GGUF_SERVER_PORT:-6008}

cd "$APP"

echo "===== 手动启动 GGUF 回退/对照服务 $PORT ====="

if curl -s "http://127.0.0.1:$PORT/health" | grep -q '"loaded":true'; then
  echo "GGUF 服务已经在运行：http://127.0.0.1:$PORT"
  exit 0
fi

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
else
  pkill -f "llm_gguf_server:app" 2>/dev/null || true
fi

# shellcheck disable=SC1091
source /root/autodl-tmp/envs/llm_env/bin/activate

export LLM_GGUF_MODEL_PATH=${LLM_GGUF_MODEL_PATH:-/root/autodl-tmp/rehab_project/models/qwen2.5-7b-instruct-gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf}
export LLM_GGUF_N_CTX=${LLM_GGUF_N_CTX:-8192}
export LLM_GGUF_N_GPU_LAYERS=${LLM_GGUF_N_GPU_LAYERS:--1}
export LLM_GGUF_N_THREADS=${LLM_GGUF_N_THREADS:-8}
export LLM_GGUF_N_BATCH=${LLM_GGUF_N_BATCH:-512}
export LLM_GGUF_N_UBATCH=${LLM_GGUF_N_UBATCH:-256}
export LLM_GGUF_MAX_TOKENS=${LLM_GGUF_MAX_TOKENS:-2048}
export LLM_GGUF_TEMPERATURE=${LLM_GGUF_TEMPERATURE:-0.1}
export LLM_GGUF_TOP_P=${LLM_GGUF_TOP_P:-0.9}
export LLM_GGUF_SERVER_HOST=127.0.0.1
export LLM_GGUF_SERVER_PORT=$PORT
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

nohup python -m uvicorn llm_gguf_server:app --host 127.0.0.1 --port "$PORT" \
  > "$LOG_FILE" 2>&1 &
echo "GGUF PID: $!"

for _ in $(seq 1 60); do
  if curl -s "http://127.0.0.1:$PORT/health" | grep -q '"loaded":true'; then
    echo "GGUF 服务已就绪：http://127.0.0.1:$PORT"
    exit 0
  fi
  sleep 2
done

echo "GGUF 服务未在预期时间内就绪，请查看日志：$LOG_FILE" >&2
exit 1
