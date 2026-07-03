#!/usr/bin/env bash

BASE=/root/autodl-tmp/rehab_project
APP=$BASE/Rehabilitation-Assessment-System-main
ENV_FILE=$APP/backend/.env

cd "$APP" || exit 1

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

MYSQL_HOST=${MYSQL_HOST:-127.0.0.1}
MYSQL_PORT=${MYSQL_PORT:-3306}
MYSQL_USER=${MYSQL_USER:-rehab_user}
MYSQL_PASSWORD=${MYSQL_PASSWORD:-}

echo "===== 1. 启动 MySQL ====="
if MYSQL_PWD="$MYSQL_PASSWORD" mysqladmin -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u"$MYSQL_USER" ping >/dev/null 2>&1; then
  echo "MySQL 已经在运行"
else
  pkill mysqld 2>/dev/null || true
  nohup mysqld --defaults-file=/root/autodl-tmp/mysql_conf/my.cnf \
    > /root/autodl-tmp/mysql_logs/start.out 2>&1 &
  echo "MySQL PID: $!"
  sleep 8
fi

MYSQL_PWD="$MYSQL_PASSWORD" mysqladmin -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u"$MYSQL_USER" ping || exit 1

echo "===== 2. 启动 GGUF 大模型服务 6007 ====="
if curl -s http://127.0.0.1:6007/health | grep -q '"loaded":true'; then
  echo "GGUF 大模型服务已经在运行"
else
  pkill -f "llm_gguf_server:app" 2>/dev/null || true
  cd "$APP" || exit 1
  source /root/autodl-tmp/envs/llm_env/bin/activate

  export LLM_GGUF_MODEL_PATH=/root/autodl-tmp/rehab_project/models/qwen2.5-7b-instruct-gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
  export LLM_GGUF_N_CTX=8192
  export LLM_GGUF_N_GPU_LAYERS=-1
  export LLM_GGUF_N_THREADS=8
  export LLM_GGUF_N_BATCH=512
  export LLM_GGUF_N_UBATCH=256
  export LLM_GGUF_MAX_TOKENS=2048
  export LLM_GGUF_TEMPERATURE=0.1
  export LLM_GGUF_TOP_P=0.9
  export LLM_GGUF_SERVER_HOST=127.0.0.1
  export LLM_GGUF_SERVER_PORT=6007
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

  nohup python -m uvicorn llm_gguf_server:app --host 127.0.0.1 --port 6007 \
    > /root/autodl-tmp/rehab_project/gguf_server.log 2>&1 &
  echo "GGUF PID: $!"
  sleep 40
fi

curl http://127.0.0.1:6007/health || exit 1
echo ""

echo "===== 3. 启动后端 8000 ====="
if curl -s http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "后端已经在运行"
else
  pkill -f "uvicorn main:app" 2>/dev/null || true
  cd "$APP/backend" || exit 1
  source /root/autodl-tmp/envs/rehab_backend/bin/activate
  nohup python -m uvicorn main:app --host 127.0.0.1 --port 8000 \
    > /root/autodl-tmp/rehab_project/backend_run.log 2>&1 &
  echo "后端 PID: $!"
  sleep 20
fi

curl http://127.0.0.1:8000/api/health || exit 1
echo ""

echo "===== 4. 检查前端生产包 ====="
pkill -f "vite" 2>/dev/null || true
if [ ! -f "$APP/frontend/dist/index.html" ]; then
  cd "$APP/frontend" || exit 1
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate /root/autodl-tmp/envs/node_env
  npm run build || exit 1
fi
test -f "$APP/frontend/dist/index.html" || exit 1
echo ""

echo "===== 5. 启动公开演示入口 Nginx 6006 ====="
if [ -f /etc/nginx/conf.d/rehab_demo.conf ]; then
  if [ ! -f "$APP/frontend/dist/index.html" ]; then
    cd "$APP/frontend" || exit 1
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate /root/autodl-tmp/envs/node_env
    npm run build || exit 1
  fi

  nginx -t || exit 1
  nginx -s reload 2>/dev/null || nginx || exit 1
  sleep 2

  status=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:6006/)
  if [ "$status" = "200" ]; then
    echo "公开演示入口已经在运行，页面内登录保护已启用"
  else
    curl -I http://127.0.0.1:6006 || true
    exit 1
  fi
else
  echo "未找到 /etc/nginx/conf.d/rehab_demo.conf，跳过公开演示入口"
fi
echo ""

echo "===== 全部启动完成 ====="
echo "云服务器内部前端：Nginx 6006 -> frontend/dist"
echo "云服务器内部后端：http://127.0.0.1:8000/docs"
echo "公网演示入口：${AutoDLService6006URL:-https://u1072937-x4xd-c1b60d69.bjb2.seetacloud.com:8443}"
