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

export APP_BUILD_COMMIT=${APP_BUILD_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}

MYSQL_HOST=${MYSQL_HOST:-127.0.0.1}
MYSQL_PORT=${MYSQL_PORT:-3306}
MYSQL_USER=${MYSQL_USER:-rehab_user}
MYSQL_PASSWORD=${MYSQL_PASSWORD:-}

echo "===== 1. 启动 MySQL ====="
if MYSQL_PWD="$MYSQL_PASSWORD" mysqladmin -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u"$MYSQL_USER" ping >/dev/null 2>&1; then
  echo "MySQL 已经在运行"
else
  nohup mysqld --defaults-file=/root/autodl-tmp/mysql_conf/my.cnf \
    > /root/autodl-tmp/mysql_logs/start.out 2>&1 &
  echo "MySQL PID: $!"
  sleep 8
fi

MYSQL_PWD="$MYSQL_PASSWORD" mysqladmin -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u"$MYSQL_USER" ping || exit 1

echo "===== 2. 启动后端 8000 ====="
if [ -f "$BASE/backend.pid" ] && kill -0 "$(cat "$BASE/backend.pid")" 2>/dev/null; then
  kill "$(cat "$BASE/backend.pid")" || exit 1
  sleep 3
else
  pkill -f "python -m uvicorn main:app --host 127.0.0.1 --port 8000" 2>/dev/null || true
fi
cd "$APP/backend" || exit 1
source /root/autodl-tmp/envs/rehab_backend/bin/activate
nohup python -m uvicorn main:app --host 127.0.0.1 --port 8000 \
  > /root/autodl-tmp/rehab_project/backend_run.log 2>&1 &
echo $! > "$BASE/backend.pid"
echo "后端 PID: $!"
ready=0
for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8000/api/ready >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [ "$ready" != "1" ]; then
  tail -80 /root/autodl-tmp/rehab_project/backend_run.log || true
  exit 1
fi
curl -fsS http://127.0.0.1:8000/api/ready || exit 1
echo ""

echo "===== 3. 检查前端生产包 ====="
while IFS= read -r pid; do
  [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
done < <(pgrep -f "$APP/frontend/node_modules/.bin/vite" 2>/dev/null || true)
sleep 1
if command -v ss >/dev/null 2>&1 && ss -ltn | grep -Eq '[:.]5173[[:space:]]'; then
  echo "检测到 5173 仍在监听。请确认并停止遗留的前端开发服务器后重试。"
  exit 1
fi
cd "$APP/frontend" || exit 1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/node_env
npm run build || exit 1
test -f "$APP/frontend/dist/index.html" || exit 1
echo ""

echo "===== 4. 启动公开演示入口 Nginx 6006 ====="
if [ -f /etc/nginx/conf.d/rehab_demo.conf ]; then
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
echo "公网演示入口：${AutoDLService6006URL:-请在云平台查看 6006 端口映射地址}"
