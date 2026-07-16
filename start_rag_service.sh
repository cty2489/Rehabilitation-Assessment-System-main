#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/rehab_project
APP=${REHAB_APP_DIR:-$BASE/current}
ENV_FILE=${RAG_ENV_FILE:-$BASE/rag.env}
PID_FILE=$BASE/rag_service.pid
LOG_FILE=$BASE/rag_service.log

test -d "$APP/rag"
test -f "$ENV_FILE"
test -x /root/autodl-tmp/envs/rag_env/bin/python

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

HOST=${RAG_SERVICE_HOST:-127.0.0.1}
PORT=${RAG_SERVICE_PORT:-8010}
if [ "$HOST" != "127.0.0.1" ]; then
  echo "RAG service must bind to 127.0.0.1"
  exit 1
fi
if [ "${RAG_ENABLED:-0}" != "1" ]; then
  echo "RAG_ENABLED must be 1 in $ENV_FILE"
  exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  kill "$(cat "$PID_FILE")"
  sleep 2
fi

cd "$APP"
nohup /root/autodl-tmp/envs/rag_env/bin/python -m uvicorn rag.service:app \
  --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 &
SERVICE_PID=$!
echo "$SERVICE_PID" > "$PID_FILE"

for _ in $(seq 1 120); do
  if curl --max-time 2 -fsS "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then
    echo "RAG_SERVICE_READY:$SERVICE_PID"
    exit 0
  fi
  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    echo "RAG_SERVICE_EXITED"
    tail -120 "$LOG_FILE" || true
    exit 1
  fi
  sleep 2
done

echo "RAG_SERVICE_READY_TIMEOUT"
tail -120 "$LOG_FILE" || true
kill "$SERVICE_PID" 2>/dev/null || true
rm -f "$PID_FILE"
exit 1
