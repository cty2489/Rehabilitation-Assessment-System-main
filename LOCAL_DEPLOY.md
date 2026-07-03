# 本地开发部署说明

本文档用于在 Windows 本机开发和调试。生产演示请优先看 `SERVER_DEPLOY.md`。

## 1. 本地结构

```text
浏览器 http://localhost:5173
  -> Vite/React 5173
  -> FastAPI 8000
  -> MySQL 3306
  -> DeepSeek API 或本机 GGUF LLM 6006
```

本地开发使用 Vite dev server；云服务器生产部署使用 Nginx + `frontend/dist`，不启动 `5173`。

## 2. 必要环境

- Python 3.10/3.11
- Node.js LTS 和 npm
- MySQL 5.7+ 或 MySQL 8+
- 可选 NVIDIA GPU，用于本机运行 PyTorch 和 GGUF LLM

## 3. 安装依赖

后端：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
```

前端：

```powershell
cd .\frontend
npm install
cd ..
```

如需本机 GGUF 报告服务：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-gguf-server.txt
```

## 4. 配置 backend\.env

复制模板：

```powershell
Copy-Item .\backend\.env.example .\backend\.env
```

至少配置：

```env
APP_ADMIN_USER=admin
APP_ADMIN_PASSWORD=change-this-password
APP_AUTH_TOKEN=generate-a-long-random-token

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your-mysql-password
MYSQL_DB=rehab_mysql
```

DeepSeek API 模式：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

本机 GGUF 模式：

```env
LLM_PROVIDER=remote
LLM_REMOTE_URL=http://127.0.0.1:6006
LLM_REMOTE_TIMEOUT=300
```

不要提交真实 `backend\.env`。

## 5. 启动

如果使用本机 GGUF，先启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-gguf-llm.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\check-gguf-llm.ps1 -Generate
```

启动后端和前端：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-local.ps1
```

打开：

```text
http://localhost:5173
```

页面登录账号来自 `APP_ADMIN_USER` / `APP_ADMIN_PASSWORD`。

## 6. 检查

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-local.ps1
```

检查脚本会验证：

- `.venv`
- `frontend\node_modules`
- `backend\.env`
- MySQL 服务
- `8000` / `5173` 端口
- `/api/health`
- `/api/auth/login`
- 登录后 `/api/stats/summary`
- 前端页面

结果导出文件默认写入项目根目录下的 `exports/`。如需改位置，在 `backend\.env` 中设置：

```env
EXPORT_ROOT=D:\rehab_exports
```

## 7. 停止

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-local.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\stop-gguf-llm.ps1
```

如端口被旧进程占用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-local.ps1 -StopPortProcesses
powershell -ExecutionPolicy Bypass -File .\scripts\stop-gguf-llm.ps1 -StopPortProcess
```

## 8. 手动启动

后端：

```powershell
cd .\backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

前端：

```powershell
cd .\frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

## 9. 常见问题

### 登录后接口仍然 401

清理浏览器 localStorage 后重新登录，或确认 `APP_AUTH_TOKEN` 已配置且后端已重启。

### 统计接口检查失败

新版本统计接口需要 Bearer token。请用更新后的 `scripts/check-local.ps1`，不要直接裸访问 `/api/stats/summary` 判断服务是否正常。

### 前端能打开但接口报错

Vite 会把 `/api` 代理到 `http://localhost:8000`。先确认后端 `8000` 正常，再确认页面已经登录。

### GGUF 模型不能用 transformers 加载

当前 GGUF 是 llama.cpp 格式，不是 safetensors/FP16。请用 `llm_gguf_server.py` 或 `scripts/start-gguf-llm.ps1` 启动。
