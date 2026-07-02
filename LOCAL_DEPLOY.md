# 本地部署说明

本文档用于在本机部署智能康复评估系统。当前推荐模式是：

- 后端 FastAPI 本地运行，端口 `8000`
- 前端 Vite/React 本地运行，端口 `5173`
- MySQL 本地运行，用于患者档案、评估记录、运动 trial、biomarker 明细存储
- 康复评分模型本地 CPU 运行
- AI 康复报告使用 DeepSeek API 生成，不在本机加载大语言模型

## 1. 本地部署结构

```text
浏览器 http://localhost:5173
        |
        v
前端 React/Vite 5173
        |
        v
后端 FastAPI/Uvicorn 8000
        |
        +-- MySQL: patients / assessments / assessment_trials / assessment_biomarkers
        |
        +-- 本地深度学习评分模型: FMA-UE / BI / hand_tone / hand_function
        |
        +-- DeepSeek API: 生成 AI 康复评估报告
```

## 2. 必要环境

本机需要具备：

- Python 虚拟环境：`.venv\Scripts\python.exe`
- 后端依赖：`backend\requirements.txt`
- Node.js LTS 和 npm
- 前端依赖：`frontend\node_modules`
- MySQL 服务
- 后端配置文件：`backend\.env`

当前项目的 `.env` 不要提交到 git，因为里面包含 DeepSeek API key 和 MySQL 密码。

## 3. 配置 backend\.env

推荐关键配置如下：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_MAX_TOKENS=4096
DEEPSEEK_TEMPERATURE=0

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_DB=rehab_mysql
```

`DEEPSEEK_API_KEY` 和 `MYSQL_PASSWORD` 也需要在 `backend\.env` 中配置，但不要写入说明文档或公开仓库。

## 4. 首次安装依赖

如果依赖已经安装过，可以跳过。

后端：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
```

前端：

```powershell
cd .\frontend
npm install
cd ..
```

如果 PowerShell 提示找不到 `npm`，但 Node.js 已经通过 WinGet 安装，可以临时使用完整路径，例如：

```powershell
$nodeDir = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\OpenJS.NodeJS.LTS_Microsoft.Winget.Source_8wekyb3d8bbwe\node-v24.18.0-win-x64"
$env:PATH = "$nodeDir;$env:PATH"
& "$nodeDir\npm.cmd" install
```

`scripts\start-local.ps1` 会自动搜索常见位置下的 `npm.cmd`，一般不需要手动处理。

## 5. 一键启动

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-local.ps1
```

启动后访问：

```text
http://localhost:5173
```

后端接口地址：

```text
http://localhost:8000
```

日志位置：

```text
.cache\local-deploy\backend.out.log
.cache\local-deploy\backend.err.log
.cache\local-deploy\frontend.out.log
.cache\local-deploy\frontend.err.log
```

## 6. 检查本地服务

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-local.ps1
```

这个脚本会检查：

- `.venv` 是否存在
- `frontend\node_modules` 是否存在
- `backend\.env` 是否存在
- MySQL 是否运行
- 端口 `8000` / `5173` 是否监听
- 后端 `/api/health` 是否可访问
- 后端 `/api/stats/summary` 是否可访问
- 前端页面是否可访问

## 7. 停止本地服务

停止由 `start-local.ps1` 启动的进程：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-local.ps1
```

如果 8000 或 5173 是之前手动启动的旧进程，可以显式停止端口占用进程：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-local.ps1 -StopPortProcesses
```

## 8. 手动启动方式

不用脚本时，可以分别打开两个 PowerShell 窗口。

窗口一：后端

```powershell
cd "C:\Users\22097\Desktop\康复大模型项目\Rehabilitation-Assessment-System-main (1)\Rehabilitation-Assessment-System-main\backend"
..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

窗口二：前端

```powershell
cd "C:\Users\22097\Desktop\康复大模型项目\Rehabilitation-Assessment-System-main (1)\Rehabilitation-Assessment-System-main\frontend"
npm run dev -- --host 0.0.0.0 --port 5173
```

## 9. 常见问题

### MySQL 连接失败

先确认 MySQL 服务是否运行：

```powershell
Get-Service -Name MySQL*
```

再确认 `backend\.env` 中的 `MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_DB` 是否正确。

### AI 报告生成失败

先确认：

- `LLM_PROVIDER=deepseek`
- `DEEPSEEK_API_KEY` 已配置
- `DEEPSEEK_MODEL=deepseek-v4-flash`
- 当前网络可以访问 `https://api.deepseek.com`

### 前端能打开，但接口报错

前端通过 Vite proxy 把 `/api` 转发到 `http://localhost:8000`，所以要确认后端 8000 正常运行。

### 不建议本机直接加载大语言模型

本机部署推荐使用 DeepSeek API。3B/7B 本地模型会占用显存和内存，报告生成上下文较长时还需要精简 prompt，稳定性不如 API 模式。
