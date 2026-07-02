# 智能康复评估系统启动流程

本文档记录当前版本 `local-gguf-llm-v1` 的本地启动流程。当前系统使用本地 Qwen2.5-7B-Instruct GGUF 量化模型生成康复评估报告。

## 1. 当前系统组成

```text
浏览器前端: http://localhost:5173
后端服务:   http://localhost:8000
本地大模型: http://localhost:6006
MySQL:      127.0.0.1:3306
```

调用链如下：

```text
浏览器
  -> 前端 React/Vite 5173
  -> 后端 FastAPI 8000
  -> MySQL 保存患者和评估记录
  -> PyTorch 康复评分模型
  -> 本地 GGUF LLM 服务 6006
  -> Qwen2.5-7B-Instruct Q4_K_M 生成康复报告
```

## 2. 进入项目目录

先打开 PowerShell，进入项目根目录：

```powershell
cd "C:\Users\22097\Desktop\康复大模型项目\Rehabilitation-Assessment-System-main (1)\Rehabilitation-Assessment-System-main"
```

## 3. 启动本地 GGUF 大模型服务

本地大模型服务负责加载 Qwen2.5-7B-Instruct GGUF 分卷模型，默认端口是 `6006`。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-gguf-llm.ps1
```

默认模型路径：

```text
C:\Users\22097\Downloads\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
```

第二个分卷必须和第一个分卷放在同一个目录，且文件名不能改：

```text
C:\Users\22097\Downloads\qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
```

## 4. 检查本地 GGUF 大模型服务

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-gguf-llm.ps1 -Generate
```

正常结果应看到类似：

```text
[OK] port 6006 listening
[OK] GGUF /health ...
[OK] GGUF /generate_messages Yes, the local GGUF model service is available.
```

## 5. 启动后端和前端

这一步会同时启动：

```text
后端 FastAPI: 8000
前端 Vite/React: 5173
```

执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-local.ps1
```

## 6. 检查整个系统

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-local.ps1
```

正常结果应看到：

```text
[OK] .venv
[OK] frontend node_modules
[OK] backend .env
[OK] MySQL service MySQL57 Running
[OK] port 8000 listening
[OK] port 5173 listening
[OK] backend /api/health ok
[OK] backend /api/stats/summary ...
[OK] frontend page HTTP 200
```

其中 `backend .env` 应显示：

```text
LLM_PROVIDER=remote
LLM_REMOTE_URL=http://127.0.0.1:6006
LLM_REMOTE_TIMEOUT=300
```

这说明后端已经切到本地 GGUF 大模型服务。

## 7. 打开系统

浏览器访问：

```text
http://localhost:5173
```

## 8. 演示业务流程

建议演示顺序：

```text
1. 打开前端系统
2. 进入“任务一与任务三对接接口页面”
3. 选择医院端或设备端数据来源
4. 上传患者评估 zip 数据包
5. 点击“解析数据包”
6. 查看患者基本信息自动回填
7. 点击“开始分析”
8. 等待系统完成评分和报告生成
9. 查看 FMA-UE / BI / 手部肌张力 / Brunnstrom 结果
10. 查看 AI 康复评估报告
11. 查看 MySQL 评估记录列表
12. 进入详情页查看 trial、biomarker、报告和 prediction JSON
```

演示时可以这样解释：

```text
系统完成了从患者数据导入、康复评分、本地大模型报告生成，到 MySQL 结构化存储和前端查看的完整闭环。
```

## 9. 停止系统

停止后端和前端：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-local.ps1
```

停止本地 GGUF 大模型服务：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-gguf-llm.ps1
```

如果端口被旧进程占用，可以强制停止指定端口：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-local.ps1 -StopPortProcesses
powershell -ExecutionPolicy Bypass -File .\scripts\stop-gguf-llm.ps1 -StopPortProcess
```

## 10. 常见问题

### 10.1 没有先启动 6006

如果没有先启动本地 GGUF 服务，后端可以启动，但生成 AI 报告时会失败。

解决：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-gguf-llm.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\check-gguf-llm.ps1 -Generate
```

### 10.2 前端打不开

确认 `5173` 是否监听：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-local.ps1
```

如果未启动，重新执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-local.ps1
```

### 10.3 后端生成报告失败

先确认 `backend\.env` 中是本地 GGUF 模式：

```env
LLM_PROVIDER=remote
LLM_REMOTE_URL=http://127.0.0.1:6006
LLM_REMOTE_TIMEOUT=300
```

再确认 GGUF 服务可用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-gguf-llm.ps1 -Generate
```

### 10.4 MySQL 连接失败

检查 MySQL 服务：

```powershell
Get-Service -Name MySQL*
```

检查 `backend\.env`：

```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_DB=rehab_mysql
```

### 10.5 GGUF 文件不能改名

两个分卷文件必须保持原名：

```text
qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
```

启动时只传第一个分卷，第二个分卷会由 llama.cpp 自动识别。

## 11. 一句话总结

```text
先启动本地 GGUF 大模型服务 6006，再启动后端 8000 和前端 5173，最后在浏览器打开 http://localhost:5173 完成演示。
```
