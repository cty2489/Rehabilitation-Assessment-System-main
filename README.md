# 智能康复评估系统

这是一个面向康复评估业务的完整本地化系统，当前版本已支持：

- 患者评估 zip 数据包导入
- EEG / EMG / IMU 多模态康复评分
- 26 项 biomarker 提取与展示
- 本地 Qwen2.5-7B-Instruct GGUF 大模型生成康复评估报告
- MySQL 结构化保存患者、评估记录、运动 trial、biomarker 明细和 AI 报告
- React 前端列表页、详情页、统计页展示

当前推荐分支：

```text
main
```

当前稳定版本标签：

```text
local-gguf-llm-v1.3
```

## 0. 分支和版本说明

对外部署和项目复现统一使用 `main` 分支。

`main` 表示当前完整可运行版本，包含前端、后端、MySQL 结构化存储、PyTorch 评分模型调用、本地 GGUF 大模型服务和部署文档。别人克隆仓库、阅读 README、复现系统时，只需要看 `main`。

`local-gguf-llm-v1.3` 是当前推荐的稳定版本标签，用来固定“本地 Qwen2.5-7B GGUF + MySQL 业务系统 + 完整部署说明 + 单一主分支说明”这一版。如果以后继续迭代，`main` 会继续向前更新，而标签可以用来回到这个确定版本。

仓库对外只保留 `main` 这一条主分支。历史阶段不通过重复分支保存，而是通过版本标签保存。这样别人打开仓库时不会误以为有两套不同代码。

## 1. 系统架构

```text
浏览器前端: http://localhost:5173
        |
        v
React / Vite 前端
        |
        v
FastAPI 后端: http://localhost:8000
        |
        +-- MySQL: patients / assessments / assessment_trials / assessment_biomarkers
        |
        +-- PyTorch 康复评分模型: FMA-UE / BI / hand_tone / hand_function
        |
        +-- 本地 GGUF LLM 服务: http://localhost:6006
              |
              v
           Qwen2.5-7B-Instruct Q4_K_M
```

## 2. 当前系统组成

| 模块 | 技术 | 默认端口 / 位置 | 说明 |
|---|---|---|---|
| 前端 | React + Vite + TypeScript | `5173` | Web 操作界面 |
| 后端 | FastAPI + Uvicorn | `8000` | 业务接口、数据解析、评分、报告生成编排 |
| 数据库 | MySQL | `3306` | 患者和评估结果结构化存储 |
| 康复评分模型 | PyTorch | 本机 GPU/CPU | 预测 4 项康复指标 |
| 本地大模型 | llama.cpp / llama-cpp-python | `6006` | 加载 Qwen2.5-7B GGUF，生成 AI 报告 |
| 备用大模型 | DeepSeek API | 云端 | 可切回 DeepSeek API 模式 |

## 3. 主要功能

### 3.1 患者与评估流程

系统支持从医院端或设备端评估数据包导入患者信息和运动数据：

```text
评估 zip 数据包
  -> 解析患者信息和 manifest
  -> 提取 trial / biomarker
  -> 康复评分模型预测
  -> 本地 Qwen2.5-7B 生成报告
  -> 写入 MySQL
  -> 前端列表页 / 详情页查看
```

### 3.2 康复评分指标

| 任务键 | 临床指标 | 类型 |
|---|---|---|
| `FMA_UE` | FMA-UE 上肢运动功能评分 | 回归 |
| `BI` | Barthel 指数 | 回归 |
| `hand_tone` | Hand MAS 手部肌张力 | 分类 |
| `hand_function` | Brunnstrom 手功能分期 | 分期 |

### 3.3 MySQL 结构化存储

核心表：

| 表名 | 说明 |
|---|---|
| `patients` | 患者主表，一名患者一条记录 |
| `assessments` | 评估主表，一名患者可对应多次评估 |
| `assessment_trials` | 每次评估下的运动 trial 明细 |
| `assessment_biomarkers` | 每次评估下的 biomarker 明细 |

关系：

```text
patients 1:N assessments
assessments 1:N assessment_trials
assessments 1:N assessment_biomarkers
```

数据库由后端启动时自动创建，建表逻辑在 `backend/mysql_db.py`。如果需要给老师或团队单独提交表结构，可以使用本地生成的 `docs/database/数据库表结构交付材料.zip`；该 `docs/` 目录默认不发布到 GitHub。

## 4. 运行环境要求

### 4.1 基础环境

- Windows 10/11
- Python 3.11
- Node.js LTS
- MySQL 5.7+ 或 MySQL 8+
- NVIDIA GPU 推荐：RTX 4070 或更高

### 4.2 当前已验证环境

```text
GPU: NVIDIA GeForce RTX 4070 12GB
PyTorch: 2.5.1+cu121
llama-cpp-python: 0.3.4 CUDA wheel
MySQL: MySQL57
```

### 4.3 不随仓库上传的大文件

以下文件不会上传到 GitHub，需要本地自行准备：

```text
backend/.env
DL_model/*.pth
*.gguf
*.db
node_modules/
.venv/
```

GGUF 模型默认放在：

```text
C:\Users\22097\Downloads\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
C:\Users\22097\Downloads\qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
```

两个分卷必须在同一目录下，文件名不能修改。启动时只指定第一个分卷，第二个分卷会自动识别。

## 5. 首次部署

### 5.1 克隆仓库

```powershell
git clone -b main https://github.com/cty2489/Rehabilitation-Assessment-System-main.git
cd Rehabilitation-Assessment-System-main
```

`main` 分支是当前对外推荐的完整系统版本。

如果需要固定到当前稳定标签：

```powershell
git checkout local-gguf-llm-v1.3
```

### 5.2 创建后端虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

安装后端依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
```

安装 GPU 版 PyTorch：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

安装本地 GGUF 服务依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install "llama-cpp-python==0.3.4" --only-binary=:all: --index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --extra-index-url https://pypi.org/simple
```

检查依赖：

```powershell
.\.venv\Scripts\python.exe -m pip check
```

### 5.3 安装前端依赖

```powershell
cd .\frontend
npm install
cd ..
```

### 5.4 配置后端环境变量

复制示例配置：

```powershell
Copy-Item .\backend\.env.example .\backend\.env
```

本地 GGUF 模式核心配置：

```env
LLM_PROVIDER=remote
LLM_REMOTE_URL=http://127.0.0.1:6006
LLM_REMOTE_TIMEOUT=300

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=你的MySQL密码
MYSQL_DB=rehab_mysql
```

如果改用 DeepSeek API 模式：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的DeepSeek密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

> 不要把真实 `backend/.env` 提交到 GitHub。

### 5.5 准备模型文件

康复评分模型 `.pth` 放入：

```text
DL_model/
```

本地大模型 GGUF 分卷放在同一目录，例如：

```text
C:\Users\22097\Downloads\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
C:\Users\22097\Downloads\qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
```

## 6. 启动流程

完整启动说明见：

- `STARTUP_FLOW.md`

最短启动流程如下。

### 6.1 进入项目根目录

```powershell
cd "C:\Users\22097\Desktop\康复大模型项目\Rehabilitation-Assessment-System-main (1)\Rehabilitation-Assessment-System-main"
```

如果是别人 clone 的目录，请换成自己的项目路径。

### 6.2 启动本地 GGUF 大模型服务

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-gguf-llm.ps1
```

检查：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-gguf-llm.ps1 -Generate
```

### 6.3 启动后端和前端

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-local.ps1
```

检查：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-local.ps1
```

### 6.4 打开系统

```text
http://localhost:5173
```

## 7. 停止系统

停止后端和前端：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-local.ps1
```

停止本地 GGUF 大模型服务：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-gguf-llm.ps1
```

如果端口被旧进程占用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-local.ps1 -StopPortProcesses
powershell -ExecutionPolicy Bypass -File .\scripts\stop-gguf-llm.ps1 -StopPortProcess
```

## 8. 演示流程

建议按以下顺序演示：

```text
1. 打开 http://localhost:5173
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

## 9. 主要 API

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/task-interface/parse` | 解析评估 zip 数据包 |
| `POST` | `/api/task-interface/offline` | 提交离线评估任务 |
| `GET` | `/api/assess/{session_id}/stream` | SSE 推送评估进度 |
| `GET` | `/api/patients` | 患者列表 |
| `GET` | `/api/patients/{id}` | 患者详情 |
| `GET` | `/api/assessments` | 评估记录列表 |
| `GET` | `/api/mysql/assessments` | MySQL 评估记录 |
| `GET` | `/api/mysql/assessments/{id}` | MySQL 评估详情 |
| `GET` | `/api/stats/summary` | 统计概览 |
| `GET` | `/api/health` | 后端健康检查 |
| `POST` | `http://localhost:6006/generate_messages` | 本地 GGUF 大模型生成接口 |

## 10. 目录结构

```text
backend/                         FastAPI 后端
frontend/                        React/Vite 前端
Deeplearning/                    康复评分模型代码
biomarkers/                      biomarker 计算与参考范围
llm/                             transformers/LoRA 相关代码
DL_model/                        康复评分模型权重目录，不随仓库上传
llm_server.py                    云 GPU transformers 大模型服务
llm_gguf_server.py               本地 GGUF 大模型服务
requirements-gguf-server.txt     GGUF 服务依赖说明
requirements-llm-server.txt      云 GPU transformers 服务依赖说明
LOCAL_DEPLOY.md                  本地部署详细说明
STARTUP_FLOW.md                  启动流程说明
scripts/start-local.ps1          启动后端和前端
scripts/check-local.ps1          检查后端、前端、MySQL
scripts/stop-local.ps1           停止后端和前端
scripts/start-gguf-llm.ps1       启动本地 GGUF 服务
scripts/check-gguf-llm.ps1       检查本地 GGUF 服务
scripts/stop-gguf-llm.ps1        停止本地 GGUF 服务
```

## 11. 常见问题

### 11.1 生成报告失败

先检查本地 GGUF 服务：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-gguf-llm.ps1 -Generate
```

再检查 `backend/.env`：

```env
LLM_PROVIDER=remote
LLM_REMOTE_URL=http://127.0.0.1:6006
LLM_REMOTE_TIMEOUT=300
```

### 11.2 前端打不开

检查：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-local.ps1
```

如果 `5173` 未监听，重新运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-local.ps1
```

### 11.3 MySQL 连接失败

检查 MySQL 服务：

```powershell
Get-Service -Name MySQL*
```

检查 `backend/.env` 中的账号、密码和库名。

### 11.4 GGUF 不能用 transformers 加载

当前 Qwen2.5-7B 是分卷 GGUF 量化模型，不是 FP16 / safetensors 格式。请使用 llama.cpp / llama-cpp-python 加载，不要使用 `AutoModelForCausalLM`。

### 11.5 模型文件太大不能上传 GitHub

这是正常的。`.gguf`、`.pth`、`.db`、`.env`、`.venv`、`node_modules` 都不应该上传 GitHub。仓库只保存代码、脚本和配置模板。

## 12. 版本标签

日常部署请直接使用 `main` 分支。下面这些标签用于回到某一个历史稳定版本，便于汇报、复现和问题定位。

| 标签 | 说明 |
|---|---|
| `deepseek-runnable` | DeepSeek API 可运行基础版本 |
| `mysql-business-system-v1` | MySQL 结构化存储和业务系统版本 |
| `local-gguf-llm-v1.3` | 本地 Qwen2.5-7B GGUF 大模型服务 + 完整部署说明 + 单一主分支说明 |
| `local-gguf-llm-v1.2` | 本地 Qwen2.5-7B GGUF 大模型服务 + 完整部署说明 + 单一主分支说明 |
| `local-gguf-llm-v1.1` | 本地 Qwen2.5-7B GGUF 大模型服务 + 最新 README 部署说明 |
| `local-gguf-llm-v1` | 本地 Qwen2.5-7B GGUF 大模型服务版本 |

## 13. 相关文档

- `STARTUP_FLOW.md`：当前系统启动流程
- `LOCAL_DEPLOY.md`：本地部署说明
- `README_Rehabilitation_Assessment_Report_Generation.md`：模型训练和报告生成研究说明
- `llm/README.md`：LLM 训练、评估和模型注册说明
