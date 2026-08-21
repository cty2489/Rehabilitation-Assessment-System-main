![Logo](logo.png)
# 智能康复评估平台智能康复平台（早期旧版已停止维护更新）

本项目由珠海复旦创新研究院的医学人工智能科技创新中心研发团队开发，EEG / EMG / IMU 多模态融合康复评估系统：**FastAPI 后端 + React 前端**。新版包含统一 RAG 候选知识库、原文页码追溯和大模型评测基础设施；生产数据、模型权重和受控知识库文件不随代码仓库发布。

## 新版发布内容

- `backend/llm_benchmark/`：固定评测输入、Prompt、报告渲染、批处理、日志、盲评和 ROUGE/BLEU 评估契约。
- `backend/clinical_pipeline/`：现有 planner-RAG 流程及其契约、验证和测试代码。
- `backend/clinical_knowledge_graph/`：非 IMU 临床知识图谱原型，默认不启用。
- `frontend/`：新版管理、评测准备、知识治理和来源查看界面。
- `knowledge_base/v1_candidate/`：RAG 构建脚本、版本配置、来源追溯契约和验收清单。

新版运行时知识库的验收快照为 269 个 sources、15355 个 chunks、15355 个 bge-m3 embeddings/points，实际 collection 为 `rehab_knowledge_v1_candidate_next`。原始 PDF、OCR 全文、Qdrant 数据、患者数据、医生评分表、报告结果和模型权重需通过受控部署流程另行提供，严禁直接提交到公开仓库。

## 4 项康复指标

| 任务键 | 临床量表 | 类型 |
|---|---|---|
| `FMA_UE` | FMA 手部评分 | 回归 0–20 |
| `BI` | Barthel 指数 | 回归 0–100 |
| `hand_tone` | Hand MAS（Modified Ashworth 手部肌张力） | 6 级 `0/1/1+/2/3/4` |
| `hand_function` | Brunnstrom 分期（手） | 分期 2–6 |

## 目录结构

```
backend/          FastAPI 服务（main / inference / report / schemas / db）
frontend/         Vite + React + TS 单页前端
DL_model/         已训练的 .pth 模型（各任务最优 fold，不入库，需自行放置）
Deeplearning/     预处理 / 模型代码（被 backend 复用）
biomarkers/       26 项生物标志物计算（EMG14 / EEG6 / IMU6）
llm/              大模型推理 / 微调代码（被 backend/report.py 与 llm_server.py 复用）
llm_server.py     云 GPU 上的 LLM 推理服务（远程模式下被 backend 调用）
```

## 环境要求

- Python 3.10+、Node.js 18+
- 报告生成使用 **Qwen2.5-7B-Instruct**（32K 上下文），4-bit 推理需要 CUDA GPU + bitsandbytes
- 无 GPU 的主机（如 Mac）可用「远程模式」，把报告生成委托给云 GPU

## 启动后端

后端有两种报告生成模式，由 `.env` 里的 `LLM_REMOTE_URL` 决定。

### 方式 A · 本机（CPU）+ 云 GPU 报告（无 GPU 主机推荐）

4 项指标预测在本地 CPU 完成，报告生成委托给云 GPU 上的 `llm_server.py`。

```bash
# 1) 云 GPU 实例：启动 LLM 服务（以 AutoDL 为例，需监听 0.0.0.0:6006）
pip install -r requirements-llm-server.txt
LLM_MODEL_ID=qwen25_7b LLM_USE_ADAPTER=0 LLM_LOAD_4BIT=1 \
    uvicorn llm_server:app --host 0.0.0.0 --port 6006
# 基座 Qwen/Qwen2.5-7B-Instruct 首次启动自动从 HuggingFace 拉取

# 2) 本机后端：只装非 LLM 依赖
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env：LLM_REMOTE_URL 填云端公网地址；LLM_MODEL_ID=qwen25_7b
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 方式 B · 整机在 GPU 上（本地模式）

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r ../requirements-llm-server.txt
cp .env.example .env
# 编辑 .env：清空 LLM_REMOTE_URL；LLM_MODEL_ID=qwen25_7b；LLM_USE_ADAPTER=0
uvicorn main:app --host 0.0.0.0 --port 8000
```

启动时会从 `../DL_model/*.pth` 加载 4 个 CMK-AGN 模型。`GET /api/health` 返回已加载的任务列表。

> DL 模型权重（`.pth`）未随仓库上传，请将各任务的模型文件放入 `DL_model/` 后再启动。

## 启动前端

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

Vite 已配置代理 `/api → http://localhost:8000`，无需跨域配置。

## 使用流程

1. 登录（前端演示登录，输入任意账号即可进入）。
2. 填写患者基本信息（编号 / 姓名 / 性别 / 年龄 / 诊断 / 病程 / 偏瘫侧）。
3. 为每个 trial 上传一对 EEG CSV 和 EMG/IMU CSV（文件顺序需对应）。
4. 点击「开始评估」，前端通过 SSE 实时显示处理进度、4 项预测结果，以及流式生成的 AI 报告。
5. 完成后可导出报告或重新评估。每次评估写入 `backend/rehab.db`（首次启动自动创建）。

## 主要 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/assess` | multipart 上传患者信息 + EEG/EMG 文件，返回 `session_id` |
| `GET` | `/api/assess/{session_id}/stream` | SSE 推送处理进度 |
| `GET` | `/api/assess/{session_id}/result` | 断线重连后获取最终结果 |
| `GET` | `/api/patients`、`/api/assessments`、`/api/stats/summary` | 患者 / 评估记录 / 统计 |
| `GET` | `/api/health` | 健康检查 |
