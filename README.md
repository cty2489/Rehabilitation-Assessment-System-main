# 智能康复评估系统

本项目是一个面向康复评估业务的完整 Web 系统，支持患者入组、评估数据包导入、EEG/EMG/IMU 多模态评分、26 项 biomarker 输出、AI 康复报告生成、MySQL 结构化存储和前端可视化查看。

> `cloud-server-v1.1.16` 是当前云服务器稳定标签，已完成真实 GPU、MySQL、设备数据包与 JSON/PDF/ZIP 回传整链路验收。

## 当前稳定基线

当前云服务器可运行基线版本：

```text
cloud-server-v1.1.16
```

该标签提供已在线上验证过的运行基线；当前分支在此基础上包含：

- Nginx 生产入口，不再依赖 Vite dev server
- FastAPI 后端、MySQL、Qwen3-8B HF 本地报告模型联动；GGUF 服务已降级为手动回退/对照，不随生产脚本默认启动
- 页面登录和后端业务接口保护
- 26 项 biomarker 计算、报告解读和缺失项标记
- 网页与设备端完整评估共用 FIFO 队列，避免单卡 GPU 并发导致互相拖慢或 OOM；前端和设备 API 都会返回排队信息
- 设备任务支持持久化恢复、`Idempotency-Key` 去重、阶段/进度查询、结果下载和幂等 ACK
- 每台设备使用独立 token 且只能访问本设备任务；旧共享 token 仅在显式迁移开关开启时可用
- 管理员可在“系统管理 → 设备凭证”生成、查看掩码、停用、轮换和撤销设备码；数据库仅保存哈希，明文只显示一次
- 满负载报告使用动态 token 预算，减少 26 biomarker 报告截断后静默降级；保守 fallback 会在报告中显式标注
- 评估结果 `result.json`、`report.pdf`、`export.zip` 持久化导出
- 独立“模型设置”页可切换已验证的报告大模型，默认只展示已准备/已验证的 HF 原版权重候选模型；Qwen3-8B、DeepSeek-R1-Distill-Qwen-7B、GLM-4-9B、Mistral-7B-Instruct-v0.3、Baichuan2-7B-Chat 与 InternLM3-8B-Instruct 已通过端到端报告结构校验
- 模型权重路径属于服务器部署配置，不在业务页面暴露；未通过报告结构校验的候选模型不能设为当前线上模型
- BI/改良 Barthel 指数已从当前上肢手功能在线推理、页面展示、统计和导出报告中移除，数据库字段仅保留旧记录兼容
- 手势库默认不启用具体处方；仓库仅提供 `backend/config/gestures_26.example.json`，需临床审核后复制为运行态 `gestures_26.json`
- 云服务器启动、验证、常见问题和本地开发文档

后续模型优化、设备接入和论文实验建议都从该标签或其后的 `main` 分支继续开发。

当前推荐部署方式是云服务器生产模式：

```text
浏览器
  -> 公网 HTTPS 地址（云平台端口映射）
  -> Nginx 6006，服务 frontend/dist 并反向代理 /api
  -> FastAPI 8000
  -> MySQL 3306 + PyTorch 评分模型 + Qwen3-8B HF 报告模型
```

## 功能概览

- 患者档案和评估记录管理
- 医院端/设备端离线 zip 数据包解析
- FMA-UE、手部肌张力、Brunnstrom 手功能分期预测
- 设备端结果显式标注为工程链路验证状态；完成设备通道映射、域偏移和临床效度验证前，不作为独立诊疗依据
- 26 项关键 biomarker 计算、展示和报告解读
- 独立“模型设置”页可选择报告生成大模型，默认只展示已准备/已验证的 HF 原版权重模型
- 当前云端默认使用 Qwen3-8B HF 原版权重生成康复评估报告；DeepSeek-R1-Distill-Qwen-7B、GLM-4-9B、Mistral-7B-Instruct-v0.3、Baichuan2-7B-Chat 和 InternLM3-8B-Instruct 可在“模型设置”中切换为 baseline 对照；Qwen2.5-7B-Instruct GGUF 仅保留为手动回退/对照
- MySQL 保存患者、评估主记录、trial 明细、biomarker 明细和报告
- React 前端提供仪表盘、患者管理、康复评估、记录总览和统计分析
- 页面内登录保护，浏览器使用短时 HttpOnly 会话 Cookie，不在 localStorage 保存管理员密钥
- 评估结果可导出 `result.json`、`report.pdf`、`export.zip`，其中 JSON/PDF 采用去重后的设备端交付结构

## 当前服务端口

| 服务 | 地址 | 说明 |
|---|---|---|
| Nginx | `0.0.0.0:6006` | 生产入口，服务前端静态文件并代理 `/api` |
| FastAPI 后端 | `127.0.0.1:8000` | 业务接口、推理编排、报告生成、数据库接口 |
| MySQL | `127.0.0.1:3306` | 业务数据库 |
| Vite dev | `5173` | 仅本地开发使用，生产部署不启动 |

MySQL X Plugin 已在生产配置中关闭，`33060` 不应监听。GGUF LLM 不随生产启动脚本启动；如需临时回退/对照，可手动执行 `start_gguf_fallback.sh` 启动 `127.0.0.1:6008`。

## 快速启动已有云服务器

云服务器重启后执行：

```bash
bash /root/autodl-tmp/rehab_project/start_rehab_system.sh
```

正常结束时会看到：

```text
===== 全部启动完成 =====
云服务器内部前端：Nginx 6006 -> frontend/dist
云服务器内部后端：http://127.0.0.1:8000/docs
```

公网访问地址由云平台端口映射生成，例如：

```text
https://<实例标识>.bjb2.seetacloud.com:8443
```

页面打开后使用 `backend/.env` 中的 `APP_ADMIN_USER` 和 `APP_ADMIN_PASSWORD` 登录。

## 从 GitHub 部署到新云服务器

适用环境：Ubuntu 20.04/22.04、AutoDL、SeetaCloud 或同类 GPU 云服务器。

1. 获取源码：

```bash
mkdir -p /root/autodl-tmp/rehab_project
cd /root/autodl-tmp/rehab_project
git clone https://github.com/cty2489/Rehabilitation-Assessment-System-main.git
cd Rehabilitation-Assessment-System-main

# 推荐先部署当前稳定基线；后续开发可直接使用 main
git checkout cloud-server-v1.1.16
```

2. 准备外部文件：

仓库不包含真实模型权重、数据库、患者数据和 `.env` 密钥。部署前需要准备：

```text
DL_model/*.pth                                           # 康复评分模型权重
/root/autodl-tmp/Qwen_data/Qwen3-8B                      # 当前推荐报告模型，HF 原版格式
/root/autodl-tmp/Qwen_data/DeepSeek-R1-Distill-Qwen-7B   # 候选对照模型，HF 原版格式
/root/autodl-tmp/Qwen_data/Baichuan2-7B-Chat             # 可选候选 baseline
/root/autodl-tmp/Qwen_data/GLM-4-9B-Chat                 # 可选候选 baseline
/root/autodl-tmp/Qwen_data/Mistral-7B-Instruct-v0.3      # 可选候选 baseline
/root/autodl-tmp/Qwen_data/InternLM3-8B-Instruct         # 可选候选 baseline
/root/autodl-tmp/rehab_project/models/.../*.gguf         # 手动可选 GGUF 回退/对照模型
backend/.env                                            # 后端环境变量
MySQL 数据目录或初始化 SQL
```

3. 配置后端环境变量：

```bash
cp backend/.env.example backend/.env
vim backend/.env
```

至少修改：

```env
APP_ADMIN_USER=your_admin_user
APP_ADMIN_PASSWORD=change-this-password
APP_AUTH_TOKEN=generate-a-long-random-token
MYSQL_PASSWORD=change-this-mysql-password
EXPORT_ROOT=/root/autodl-tmp/rehab_project/exports
PDF_LATIN_FONT_PATH=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
```

4. 安装依赖、构建前端、配置 MySQL 和 Nginx：

完整命令请按 [`SERVER_DEPLOY.md`](SERVER_DEPLOY.md) 执行。部署好后复制启动脚本：

```bash
cp /root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main/start_rehab_system.sh \
  /root/autodl-tmp/rehab_project/start_rehab_system.sh
chmod +x /root/autodl-tmp/rehab_project/start_rehab_system.sh
```

5. 启动并验证：

```bash
bash /root/autodl-tmp/rehab_project/start_rehab_system.sh
curl -I http://127.0.0.1:6006/
curl http://127.0.0.1:8000/api/health
```

如果云平台把服务器 `6006` 映射成公网 HTTPS 地址，浏览器打开映射地址即可访问系统。

## 文档入口

| 文档 | 用途 |
|---|---|
| `SERVER_DEPLOY.md` | 云服务器/AutoDL/SeetaCloud 从零部署和生产运维 |
| `LOCAL_DEPLOY.md` | Windows 本地开发部署 |
| `STARTUP_FLOW.md` | 已部署环境的启动、验证和演示流程 |
| `docs/DEVICE_API.md` | 训练设备端 HTTPS 上传、轮询、下载、ACK 接口 |
| `docs/RAG_INGESTION.md` | RAG 知识入库第一步、质量门禁和私有资料目录约定 |
| `docs/schemas/device-job-v1.schema.json` | 设备任务状态响应的机器校验 schema |
| `CHANGELOG.md` | 稳定版本和重要变更记录 |
| `backend/.env.example` | 后端环境变量模板 |
| `start_rehab_system.sh` | 云服务器生产启动脚本模板 |
| `start_gguf_fallback.sh` | 手动启动 GGUF 回退/对照服务 |
| `README_Rehabilitation_Assessment_Report_Generation.md` | 报告生成和模型研究说明 |
| `biomarkers/README.md` | biomarker 计算说明 |
| `llm/README.md` | LLM 训练、评估和模型注册说明 |

## 目录结构

```text
backend/                     FastAPI 后端
frontend/                    React/Vite 前端
Deeplearning/                康复评分模型代码
biomarkers/                  biomarker 计算与证据元数据（不作为临床参考范围展示）
llm/                         transformers/LoRA 相关代码
rag/                         RAG 文档解析、知识治理和后续检索代码
knowledge_base/              RAG 配置与评测集；原文和运行数据不提交 Git
DL_model/                    康复评分模型权重目录，不随仓库上传
llm_gguf_server.py           手动可选 GGUF 大模型 HTTP 服务
start_gguf_fallback.sh       手动可选 GGUF 回退/对照启动脚本
requirements-gguf-server.txt 手动可选 GGUF 服务依赖
requirements-llm-server.txt  transformers LLM 服务依赖
scripts/                     Windows 本地开发启动/检查脚本
```

## 重要配置

后端配置文件：

```text
backend/.env
```

生产部署至少需要配置：

```env
APP_ADMIN_USER=your_admin_user
APP_ADMIN_PASSWORD=change-this-password
APP_AUTH_TOKEN=generate-a-long-random-token

LLM_PROVIDER=local
LLM_REMOTE_URL=
LLM_REMOTE_TIMEOUT=300
LLM_LOAD_4BIT=0
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=rehab_user
MYSQL_PASSWORD=change-this-mysql-password
MYSQL_DB=rehab_mysql
EXPORT_ROOT=/root/autodl-tmp/rehab_project/exports
```

### 大模型选择配置

左侧“模型设置”页会调用：

```text
GET   /api/settings/llm
PATCH /api/settings/llm
PATCH /api/settings/llm/models/{model_id}
```

保存后的运行配置默认写入：

```text
backend/config/llm_settings.json
```

该文件属于服务器运行态配置，已加入 `.gitignore`，不要提交。新服务器第一次启动且还未保存页面配置时，后端仍按 `.env` 中的 `LLM_PROVIDER`、`LLM_REMOTE_URL` 等旧配置运行；管理员在页面点击“保存设置”后，后续报告生成才由该配置文件接管。

业务页面只做“选择哪个已验证模型出报告”，不展示也不编辑权重路径。本地权重路径、远程服务地址、adapter 目录等属于部署配置，建议由运维/开发人员通过 `.env`、`LLM_MODEL_ROOT`、`LLM_ORIGINAL_MODEL_ROOT` 或 `backend/config/llm_settings.json` 管理。权重存在但端到端报告 JSON 结构未验证通过的模型会显示为候选待验证，不能设为当前线上报告模型。

### 手势库配置

仓库提供 `backend/config/gestures_26.example.json` 作为 26 手势库 schema 和候选动作示例。它不是临床确认库，不会自动启用。正式启用步骤：

```bash
cp backend/config/gestures_26.example.json backend/config/gestures_26.json
# 临床团队审核/替换名称、适应分期、辅助力度和安全说明后，重启后端
```

`backend/config/gestures_26.json` 属于运行态配置，已加入 `.gitignore`。未启用时，报告会保留“手势库待补充”占位，不让大模型生成具体训练手势。

默认候选模型包括：

```text
国产：Qwen3-8B、DeepSeek-R1-Distill-Qwen-7B、Baichuan2-7B-Chat、GLM-4-9B、InternLM3-8B-Instruct
国外：Mistral-7B-Instruct-v0.3
```

本地 HF 原版权重用于 baseline 和后续微调，默认优先查找：

```env
LLM_ORIGINAL_MODEL_ROOT=/root/autodl-tmp/Qwen_data
```

可自动识别的 HF 原版权重目录示例：

```text
/root/autodl-tmp/Qwen_data/Qwen3-8B
/root/autodl-tmp/Qwen_data/DeepSeek-R1-Distill-Qwen-7B
/root/autodl-tmp/Qwen_data/Baichuan2-7B-Chat
/root/autodl-tmp/Qwen_data/GLM-4-9B-0414
/root/autodl-tmp/Qwen_data/GLM-4-9B-Chat
/root/autodl-tmp/Qwen_data/Mistral-7B-Instruct-v0.3
/root/autodl-tmp/Qwen_data/InternLM3-8B-Instruct
```

当前云端验证结论：

```text
qwen3_8b_hf：已通过端到端报告链路测试，当前推荐作为线上默认报告模型。
deepseek_r1_distill_qwen7b：已通过真实 26 biomarker 报告 JSON 结构校验，采用分段结构化生成避免 R1 推理模型输出截断；可在页面切换为 baseline 对照，但当前生成耗时约 2 分钟/份，临床文本质量仍建议后续通过知识库增强或微调优化。
glm4_9b：GLM-4-9B-Chat 已通过真实 26 biomarker 报告 JSON 结构校验，采用分段结构化生成和较高重复惩罚避免整段输出截断/复读；本次 `mysql_assessment_33` 验证耗时约 2.5 分钟/份，临床文本偏模板化，仍建议后续通过知识库增强或微调优化。
mistral7b_v03：已通过真实 26 biomarker 报告 JSON 结构校验，可在页面切换为国外 baseline 对照；本次 `mysql_assessment_33` 验证耗时约 2.7 分钟/份，仍建议后续用真实病例集做批量质量评测。
baichuan2_7b_chat：已通过真实 26 biomarker 报告 JSON 结构校验，可在页面切换为国产低阶 baseline；本次 `mysql_assessment_33` 验证耗时约 1.5 分钟/份，但临床文本存在模板化、占位化和复制输入行倾向，不推荐作为默认报告模型。
internlm3_8b：已通过真实 26 biomarker 报告 JSON 结构校验，可在页面切换为国产 baseline 对照。
qwen25_7b_gguf：不再出现在默认“模型设置”候选列表，也不随生产脚本启动；仅在需要临时回退/对照时手动执行 `start_gguf_fallback.sh`。
```

其它本地 HF 权重默认查找根目录：

```env
LLM_MODEL_ROOT=/root/autodl-tmp/rehab_project/models
```

如果需要改配置文件位置，可设置 `LLM_SETTINGS_PATH`。

不要提交真实的 `backend/.env`、数据库密码、API key、模型权重和患者数据。

## 常用验证

```bash
# 前端入口应返回 200
curl -I http://127.0.0.1:6006/

# 后端存活检查；完整就绪检查必须返回 200
curl http://127.0.0.1:8000/api/health
curl -f http://127.0.0.1:8000/api/ready

# 未登录访问业务数据应返回 401
curl -i http://127.0.0.1:8000/api/stats/summary

# 登录后可下载评估结果文件
# GET /api/mysql/assessments/{id}/export.json
# GET /api/mysql/assessments/{id}/report.pdf
# GET /api/mysql/assessments/{id}/export.zip

# 端口检查：生产不应出现 5173、33060；6008 只在手动启动 GGUF 回退时出现
ss -ltnp | grep -E ':(3306|33060|5173|6006|6008|8000)' || true
```

代码变更提交前运行：

```bash
python -m pip install -r backend/requirements.txt -r backend/requirements-dev.txt
PYTHONPATH=backend:. python -m pytest backend -q
cd frontend && npm ci && npm run build
cd .. && bash -n start_rehab_system.sh start_gguf_fallback.sh
```

CI 只运行不依赖 GPU/模型权重的单元测试，使用轻量的 `backend/requirements-test.txt`；真实模型、MySQL、PDF 文件和设备数据包仍须在部署环境执行端到端验收。

## 版本说明

推荐规则：

```text
稳定演示/复现实验：使用 cloud-server-v1.1.16
日常继续开发：使用 main
```

如需查看所有稳定点：

```bash
git tag --list --sort=-creatordate
```
