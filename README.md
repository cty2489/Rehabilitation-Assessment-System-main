# 智能康复评估系统

本项目是一个面向康复评估业务的完整 Web 系统，支持患者入组、评估数据包导入、EEG/EMG/IMU 多模态评分、26 项 biomarker 输出、AI 康复报告生成、MySQL 结构化存储和前端可视化查看。

## 当前稳定基线

当前云服务器可运行基线版本：

```text
cloud-server-v1.1.7
```

这个标签对应已经在线上验证过的版本，包含：

- Nginx 生产入口，不再依赖 Vite dev server
- FastAPI 后端、MySQL、Qwen3-8B HF 本地报告模型联动，GGUF 服务保留为回退/对照
- 页面登录和 Bearer token 业务接口保护
- 26 项 biomarker 计算、报告解读和缺失项标记
- 评估结果 `result.json`、`report.pdf`、`export.zip` 持久化导出
- 独立“模型设置”页可切换已验证的报告大模型，默认内置 5 个国产和 2 个国外候选模型
- 模型权重路径属于服务器部署配置，不在业务页面暴露；未通过报告结构校验的候选模型不能设为当前线上模型
- BI/改良 Barthel 指数已从当前上肢手功能在线推理、页面展示、统计和导出报告中移除，数据库字段仅保留旧记录兼容
- 云服务器启动、验证、常见问题和本地开发文档

后续模型优化、设备接入和论文实验建议都从该标签或其后的 `main` 分支继续开发。

当前推荐部署方式是云服务器生产模式：

```text
浏览器
  -> 公网 HTTPS 地址（云平台端口映射）
  -> Nginx 6006，服务 frontend/dist 并反向代理 /api
  -> FastAPI 8000
  -> MySQL 3306 + PyTorch 评分模型 + Qwen3-8B HF 报告模型
  -> 可选 GGUF LLM 6007 回退/对照服务
```

## 功能概览

- 患者档案和评估记录管理
- 医院端/设备端离线 zip 数据包解析
- FMA-UE、手部肌张力、Brunnstrom 手功能分期预测
- 26 项关键 biomarker 计算、展示和报告解读
- 独立“模型设置”页可选择报告生成大模型，默认内置 5 个国产和 2 个国外候选模型
- 当前云端默认使用 Qwen3-8B HF 原版权重生成康复评估报告；Qwen2.5-7B-Instruct GGUF 保留为回退/对照
- MySQL 保存患者、评估主记录、trial 明细、biomarker 明细和报告
- React 前端提供仪表盘、患者管理、康复评估、记录总览和统计分析
- 页面内登录保护，后端使用 Bearer token 保护读写接口
- 评估结果可导出 `result.json`、`report.pdf`、`export.zip`，其中 JSON/PDF 采用去重后的设备端交付结构

## 当前服务端口

| 服务 | 地址 | 说明 |
|---|---|---|
| Nginx | `0.0.0.0:6006` | 生产入口，服务前端静态文件并代理 `/api` |
| FastAPI 后端 | `127.0.0.1:8000` | 业务接口、推理编排、报告生成、数据库接口 |
| GGUF LLM | `127.0.0.1:6007` | 可选回退/对照：Qwen2.5-7B-Instruct 报告生成服务 |
| MySQL | `127.0.0.1:3306` | 业务数据库 |
| Vite dev | `5173` | 仅本地开发使用，生产部署不启动 |

MySQL X Plugin 已在生产配置中关闭，`33060` 不应监听。

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
git checkout cloud-server-v1.1.7
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
/root/autodl-tmp/rehab_project/models/.../*.gguf         # 可选 GGUF 回退/对照模型
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
| `CHANGELOG.md` | 稳定版本和重要变更记录 |
| `backend/.env.example` | 后端环境变量模板 |
| `start_rehab_system.sh` | 云服务器生产启动脚本模板 |
| `README_Rehabilitation_Assessment_Report_Generation.md` | 报告生成和模型研究说明 |
| `biomarkers/README.md` | biomarker 计算说明 |
| `llm/README.md` | LLM 训练、评估和模型注册说明 |

## 目录结构

```text
backend/                     FastAPI 后端
frontend/                    React/Vite 前端
Deeplearning/                康复评分模型代码
biomarkers/                  biomarker 计算与参考范围
llm/                         transformers/LoRA 相关代码
DL_model/                    康复评分模型权重目录，不随仓库上传
llm_gguf_server.py           GGUF 大模型 HTTP 服务
requirements-gguf-server.txt GGUF 服务依赖
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

LLM_PROVIDER=remote
LLM_REMOTE_URL=http://127.0.0.1:6007
LLM_REMOTE_TIMEOUT=300

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

默认候选模型包括：

```text
国产：Qwen2.5-7B-Instruct GGUF、Qwen3-8B、DeepSeek-R1-Distill-Qwen-7B、Baichuan2-7B-Chat、GLM-4-9B
国外：Mistral-7B-Instruct-v0.3、Llama-3-8B-Instruct
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
/root/autodl-tmp/Qwen_data/Meta-Llama-3-8B-Instruct
/root/autodl-tmp/Qwen_data/Llama-3-8B-Instruct
```

当前云端验证结论：

```text
qwen3_8b_hf：已通过端到端报告链路测试，可作为当前线上默认报告模型。
deepseek_r1_distill_qwen7b：权重可加载、可生成，但报告 JSON 结构尚未通过端到端校验，页面暂不允许切为线上报告模型。
glm4_9b：GLM-4-9B-Chat 可加载，小样例 JSON 可通过；真实 26 biomarker 报告在当前 token 预算内输出截断，暂不允许切为线上报告模型。
baichuan2_7b_chat：当前本地权重加载触发 PyTorch 2.6 torch.load 安全限制，暂不允许切为线上报告模型。
mistral7b_v03：当前云服务器目录缺少 tokenizer 文件，且已按空间策略删除，暂不允许切为线上报告模型。
qwen25_7b_gguf：保留为可用回退/对照，不再作为当前云端默认报告模型。
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

# 后端健康检查应返回 200
curl http://127.0.0.1:8000/api/health

# 未登录访问业务数据应返回 401 Bearer
curl -i http://127.0.0.1:8000/api/stats/summary

# 登录后可下载评估结果文件
# GET /api/mysql/assessments/{id}/export.json
# GET /api/mysql/assessments/{id}/report.pdf
# GET /api/mysql/assessments/{id}/export.zip

# 端口检查：生产不应出现 5173 和 33060
ss -ltnp | grep -E ':(3306|33060|5173|6006|6007|8000)' || true
```

## 版本说明

推荐规则：

```text
稳定演示/复现实验：使用 cloud-server-v1.1.7
日常继续开发：使用 main
```

如需查看所有稳定点：

```bash
git tag --list --sort=-creatordate
```
