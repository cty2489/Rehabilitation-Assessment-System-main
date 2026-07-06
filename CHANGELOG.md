# Changelog

## cloud-server-v1.1.5 - 2026-07-07

### 改进

- “模型设置”改为独立页面，仅用于切换已验证的报告大模型；本地权重路径、远程服务地址和 adapter 目录不再暴露在业务页面。
- “系统管理”页恢复为账户与服务状态管理，不再与“模型设置”重复。
- `deepseek_r1_distill_qwen7b` 标记为候选待验证：权重可存在，但因端到端报告 JSON 结构未通过校验，暂不允许切为线上报告模型。
- 当前线上报告模型恢复为 `qwen3_8b_hf`，避免误切换到未验证候选模型导致报告慢或结构失败。
- BI/改良 Barthel 指数从当前上肢手功能在线推理、页面结果卡、统计、入组表单和导出报告中移除；数据库字段仅保留旧记录兼容。
- 在线推理不再加载 BI 评分模型，减少启动和分析阶段的模型工作量。

## cloud-server-v1.1.4 - 2026-07-06

### 改进

- 左侧菜单新增“模型设置”入口，直接定位到大模型选择区域。
- 右上角用户菜单新增“模型设置”入口。
- 保留“系统管理”页面原有账户、系统状态和大模型设置功能。

## cloud-server-v1.1.3 - 2026-07-06

### 改进

- 当前云端报告模型从 `qwen25_7b_gguf` 切换为 `qwen3_8b_hf`，后者已通过端到端报告链路测试。
- 本地 HF 报告模型生成时会尽量关闭 Qwen3/DeepSeek-R1 风格的 thinking 输出，并在解析 JSON 前剥离 `<think>...</think>` 或孤立 `</think>` 前缀。
- 文档补充 HF 原版权重推理依赖的已验证版本组合，避免部署时误升级服务器现有 PyTorch/CUDA 环境。
- 明确 `deepseek_r1_distill_qwen7b` 当前仅作为候选对照：权重可加载、可生成，但报告 JSON 结构尚不稳定，不建议设为默认。
- `qwen25_7b_gguf` 保留为可用回退/对照服务。

## cloud-server-v1.1.2 - 2026-07-06

### 改进

- Qwen3-8B 和 DeepSeek-R1-Distill-Qwen-7B 默认优先识别 `/root/autodl-tmp/Qwen_data` 下的原版 HF 权重目录。
- 新增 `LLM_ORIGINAL_MODEL_ROOT` 配置项，用于存放后续 baseline 和微调使用的原版模型。
- 文档明确区分：GGUF 用于当前远程报告服务，HF 原版目录用于模型对比和后续微调。

## cloud-server-v1.1.1 - 2026-07-06

### 改进

- 系统管理页支持直接保存每个报告模型的本地权重路径或远程服务地址。
- Qwen3-8B 自动识别服务器已有路径 `/root/autodl-tmp/Qwen_data/Qwen3-8B`。
- 未放置权重或服务不可用的模型会显示为未就绪，后端拒绝将其设为当前报告模型。
- `llm/model_registry.py` 补齐 7 个 baseline 候选模型短名，训练、生成和页面配置使用同一套 `model_id`。
- `llm/README.md` 更新为 7 候选 baseline 说明。

## cloud-server-v1.1.0 - 2026-07-06

### 新增

- 系统管理页新增“大模型设置”，可选择报告生成使用的大模型。
- 默认内置 7 个报告模型候选：Qwen2.5-7B-Instruct GGUF、Qwen3-8B、DeepSeek-R1-Distill-Qwen-7B、Baichuan2-7B-Chat、GLM-4-9B、Mistral-7B-Instruct-v0.3、Llama-3-8B-Instruct。
- 后端新增 `GET /api/settings/llm` 和 `PATCH /api/settings/llm`，使用页面登录后的 Bearer token 保护。
- 运行态模型选择保存到 `backend/config/llm_settings.json`，不随 Git 提交。

### 改进

- 未保存页面配置前，报告生成仍兼容 `.env` 中的 `LLM_PROVIDER`、`LLM_REMOTE_URL` 等旧部署方式。
- 本地 HF 候选模型会显示权重路径是否存在；权重未放置时标记为 `not_ready`。
- `README.md`、`SERVER_DEPLOY.md` 和 `backend/.env.example` 补充大模型候选、权重根目录和设置文件说明。

## device-api-v0.3 - 2026-07-06

### 改进

- `result.json` 升级为 `rehab.assessment_result.v2`，改为设备端友好的精简结构。
- `report.pdf` 改为直接从 v2 结构渲染，不再先列 biomarker 简表再粘贴完整 Markdown 报告。
- 数据不足或当前采集格式暂不支持的 biomarker 只进入 coverage/missing_keys，不生成临床解读。

## cloud-server-v1.0.0 - 2026-07-05

当前云服务器可运行基线版本。后续模型优化、设备采集接入和论文实验建议以此为基础继续开发。

### 已验证能力

- Nginx 生产入口服务 `frontend/dist`，并代理 `/api` 到 FastAPI
- FastAPI、MySQL、GGUF LLM 服务可在云服务器内联动运行
- 页面登录保护和 Bearer token 业务接口保护
- 评估数据包上传、trial 解析、康复评分预测和 26 项 biomarker 输出
- AI 康复报告生成
- 评估结果持久化导出为 `result.json`、`report.pdf`、`export.zip`
- `start_rehab_system.sh` 一键启动并验证 MySQL、LLM、后端和 Nginx
- `README.md`、`SERVER_DEPLOY.md`、`LOCAL_DEPLOY.md`、`STARTUP_FLOW.md` 覆盖部署、启动和演示流程

### 不随仓库发布的内容

- 真实 `backend/.env`
- 数据库密码、API key、登录密码
- 患者数据和导出结果
- 大模型 GGUF 权重
- 康复评分模型 `.pth` 权重
- MySQL 数据目录

### 推荐部署方式

```bash
git clone https://github.com/cty2489/Rehabilitation-Assessment-System-main.git
cd Rehabilitation-Assessment-System-main
git checkout cloud-server-v1.0.0
```

然后按 `SERVER_DEPLOY.md` 完成模型权重、数据库、环境变量、前端构建、Nginx 和启动脚本配置。
