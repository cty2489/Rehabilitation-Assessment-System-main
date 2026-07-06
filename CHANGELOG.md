# Changelog

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
