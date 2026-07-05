# Changelog

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
