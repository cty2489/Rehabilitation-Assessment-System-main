# 智能康复评估系统

本项目是一个面向康复评估业务的完整 Web 系统，支持患者入组、评估数据包导入、EEG/EMG/IMU 多模态评分、26 项 biomarker 输出、AI 康复报告生成、MySQL 结构化存储和前端可视化查看。

当前推荐部署方式是云服务器生产模式：

```text
浏览器
  -> 公网 HTTPS 地址（云平台端口映射）
  -> Nginx 6006，服务 frontend/dist 并反向代理 /api
  -> FastAPI 8000
  -> MySQL 3306 + PyTorch 评分模型 + GGUF LLM 6007
```

## 功能概览

- 患者档案和评估记录管理
- 医院端/设备端离线 zip 数据包解析
- FMA-UE、BI、手部肌张力、Brunnstrom 手功能分期预测
- 26 项关键 biomarker 计算、展示和报告解读
- 本地 Qwen2.5-7B-Instruct GGUF 服务生成康复评估报告
- MySQL 保存患者、评估主记录、trial 明细、biomarker 明细和报告
- React 前端提供仪表盘、患者管理、康复评估、记录总览和统计分析
- 页面内登录保护，后端使用 Bearer token 保护读写接口

## 当前服务端口

| 服务 | 地址 | 说明 |
|---|---|---|
| Nginx | `0.0.0.0:6006` | 生产入口，服务前端静态文件并代理 `/api` |
| FastAPI 后端 | `127.0.0.1:8000` | 业务接口、推理编排、报告生成、数据库接口 |
| GGUF LLM | `127.0.0.1:6007` | Qwen2.5-7B-Instruct 报告生成服务 |
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

## 文档入口

| 文档 | 用途 |
|---|---|
| `SERVER_DEPLOY.md` | 云服务器/AutoDL/SeetaCloud 从零部署和生产运维 |
| `LOCAL_DEPLOY.md` | Windows 本地开发部署 |
| `STARTUP_FLOW.md` | 已部署环境的启动、验证和演示流程 |
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
```

不要提交真实的 `backend/.env`、数据库密码、API key、模型权重和患者数据。

## 常用验证

```bash
# 前端入口应返回 200
curl -I http://127.0.0.1:6006/

# 后端健康检查应返回 200
curl http://127.0.0.1:8000/api/health

# 未登录访问业务数据应返回 401 Bearer
curl -i http://127.0.0.1:8000/api/stats/summary

# 端口检查：生产不应出现 5173 和 33060
ss -ltnp | grep -E ':(3306|33060|5173|6006|6007|8000)' || true
```

## 版本说明

日常部署使用 `main` 分支。历史稳定状态可通过 Git tag 固定，但部署文档以当前 `main` 为准。
