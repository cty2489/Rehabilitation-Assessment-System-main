# 云服务器部署说明

本文档用于在 Ubuntu 云服务器、AutoDL 或 SeetaCloud 环境中部署智能康复评估系统。当前生产推荐方式是：Nginx 服务前端生产包，FastAPI 只监听本机，GGUF LLM 和 MySQL 均只监听本机。

## 1. 服务拓扑

```text
公网浏览器
  -> 云平台 HTTPS 端口映射
  -> 服务器 Nginx 0.0.0.0:6006
       |-- /               -> frontend/dist
       |-- /api/*          -> 127.0.0.1:8000/api/*
  -> FastAPI 127.0.0.1:8000
       |-- MySQL 127.0.0.1:3306
       |-- GGUF LLM 127.0.0.1:6007
       |-- PyTorch 评分模型 DL_model/*.pth
```

生产环境不启动 Vite dev server。`5173` 仅用于本地开发。

## 2. 目录约定

```text
/root/autodl-tmp/rehab_project
├── Rehabilitation-Assessment-System-main      # 项目源码
├── models/qwen2.5-7b-instruct-gguf            # GGUF 模型分卷
├── mysql_conf/my.cnf                          # MySQL 配置
├── mysql_data                                 # MySQL 数据目录
├── mysql_logs                                 # MySQL 日志
├── mysql_run                                  # MySQL socket/pid
├── mysql_tmp                                  # MySQL 临时目录
├── exports                                    # 评估结果 JSON/PDF/ZIP 导出文件
├── backend_run.log                            # FastAPI 日志
├── gguf_server.log                            # GGUF LLM 日志
└── start_rehab_system.sh                      # 一键启动脚本
```

### 2.1 从 GitHub 获取源码

首次部署建议从 GitHub 拉取当前稳定基线：

```bash
mkdir -p /root/autodl-tmp/rehab_project
cd /root/autodl-tmp/rehab_project
git clone https://github.com/cty2489/Rehabilitation-Assessment-System-main.git
cd Rehabilitation-Assessment-System-main
git checkout cloud-server-v1.1.0
```

如果是继续开发或验证最新代码，也可以使用 `main` 分支：

```bash
git checkout main
git pull
```

仓库不包含真实密钥、患者数据、MySQL 数据目录、GGUF 大模型文件和康复评分 `.pth` 权重。这些文件需要按本机实际路径单独准备。

## 3. 系统依赖

建议准备三个隔离环境：

| 环境 | 路径示例 | 用途 |
|---|---|---|
| 后端环境 | `/root/autodl-tmp/envs/rehab_backend` | FastAPI、PyTorch、biomarker、MySQL client |
| LLM 环境 | `/root/autodl-tmp/envs/llm_env` | llama-cpp-python / GGUF 服务 |
| Node 环境 | `/root/autodl-tmp/envs/node_env` | 前端构建 |

后端依赖：

```bash
cd /root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main
source /root/autodl-tmp/envs/rehab_backend/bin/activate
pip install -r backend/requirements.txt
```

GGUF 服务依赖：

```bash
source /root/autodl-tmp/envs/llm_env/bin/activate
pip install -r requirements-gguf-server.txt
```

前端依赖与构建：

```bash
cd /root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main/frontend
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/node_env
npm install
npm run build
```

## 4. MySQL 配置

推荐配置文件：

```text
/root/autodl-tmp/mysql_conf/my.cnf
```

关键项：

```ini
[mysqld]
user=mysql
datadir=/root/autodl-tmp/mysql_data
socket=/root/autodl-tmp/mysql_run/mysqld.sock
pid-file=/root/autodl-tmp/mysql_run/mysqld.pid
tmpdir=/root/autodl-tmp/mysql_tmp
log-error=/root/autodl-tmp/mysql_logs/error.log
bind-address=127.0.0.1
port=3306
mysqlx=0
skip-name-resolve
character-set-server=utf8mb4
collation-server=utf8mb4_unicode_ci
```

`mysqlx=0` 用于关闭 MySQL X Plugin，避免额外监听 `33060`。

初始化数据库和用户时请使用自己的密码：

```sql
CREATE DATABASE IF NOT EXISTS rehab_mysql DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'rehab_user'@'127.0.0.1' IDENTIFIED BY 'change-this-mysql-password';
GRANT ALL PRIVILEGES ON rehab_mysql.* TO 'rehab_user'@'127.0.0.1';
FLUSH PRIVILEGES;
```

## 5. 后端环境变量

复制模板：

```bash
cd /root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main
cp backend/.env.example backend/.env
```

至少修改这些值：

```env
APP_ADMIN_USER=your_admin_user
APP_ADMIN_PASSWORD=change-this-password
APP_AUTH_TOKEN=generate-a-long-random-token
DEVICE_API_TOKEN=generate-a-different-long-random-token

LLM_PROVIDER=remote
LLM_REMOTE_URL=http://127.0.0.1:6007
LLM_REMOTE_TIMEOUT=300
LLM_MODEL_ROOT=/root/autodl-tmp/rehab_project/models

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=rehab_user
MYSQL_PASSWORD=change-this-mysql-password
MYSQL_DB=rehab_mysql
```

生成随机 token：

```bash
python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
```

上传限额也在 `.env` 中配置。默认值适合演示环境：

```env
MAX_UPLOAD_FILE_BYTES=2147483648
MAX_ZIP_BYTES=4294967296
MAX_ZIP_EXTRACTED_BYTES=10737418240
MAX_ZIP_MEMBERS=2000
MAX_TRIALS=30
SESSION_TTL_HOURS=168
EXPORT_ROOT=/root/autodl-tmp/rehab_project/exports
```

### 5.1 大模型设置页

登录后进入“系统管理 → 大模型设置”，可以选择下一次报告生成使用的大模型，也可以直接保存每个模型的本地权重路径或远程服务地址。页面默认内置 7 个候选：

| 类型 | 模型 |
|---|---|
| 国产 | Qwen2.5-7B-Instruct GGUF、Qwen3-8B、DeepSeek-R1-Distill-Qwen-7B、Baichuan2-7B-Chat、GLM-4-9B |
| 国外 | Mistral-7B-Instruct-v0.3、Llama-3-8B-Instruct |

保存后会生成运行态配置：

```text
/root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main/backend/config/llm_settings.json
```

该文件不随 Git 提交，适合每台服务器按自己的模型路径独立保存。未点击“保存设置”前，后端继续使用 `.env` 中的 `LLM_PROVIDER`、`LLM_REMOTE_URL` 等配置，便于兼容老部署。权重路径不存在的本地模型会显示为未就绪，不能设为当前报告模型。

本地 HF 权重默认按 `LLM_MODEL_ROOT` 查找，例如：

```text
/root/autodl-tmp/rehab_project/models/Qwen3-8B
/root/autodl-tmp/rehab_project/models/DeepSeek-R1-Distill-Qwen-7B
/root/autodl-tmp/rehab_project/models/Baichuan2-7B-Chat
/root/autodl-tmp/rehab_project/models/GLM-4-9B-0414
/root/autodl-tmp/rehab_project/models/Mistral-7B-Instruct-v0.3
/root/autodl-tmp/rehab_project/models/Meta-Llama-3-8B-Instruct
```

如果模型放在其他位置，可以通过环境变量修改根目录或配置文件路径：

```env
LLM_MODEL_ROOT=/data/models
LLM_SETTINGS_PATH=/data/rehab_config/llm_settings.json
```

## 6. Nginx 配置

配置文件：

```text
/etc/nginx/conf.d/rehab_demo.conf
```

推荐内容：

```nginx
server {
    listen 6006;
    server_name _;

    root /root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main/frontend/dist;
    index index.html;

    client_max_body_size 4g;
    client_body_timeout 1800s;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 1800s;
        proxy_read_timeout 1800s;
        proxy_buffering off;
        proxy_request_buffering off;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

当前推荐不使用 Nginx Basic Auth，避免浏览器反复弹框。系统身份验证由页面登录和后端 Bearer token 完成。

## 7. 一键启动

仓库根目录提供启动脚本模板：

```bash
cp /root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main/start_rehab_system.sh \
  /root/autodl-tmp/rehab_project/start_rehab_system.sh
chmod +x /root/autodl-tmp/rehab_project/start_rehab_system.sh
```

启动：

```bash
bash /root/autodl-tmp/rehab_project/start_rehab_system.sh
```

它会依次处理：

1. 启动 MySQL `127.0.0.1:3306`
2. 启动 GGUF LLM `127.0.0.1:6007`
3. 启动 FastAPI `127.0.0.1:8000`
4. 确认 `frontend/dist/index.html` 存在，必要时构建
5. 启动或重载 Nginx `0.0.0.0:6006`

## 8. 结果文件导出

评估结果会按需生成到：

```text
/root/autodl-tmp/rehab_project/exports/assessments/{assessment_id}/
├── result.json
├── report.pdf
├── manifest.json
└── export.zip
```

文件用途：

| 文件 | 用途 |
|---|---|
| `result.json` | 给设备端/系统端读取的 v2 精简结构化结果 |
| `report.pdf` | 给医生、患者或设备端留档查看，内容与 v2 结构一致且不重复粘贴整篇 Markdown |
| `manifest.json` | 文件版本、生成时间、sha256 校验信息 |
| `export.zip` | 打包结果，适合设备端一次性拉取或人工发送 |

`result.json` 的 `schema_version` 为 `rehab.assessment_result.v2`。正式交付文件不再包含
`report.content`、`biomarkers_raw`、`prediction_json` 或 trial 全量调试字段；数据不足的
biomarker 只进入 `biomarker_coverage.missing_keys`，不生成临床解读。

接口：

```text
GET  /api/mysql/assessments/{id}/export.json
GET  /api/mysql/assessments/{id}/report.pdf
GET  /api/mysql/assessments/{id}/export.zip
POST /api/mysql/assessments/{id}/exports/regenerate
```

这些接口都需要页面登录后的 Bearer token。设备端自动对接时，推荐优先拉取 `export.zip`。

## 9. 设备端 HTTPS 对接

训练设备端不要使用页面管理员账号。云端为设备端提供独立 token：

```env
DEVICE_API_TOKEN=generate-a-different-long-random-token
```

第一版设备端流程：

```text
POST /api/device/v1/assessments          上传 active 评估 zip 并获取 job_id
POST /api/device/v1/assessments/raw      application/zip 直传 zip 的兼容接口
GET  /api/device/v1/jobs/{job_id}        查询 queued/running/completed/failed
GET  /api/device/v1/jobs/{job_id}/export.zip
POST /api/device/v1/jobs/{job_id}/ack    设备端确认已保存结果
```

详细协议见 `docs/DEVICE_API.md`。

## 10. 访问方式

服务器内部：

```text
http://127.0.0.1:6006
```

公网访问使用云平台暴露的 6006 服务地址。例如 AutoDL/SeetaCloud 会提供类似：

```text
https://<instance-id>.bjb2.seetacloud.com:8443
```

打开页面后使用 `APP_ADMIN_USER` 和 `APP_ADMIN_PASSWORD` 登录。

## 11. 部署验证

```bash
# 前端入口
curl -I http://127.0.0.1:6006/

# 后端健康检查
curl http://127.0.0.1:8000/api/health

# 登录接口
curl -s -H 'Content-Type: application/json' \
  -d '{"username":"your_admin_user","password":"change-this-password"}' \
  http://127.0.0.1:8000/api/auth/login

# 未登录访问业务数据应返回 401 Bearer
curl -i http://127.0.0.1:8000/api/stats/summary

# 生产端口检查
ss -ltnp | grep -E ':(3306|33060|5173|6006|6007|8000)' || true
```

期望：

```text
6006 监听 0.0.0.0
6007 监听 127.0.0.1
8000 监听 127.0.0.1
3306 监听 127.0.0.1
5173 不监听
33060 不监听
```

## 12. 常见问题

### 页面一直弹浏览器登录框

这是 Nginx Basic Auth。当前部署不推荐开启。检查：

```bash
grep auth_basic /etc/nginx/conf.d/rehab_demo.conf || true
nginx -t && nginx -s reload
```

### 页面能打开，但列表或统计加载失败

先确认页面内已经登录。后端业务接口需要 Bearer token，未登录访问会返回 `401`。

### 重启脚本误判 Nginx 失败

确认脚本检查的是首页 `200`，不是旧版本 Basic Auth 的 `401`。

### MySQL 33060 仍在监听

确认 `/root/autodl-tmp/mysql_conf/my.cnf` 中有：

```ini
mysqlx=0
```

然后重启 MySQL。

### 上传大 zip 失败

同时检查三处：

1. Nginx `client_max_body_size`
2. 后端 `.env` 中的 `MAX_ZIP_BYTES`
3. 服务器磁盘空间 `/root/autodl-tmp`

### PDF 导出失败

确认后端环境安装了 `reportlab`：

```bash
source /root/autodl-tmp/envs/rehab_backend/bin/activate
pip install -r /root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main/backend/requirements.txt
```

### GGUF 报告生成失败

检查：

```bash
curl http://127.0.0.1:6007/health
tail -n 100 /root/autodl-tmp/rehab_project/gguf_server.log
```

后端 `.env` 应为：

```env
LLM_PROVIDER=remote
LLM_REMOTE_URL=http://127.0.0.1:6007
```
