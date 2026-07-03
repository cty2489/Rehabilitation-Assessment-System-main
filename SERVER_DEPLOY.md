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
├── backend_run.log                            # FastAPI 日志
├── gguf_server.log                            # GGUF LLM 日志
└── start_rehab_system.sh                      # 一键启动脚本
```

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

LLM_PROVIDER=remote
LLM_REMOTE_URL=http://127.0.0.1:6007
LLM_REMOTE_TIMEOUT=300

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

## 8. 访问方式

服务器内部：

```text
http://127.0.0.1:6006
```

公网访问使用云平台暴露的 6006 服务地址。例如 AutoDL/SeetaCloud 会提供类似：

```text
https://<instance-id>.bjb2.seetacloud.com:8443
```

打开页面后使用 `APP_ADMIN_USER` 和 `APP_ADMIN_PASSWORD` 登录。

## 9. 部署验证

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

## 10. 常见问题

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
