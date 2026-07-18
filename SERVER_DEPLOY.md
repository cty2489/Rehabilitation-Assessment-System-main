# 云服务器部署说明

本文档用于在 Ubuntu 云服务器、AutoDL 或 SeetaCloud 环境中部署智能康复评估系统。当前生产推荐方式是：Nginx 服务前端生产包，FastAPI 只监听本机，MySQL 只监听本机；MySQL 是患者、评估、导出和设备任务的必需业务存储；报告大模型当前推荐使用 FastAPI 进程内加载的 Qwen3-8B HF 原版权重。GGUF 服务仅作为手动回退/对照，不随生产启动脚本默认启动。

## 1. 服务拓扑

```text
公网浏览器
  -> 云平台 HTTPS 端口映射
  -> 服务器 Nginx 0.0.0.0:6006
       |-- /               -> frontend/dist
       |-- /api/*          -> 127.0.0.1:8000/api/*
  -> FastAPI 127.0.0.1:8000
       |-- MySQL 127.0.0.1:3306
       |-- RAG 服务 127.0.0.1:8010（可选 Shadow / 受控 Assist）
       |-- Qwen3-8B HF 本地报告模型（默认）
       |-- PyTorch 评分模型 DL_model/*.pth
```

生产环境不启动 Vite dev server。`5173` 仅用于本地开发。

## 2. 目录约定

```text
/root/autodl-tmp/rehab_project
├── Rehabilitation-Assessment-System-main      # 项目源码
├── ../Qwen_data/Qwen3-8B                      # 当前推荐报告模型，HF 原版格式
├── ../Qwen_data/DeepSeek-R1-Distill-Qwen-7B   # 候选对照模型，HF 原版格式
├── ../Qwen_data/InternLM3-8B-Instruct         # 候选对照模型，HF 原版格式
├── models/qwen2.5-7b-instruct-gguf            # 手动可选 GGUF 模型分卷
├── mysql_conf/my.cnf                          # MySQL 配置
├── mysql_data                                 # MySQL 数据目录
├── mysql_logs                                 # MySQL 日志
├── mysql_run                                  # MySQL socket/pid
├── mysql_tmp                                  # MySQL 临时目录
├── exports                                    # 评估结果 JSON/PDF/ZIP 导出文件
├── backend_run.log                            # FastAPI 日志
├── gguf_server.log                            # 手动可选 GGUF LLM 日志
├── start_rehab_system.sh                      # 生产一键启动脚本
└── start_gguf_fallback.sh                     # 手动 GGUF 回退/对照启动脚本
```

### 2.1 从 GitHub 获取源码

首次部署建议从 GitHub 拉取当前稳定基线：

```bash
mkdir -p /root/autodl-tmp/rehab_project
cd /root/autodl-tmp/rehab_project
git clone https://github.com/cty2489/Rehabilitation-Assessment-System-main.git
cd Rehabilitation-Assessment-System-main
git checkout cloud-server-v1.1.24
```

如果是继续开发或验证最新代码，也可以使用 `main` 分支：

```bash
git checkout main
git pull
```

仓库不包含真实密钥、患者数据、MySQL 数据目录、HF/GGUF 大模型文件和康复评分 `.pth` 权重。这些文件需要按本机实际路径单独准备。

## 3. 系统依赖

建议准备三个隔离环境：

| 环境 | 路径示例 | 用途 |
|---|---|---|
| 后端环境 | `/root/autodl-tmp/envs/rehab_backend` | FastAPI、PyTorch、biomarker、MySQL client |
| LLM 环境 | `/root/autodl-tmp/envs/llm_env` | 手动可选 llama-cpp-python / GGUF 回退服务 |
| Node 环境 | `/root/autodl-tmp/envs/node_env` | 前端构建 |

后端依赖：

```bash
apt-get update
apt-get install -y fonts-dejavu-core

cd /root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main
source /root/autodl-tmp/envs/rehab_backend/bin/activate
pip install -r backend/requirements.txt
```

如果要使用 Qwen3-8B / DeepSeek-R1-Distill-Qwen-7B 这类 HF 原版权重在后端进程内出报告，还需要在后端环境安装已验证的本地推理依赖。当前云端验证组合如下，没有升级服务器原有 PyTorch：

```bash
source /root/autodl-tmp/envs/rehab_backend/bin/activate
pip install \
  transformers==4.52.4 \
  accelerate==0.30.1 \
  bitsandbytes==0.43.3 \
  peft==0.11.1 \
  sentencepiece==0.2.1
```

仅当需要手动使用 GGUF 回退/对照服务时，再安装 GGUF 依赖：

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
ALLOW_LEGACY_ADMIN_BEARER=0
ALLOW_LEGACY_DEVICE_TOKEN=0
DEVICE_API_TOKEN=
DEVICE_API_TOKENS_JSON='{"device_002":"generate-device-002-token","device_003":"generate-device-003-token"}'
DEVICE_REQUIRE_REGISTERED_PATIENT=0

LLM_PROVIDER=local
LLM_REMOTE_URL=
LLM_REMOTE_TIMEOUT=300
LLM_MODEL_ROOT=/root/autodl-tmp/rehab_project/models
LLM_ORIGINAL_MODEL_ROOT=/root/autodl-tmp/Qwen_data
LLM_LOAD_4BIT=0
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

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
MAX_UPLOAD_FILE_BYTES=536870912
MAX_SESSION_UPLOAD_BYTES=2147483648
MAX_ZIP_BYTES=1073741824
MAX_ZIP_EXTRACTED_BYTES=4294967296
MAX_ZIP_MEMBERS=500
MAX_ZIP_COMPRESSION_RATIO=200
MIN_FREE_DISK_BYTES=2147483648
MAX_TRIALS=30
SESSION_TTL_HOURS=168
EXPORT_ROOT=/root/autodl-tmp/rehab_project/exports
PDF_LATIN_FONT_PATH=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
```

### 5.1 大模型设置页

登录后进入左侧“模型设置”，可以选择下一次报告生成使用的大模型。页面只负责切换已验证的线上报告模型，不展示也不编辑本地权重路径或远程服务地址，避免业务操作误改部署路径。页面默认只展示已准备/已验证的 HF 原版权重候选：

| 类型 | 模型 |
|---|---|
| 国产 | Qwen3-8B、DeepSeek-R1-Distill-Qwen-7B、Baichuan2-7B-Chat、GLM-4-9B、InternLM3-8B-Instruct |
| 国外 | Mistral-7B-Instruct-v0.3 |

保存后会生成运行态配置：

```text
/root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main/backend/config/llm_settings.json
```

该文件不随 Git 提交，适合每台服务器按自己的模型路径独立保存。未点击“保存设置”前，后端继续使用 `.env` 中的 `LLM_PROVIDER`、`LLM_REMOTE_URL` 等配置；新部署建议使用 `LLM_PROVIDER=local`，由默认 `qwen3_8b_hf` 接管报告生成。

本地权重路径、远程服务地址、adapter 目录等属于部署配置，由 `.env`、`LLM_MODEL_ROOT`、`LLM_ORIGINAL_MODEL_ROOT` 或上述运行态配置文件管理。权重不存在的本地模型会显示为未就绪；权重存在但端到端报告 JSON 结构尚未验证通过的模型会显示为候选待验证，不能设为当前线上报告模型。

### 5.2 手势库配置

仓库只提供 `backend/config/gestures_26.example.json` 作为 26 手势库 schema 和候选动作示例。它不是临床确认库，不会自动启用。正式启用前请让康复团队审核/替换名称、适应分期、辅助力度和安全说明，然后复制为运行态文件并重启后端：

```bash
cp backend/config/gestures_26.example.json backend/config/gestures_26.json
bash /root/autodl-tmp/rehab_project/restart_rehab_backend.sh
```

`backend/config/gestures_26.json` 已加入 `.gitignore`，每台服务器可按自己的临床确认版本维护。未启用时，报告第四节会显示“手势库待补充”，不会让大模型生成具体训练手势。

可微调的原版 HF 模型默认按 `LLM_ORIGINAL_MODEL_ROOT` 优先查找。推荐放置示例：

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

| 模型 ID | 结论 |
|---|---|
| `qwen3_8b_hf` | 已通过端到端报告链路测试，可作为当前线上默认报告模型 |
| `deepseek_r1_distill_qwen7b` | 已通过真实 26 biomarker 报告 JSON 结构校验，可作为 baseline 对照 |
| `glm4_9b` | 已通过真实 26 biomarker 报告 JSON 结构校验，可作为 baseline 对照，生成较慢且文本偏模板化 |
| `baichuan2_7b_chat` | 已通过真实 26 biomarker 报告 JSON 结构校验，可作为国产低阶 baseline，不推荐默认 |
| `mistral7b_v03` | 已通过真实 26 biomarker 报告 JSON 结构校验，可作为国外 baseline 对照 |
| `internlm3_8b` | 已通过真实 26 biomarker 报告 JSON 结构校验，可作为国产 baseline 对照 |
| `qwen25_7b_gguf` | 不在默认模型设置页展示；仅通过 `start_gguf_fallback.sh` 手动作为回退/对照 |

如果不放在 `LLM_ORIGINAL_MODEL_ROOT`，其它本地 HF 权重也会按 `LLM_MODEL_ROOT` 查找，例如：

```text
/root/autodl-tmp/rehab_project/models/Baichuan2-7B-Chat
/root/autodl-tmp/rehab_project/models/GLM-4-9B-0414
/root/autodl-tmp/rehab_project/models/Mistral-7B-Instruct-v0.3
/root/autodl-tmp/rehab_project/models/InternLM3-8B-Instruct
```

如果模型放在其他位置，可以通过环境变量修改根目录或配置文件路径：

```env
LLM_ORIGINAL_MODEL_ROOT=/data/original_hf_models
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
map $http_x_forwarded_proto $rehab_forwarded_proto {
    default $http_x_forwarded_proto;
    ""      $scheme;
}

map $http_x_forwarded_host $rehab_forwarded_host {
    default $http_x_forwarded_host;
    ""      $http_host;
}

server {
    listen 6006;
    server_name _;

    root /root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main/frontend/dist;
    index index.html;

    # Nginx 只做外层上限；后端仍按文件类型执行更严格的限制。
    client_max_body_size 3g;
    client_body_timeout 1800s;

    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "no-referrer" always;
    add_header Permissions-Policy "camera=(), microphone=(), geolocation=()" always;
    add_header Content-Security-Policy "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'" always;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Host $rehab_forwarded_host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $rehab_forwarded_proto;
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

当前推荐不使用 Nginx Basic Auth，避免浏览器反复弹框。页面登录后由后端签发短时 HttpOnly 会话 Cookie；浏览器不保存长期管理员 token。

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
2. 启动 FastAPI `127.0.0.1:8000`
3. 确认 `frontend/dist/index.html` 存在，必要时构建
4. 启动或重载 Nginx `0.0.0.0:6006`

报告生成会进入全局队列，同一时间只运行一份 LLM 报告，前端会显示“报告排队中，前面还有 N 份”。评分结果会先生成，排队只影响报告文本阶段。

如需临时使用 GGUF 回退/对照服务，单独执行：

```bash
cp /root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main/start_gguf_fallback.sh \
  /root/autodl-tmp/rehab_project/start_gguf_fallback.sh
chmod +x /root/autodl-tmp/rehab_project/start_gguf_fallback.sh
bash /root/autodl-tmp/rehab_project/start_gguf_fallback.sh
```

### 7.1 可选 RAG Shadow 与内部试运行服务

RAG 不随生产一键脚本自动启动，也不与报告后端共用 Python 环境。生产基线使用 `shadow`：检索结果写入受限权限轨迹，但不会进入提示词，也不会改变网页、JSON 或 PDF 报告。未完成正式专家审核的结构化知识只能放入独立试运行集合；如需内部体验 Assist，必须显式启用审批和 Demo 提示词开关，并保留报告中的未审核警示与引用校验。

Assist 使用两类接口：`/v1/retrieve` 对去标识化综合问题做 BGE-M3 向量检索，`/v1/lookup` 对固定 biomarker 的唯一 `system_key` 做精确查找。后者不使用相似度或 Top-K，因此不会因为召回排序把某一项指标绑定到错误知识。当前试运行集合为 `rehab_knowledge_trial_v0_2`，35 条知识仍全部是 `clinical_ready=false`，只能用于内部技术验证。

独立服务只监听 `127.0.0.1:8010`，启动模板为：

```bash
cp /root/autodl-tmp/rehab_project/current/start_rag_service.sh \
  /root/autodl-tmp/rehab_project/start_rag_service.sh
chmod +x /root/autodl-tmp/rehab_project/start_rag_service.sh
bash /root/autodl-tmp/rehab_project/start_rag_service.sh
curl -f http://127.0.0.1:8010/health
```

知识原文、切块、索引和 `rag.env` 必须保存在 `/root/autodl-tmp/rehab_project/` 稳定数据目录，不要放进可替换的 Git release。正式门禁见 `docs/RAG_GROUNDING.md`；结构化审阅 JSON、内部试运行索引、真实模型冒烟和回退命令见 `docs/RAG_TRIAL_ASSIST.md`。

为了让管理员“知识与证据治理”页面读取同一份发布包，在 `backend/.env` 中同步配置：

```env
RAG_COLLECTION=rehab_knowledge_trial_v0_2
KNOWLEDGE_RUNTIME_ROOT=/root/autodl-tmp/rehab_project/knowledge_base/runtime
```

每次更新知识内容后必须重新运行 `rag_prepare_review_json.py`，确认生成 `sources.jsonl`，再重建索引。页面读取的是受治理发布包，Qdrant 只作为检索索引，不作为知识原始数据库。

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
biomarker 只进入 `biomarker_coverage.missing_keys`，不生成临床解读。启用 Assist 且实际采用
知识时，`knowledge_evidence` 保存逐句数字引用、条目 ID、知识状态、审核状态、来源 ID 和去重
参考文献；网页、Word、JSON 与 PDF 使用同一套 `【1】【2】` 编号。未使用知识时该对象保留稳定
结构并返回 `used_in_report=false`。

接口：

```text
GET  /api/mysql/assessments/{id}/export.json
GET  /api/mysql/assessments/{id}/report.pdf
GET  /api/mysql/assessments/{id}/export.zip
POST /api/mysql/assessments/{id}/exports/regenerate
```

这些接口都需要页面登录后的会话。设备端自动对接时，推荐优先拉取 `export.zip`。

## 9. 设备端 HTTPS 对接

训练设备端不要使用页面管理员账号。管理员应在“系统管理 → 设备凭证”为每台设备生成独立 token。环境变量只建议用于首次引导或旧设备迁移：

```env
ALLOW_LEGACY_DEVICE_TOKEN=0
DEVICE_API_TOKEN=
```

第一版设备端流程：

```text
POST /api/device/v1/patients             注册患者基本资料（首次评估前）
POST /api/device/v1/assessments          上传 active 评估 zip 并获取 job_id
POST /api/device/v1/assessments/raw      application/zip 直传 zip 的兼容接口
GET  /api/device/v1/jobs/{job_id}        查询 queued/running/completed/failed
GET  /api/device/v1/jobs/{job_id}/export.zip
POST /api/device/v1/jobs/{job_id}/ack    设备端确认已保存结果
```

新患者由设备端生成 `DEV001_0001` 形式的全局 `patient_id`，先按
`rehab.patient.v1` JSON Schema 注册；网络超时后重复提交同一患者不会创建重复记录，
身份字段冲突则返回 HTTP 409。设备升级期间保持
`DEVICE_REQUIRE_REGISTERED_PATIENT=0`；全部设备完成联调后改为 `1` 并重启后端，
此后评估上传只读取云端患者档案，未注册编号返回 HTTP 404。

上传时建议发送 `Idempotency-Key: <device_id>:<assessment_id>`。网页评估和
设备评估共用单 GPU FIFO 队列；设备状态响应包含 `phase`、`queue_position`、
`queue_ahead`、`progress_percent` 和 `poll_after_seconds`。设备 ZIP 默认持久化到
`/root/autodl-tmp/rehab_project/device_jobs`，服务重启后会恢复未完成任务。
设备ACK成功后该任务的原始上传副本会清理，导出的结果文件和数据库记录继续保留。
未 ACK 的终态任务原始包默认保留 168 小时，之后按 `DEVICE_INPUT_TTL_HOURS` 清理；
这不会删除结构化评估记录或导出结果。

统一队列是单进程调度器，生产环境必须保持一个 Uvicorn worker；不要给启动命令
增加 `--workers 2` 或更高值。同一服务器需要多进程/多GPU时，应改用独立队列服务。

`DEVICE_API_TOKEN` 是旧共享码，默认不会被接受。只有迁移旧设备时才临时设置
`ALLOW_LEGACY_DEVICE_TOKEN=1`，迁移完成后应恢复为 `0`。新增设备使用网页生成的
独立 token；也可在首次启动时通过 `DEVICE_API_TOKENS_JSON` 引导导入。独立 token
只能操作绑定 `device_id` 的任务。

首次启动会把上述环境变量中的设备码导入 MySQL `device_credentials` 表，只保存
SHA-256 哈希和掩码。数据库中存在凭证后，设备鉴权以数据库为准；管理员可在网页
“系统管理 → 设备凭证”生成、停用、轮换和撤销。确认导入成功后可从 `.env` 删除
`DEVICE_API_TOKEN` 与 `DEVICE_API_TOKENS_JSON` 明文，数据库中的哈希凭证继续有效。

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

# 后端存活与就绪检查
curl http://127.0.0.1:8000/api/health
curl -f http://127.0.0.1:8000/api/ready

# 登录并保存 HttpOnly 会话 Cookie（响应正文不会返回会话令牌）
curl -s -c /tmp/rehab-admin.cookies -H 'Content-Type: application/json' \
  -d '{"username":"your_admin_user","password":"change-this-password"}' \
  http://127.0.0.1:8000/api/auth/login

# 使用保存的 Cookie 验证管理员会话
curl -f -b /tmp/rehab-admin.cookies http://127.0.0.1:8000/api/auth/session
rm -f /tmp/rehab-admin.cookies

# 未登录访问业务数据应返回 401
curl -i http://127.0.0.1:8000/api/stats/summary

# 生产端口检查
ss -ltnp | grep -E ':(3306|33060|5173|6006|6008|8000|8010)' || true
```

期望：

```text
6006 监听 0.0.0.0
8000 监听 127.0.0.1
3306 监听 127.0.0.1
5173 不监听
33060 不监听
6008 默认不监听；只有手动启动 GGUF 回退/对照时才监听 127.0.0.1
8010 默认不监听；启用 RAG Shadow 时只监听 127.0.0.1
```

## 12. 常见问题

### 页面一直弹浏览器登录框

这是 Nginx Basic Auth。当前部署不推荐开启。检查：

```bash
grep auth_basic /etc/nginx/conf.d/rehab_demo.conf || true
nginx -t && nginx -s reload
```

### 页面能打开，但列表或统计加载失败

先确认页面内已经登录，并检查浏览器是否接受同站 Cookie。后端业务接口需要有效的短时会话，未登录或会话过期会返回 `401/403`。

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

确认后端环境安装了 `reportlab`，系统存在用于英文、数字和单位的 TrueType 字体：

```bash
apt-get update
apt-get install -y fonts-dejavu-core
test -f /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf && echo FONT_OK

source /root/autodl-tmp/envs/rehab_backend/bin/activate
pip install -r /root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main/backend/requirements.txt
```

默认会自动查找 DejaVu Sans。若服务器使用其他字体路径，在 `backend/.env` 设置
`PDF_LATIN_FONT_PATH=/absolute/path/to/font.ttf`，重启后端并重新生成报告。

### 报告大模型生成失败

如果当前设置页选择的是 `qwen3_8b_hf`，先检查权重目录和后端本地推理依赖：

```bash
test -d /root/autodl-tmp/Qwen_data/Qwen3-8B && echo OK
source /root/autodl-tmp/envs/rehab_backend/bin/activate
python - <<'PY'
import torch, transformers, accelerate, bitsandbytes
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("bitsandbytes", bitsandbytes.__version__)
PY
tail -n 120 /root/autodl-tmp/rehab_project/backend_run.log
```

如果人工切到 remote/GGUF 回退模式，先单独启动回退服务并检查：

```bash
bash /root/autodl-tmp/rehab_project/start_gguf_fallback.sh
curl http://127.0.0.1:6008/health
tail -n 100 /root/autodl-tmp/rehab_project/gguf_server.log
```

后端 `.env` 应临时改为：

```env
LLM_PROVIDER=remote
LLM_REMOTE_URL=http://127.0.0.1:6008
```
