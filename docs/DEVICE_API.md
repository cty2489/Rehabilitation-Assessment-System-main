# 设备端 HTTPS 对接接口

本文档用于训练设备端与云端康复评估系统对接。第一版采用：

```text
设备端上传评估 zip
→ 云端进入统一 FIFO 队列
→ 设备端按 job_id 轮询状态
→ 设备端下载 export.zip / result.json / report.pdf
→ 设备端 ACK 确认收到
```

不要求训练设备有公网 IP，所有通信均由设备端主动通过 HTTPS 请求云端。
网页评估与设备评估共用同一个单 GPU 队列，完整的深度模型、biomarker、
大模型报告和导出流程一次只运行一个任务。

## 1. 鉴权

所有 `/api/device/v1/*` 接口都需要设备端 token：

```http
Authorization: Bearer <DEVICE_TOKEN>
```

上传接口另外建议携带：

```http
Idempotency-Key: <device_id>:<assessment_id>
```

云端在 `backend/.env` 中配置：

```env
ALLOW_LEGACY_DEVICE_TOKEN=0
DEVICE_API_TOKEN=
DEVICE_API_TOKENS_JSON='{"device_002":"generate-device-002-token","device_003":"generate-device-003-token"}'
```

设备端 token 应与页面管理员 token `APP_AUTH_TOKEN` 分开。
`DEVICE_API_TOKEN` 是旧共享凭证且默认禁用；仅在短期迁移时设置
`ALLOW_LEGACY_DEVICE_TOKEN=1`。新设备应由管理员在网页生成按 `device_id`
独立分配的 token，也可由 `DEVICE_API_TOKENS_JSON` 在首次启动时导入。独立凭证提交任务时
会自动绑定设备ID，并且只能查询、下载和ACK本设备创建的任务。设备端若同时传入
`device_id`，它必须与凭证绑定的ID一致，否则返回 HTTP 403。

设备组之间不得互换 token，也不要把真实 token 写入源码、日志或 Git 仓库。
`Idempotency-Key` 不是鉴权信息，但设备端应为同一次评估始终使用相同值。
网络超时后用相同 key 重传，云端会返回原 `job_id`；相同 key 对应不同 ZIP
时返回 HTTP 409，避免误覆盖。

管理员登录网页后可在“系统管理 → 设备凭证”管理设备码。列表只显示掩码；完整
设备码仅在新建或轮换成功后显示一次。停用可恢复，撤销会立即失效；历史任务不会
随凭证撤销而删除。

## 2. 数据包格式

设备端上传一个 zip，一个 zip 代表一次评估。当前已验证的设备包结构：

```text
patient_P001_eval_20260629/
├── manifest.json
├── daily_training.json
├── active_assessment/
│   ├── action_SS1/
│   │   ├── trial_01_eeg.bdf
│   │   ├── trial_01_emg_imu.csv
│   │   ├── trial_02_eeg.bdf
│   │   └── trial_02_emg_imu.csv
│   ├── action_SS2/
│   └── action_SS3/
└── passive_assessment/
    ├── action_SS1/
    ├── action_SS2/
    └── action_SS3/
```

`manifest.json` 至少需要包含：

```json
{
  "patient_id": "P001",
  "device_id": "device_001",
  "assessment_id": "EVAL_20260629_001",
  "assessment_time": "2026-06-29T17:30:00+08:00",
  "data_description": {
    "assessment_types": ["active", "passive"],
    "actions_per_type": 3,
    "trials_per_action": 3,
    "emg_sampling_rate_hz": 200,
    "eeg_sampling_rate_hz": 512,
    "imu_included_in_emg_csv": true,
    "imu_sampling_rate_hz": 50
  },
  "assessments": [
    {
      "assessment_type": "active",
      "action_id": "action_SS1",
      "action_name": "握拳",
      "trials": [
        {
          "trial_index": 1,
          "emg_imu_file": "active_assessment/action_SS1/trial_01_emg_imu.csv",
          "eeg_file": "active_assessment/action_SS1/trial_01_eeg.bdf"
        }
      ]
    }
  ]
}
```

第一版云端只纳入 `active_assessment` 做深度学习评分。`passive_assessment` 可继续放在包内，后续用于被动活动范围、肌张力、痉挛相关分析。

当前设备格式到云端的传输、排队、推理和文件回传已完成工程链路验证，但设备通道与肌肉对应、跨设备域偏移以及临床效度尚未完成正式验证。因此设备端 JSON/PDF 会标注 `validation_status=engineering_validation_only`；该结果不能直接作为诊断或治疗决定，必须由康复专业人员结合量表、动作表现和信号质控复核。

`trial_*_emg_imu.csv` 当前格式：

```text
EMG采样时间点,EMG通道1..EMG通道8,
IMU采样时间点,IMU加速度计X/Y/Z,IMU陀螺仪X/Y/Z,IMU预留1/2/3
```

## 3. 上传评估数据包

云端支持两种上传方式：

```text
推荐方式：multipart/form-data，文件字段名 package
兼容方式：application/zip，直接把 zip 二进制作为请求体
```

如果设备端是 Python、Qt、浏览器或普通 HTTP client，优先用 `multipart/form-data`。
如果嵌入式程序只方便直传文件流，可以用 `application/zip`。

### 3.1 multipart/form-data 上传

```http
POST /api/device/v1/assessments
Content-Type: multipart/form-data
Authorization: Bearer <DEVICE_API_TOKEN>
```

表单字段：

| 字段 | 必填 | 说明 |
|---|---:|---|
| `package` | 是 | 设备端评估 zip |
| `institution` | 否 | 默认 `device` |
| `device_id` | 否 | 设备编号；也可由 `manifest.json` 提供 |
| `patient_id` | 否 | 患者编号；表单优先，其次读 `manifest.json` |
| `name` | 否 | 姓名；缺省时用患者档案或 patient_id |
| `sex` | 否 | `男` / `女`；缺省为患者档案或 `男` |
| `age` | 否 | 年龄 |
| `diagnosis` | 否 | 诊断；缺省为患者档案或 `未填写` |
| `disease_days` | 否 | 病程天数 |
| `paralysis_side` | 否 | `左` / `右`；缺省为患者档案或 `左` |

示例：

```bash
curl -X POST "https://<cloud-host>/api/device/v1/assessments" \
  -H "Authorization: Bearer ${DEVICE_API_TOKEN}" \
  -H "Idempotency-Key: device_001:EVAL_20260629_001" \
  -F "institution=device" \
  -F "device_id=device_001" \
  -F "package=@patient_P001_eval_20260629.zip" \
  -F "patient_id=P001" \
  -F "name=张三" \
  -F "sex=男" \
  -F "age=62" \
  -F "diagnosis=脑卒中" \
  -F "disease_days=120" \
  -F "paralysis_side=左"
```

Python 示例：

```python
import requests

CLOUD_URL = "https://<cloud-host>"
DEVICE_API_TOKEN = "设备端token"

headers = {
    "Authorization": f"Bearer {DEVICE_API_TOKEN}",
    "Idempotency-Key": "device_001:EVAL_20260629_001",
}
data = {
    "institution": "device",
    "device_id": "device_001",
    "patient_id": "P001",
    "name": "张三",
    "sex": "男",
    "age": "62",
    "diagnosis": "脑卒中",
    "disease_days": "120",
    "paralysis_side": "左",
}
with open("patient_P001_eval_20260629.zip", "rb") as f:
    files = {"package": ("patient_P001_eval_20260629.zip", f, "application/zip")}
    r = requests.post(
        f"{CLOUD_URL}/api/device/v1/assessments",
        headers=headers,
        data=data,
        files=files,
        timeout=300,
    )
r.raise_for_status()
print(r.json())
```

注意：使用 `requests` 的 `files=` 参数时，不要手动设置 `Content-Type`，让 `requests`
自动生成带 boundary 的 `multipart/form-data`。

### 3.2 application/zip raw 上传

如果设备端选择 `application/zip`，请求体就是 zip 文件本身，不再有 `package`
这个表单字段。元数据放在 query 参数或 `X-*` 请求头中。

```http
POST /api/device/v1/assessments/raw?patient_id=P001&name=张三&sex=男&age=62&diagnosis=脑卒中&disease_days=120&paralysis_side=左
Content-Type: application/zip
Authorization: Bearer <DEVICE_API_TOKEN>
X-Device-ID: device_001
X-Filename: patient_P001_eval_20260629.zip
```

curl 示例：

```bash
curl -X POST \
  "https://<cloud-host>/api/device/v1/assessments/raw?patient_id=P001&name=张三&sex=男&age=62&diagnosis=脑卒中&disease_days=120&paralysis_side=左" \
  -H "Authorization: Bearer ${DEVICE_API_TOKEN}" \
  -H "Idempotency-Key: device_001:EVAL_20260629_001" \
  -H "Content-Type: application/zip" \
  -H "X-Device-ID: device_001" \
  -H "X-Filename: patient_P001_eval_20260629.zip" \
  --data-binary "@patient_P001_eval_20260629.zip"
```

Python 示例：

```python
import requests

CLOUD_URL = "https://<cloud-host>"
DEVICE_API_TOKEN = "设备端token"

headers = {
    "Authorization": f"Bearer {DEVICE_API_TOKEN}",
    "Idempotency-Key": "device_001:EVAL_20260629_001",
    "Content-Type": "application/zip",
    "X-Device-ID": "device_001",
    "X-Filename": "patient_P001_eval_20260629.zip",
}
params = {
    "patient_id": "P001",
    "name": "张三",
    "sex": "男",
    "age": "62",
    "diagnosis": "脑卒中",
    "disease_days": "120",
    "paralysis_side": "左",
}
with open("patient_P001_eval_20260629.zip", "rb") as f:
    r = requests.post(
        f"{CLOUD_URL}/api/device/v1/assessments/raw",
        headers=headers,
        params=params,
        data=f,
        timeout=300,
    )
r.raise_for_status()
print(r.json())
```

返回：

```json
{
  "schema_version": "rehab.device_job.v1",
  "job_id": "devjob_9f1c0d44e3a741df",
  "device_id": "device_001",
  "session_id": "7b0bb3d9f1a2",
  "assessment_id": "EVAL_20260629_001",
  "patient_id": "P001",
  "package_hash": "sha256...",
  "status": "queued",
  "phase": "waiting",
  "queue_position": 2,
  "queue_ahead": 1,
  "progress_percent": 0,
  "poll_after_seconds": 5,
  "message": "已接收评估数据，等待处理",
  "status_url": "/api/device/v1/jobs/devjob_9f1c0d44e3a741df",
  "n_trials": 9,
  "parse_warnings": []
}
```

## 4. 查询任务状态

```bash
curl "https://<cloud-host>/api/device/v1/jobs/${JOB_ID}" \
  -H "Authorization: Bearer ${DEVICE_API_TOKEN}"
```

状态值：

| 状态 | 含义 |
|---|---|
| `queued` | 已接收，等待后台启动 |
| `running` | 正在分析 |
| `completed` | 分析完成，可下载 |
| `failed` | 分析失败，查看 `error_message` |
| `delivered` | 设备端已 ACK 确认收到 |

`status` 是设备程序控制流程的唯一依据。`phase` 只用于显示当前阶段：

| `phase` | 含义 |
|---|---|
| `waiting` | 等待队列调度 |
| `dl_inference` | 信号处理、深度模型评分和 biomarker 提取 |
| `llm_reporting` | 大模型生成临床报告 |
| `exporting` | 写入数据库并准备 JSON/PDF/ZIP |
| `finished` | 已完成 |
| `failed` | 已失败 |

排队时 `queue_position` 从 1 开始，`queue_ahead` 表示前方尚未完成的任务数；
运行或结束后两个字段为 0。设备端按 `poll_after_seconds` 轮询，不要根据中文
`message` 判断逻辑，也不要因为长时间处于 `queued/running` 而重复上传。

完成后返回会包含文件地址：

```json
{
  "schema_version": "rehab.device_job.v1",
  "job_id": "devjob_9f1c0d44e3a741df",
  "status": "completed",
  "phase": "finished",
  "queue_position": 0,
  "queue_ahead": 0,
  "progress_percent": 100,
  "assessment_db_id": 8,
  "files": {
    "json": "/api/device/v1/jobs/devjob_9f1c0d44e3a741df/result.json",
    "pdf": "/api/device/v1/jobs/devjob_9f1c0d44e3a741df/report.pdf",
    "zip": "/api/device/v1/jobs/devjob_9f1c0d44e3a741df/export.zip"
  }
}
```

失败时还会返回稳定错误结构：

```json
{
  "schema_version": "rehab.device_job.v1",
  "job_id": "devjob_9f1c0d44e3a741df",
  "status": "failed",
  "phase": "failed",
  "error": {
    "code": "ANALYSIS_FAILED",
    "message": "具体错误说明",
    "retryable": true
  }
}
```

机器可校验的响应 schema 位于 `docs/schemas/device-job-v1.schema.json`。
以后云端可能增加字段，设备端必须忽略不认识的字段。

## 5. 下载结果

推荐设备端优先下载 `export.zip`：

```bash
curl -L "https://<cloud-host>/api/device/v1/jobs/${JOB_ID}/export.zip" \
  -H "Authorization: Bearer ${DEVICE_API_TOKEN}" \
  -o export.zip
```

`export.zip` 包含：

```text
manifest.json
result.json
report.pdf
```

`result.json` 使用 `rehab.assessment_result.v2`。该版本是设备端友好的精简结构，
不会再重复保存整篇 Markdown 报告、原始 `biomarkers_raw`、`prediction_json`
或 trial 明细。设备端通常只需要读取以下字段：

```text
schema_version
report_metadata
patient_basic_info
stage_assessment
clinical_scores
biomarker_coverage
biomarker_interpretation_policy
biomarker_sections
knowledge_evidence
subtype_classification_and_treatment_strategy
next_week_training_plan
warnings_and_recommendations
natural_language_summary
```

其中 `biomarker_sections[].indicators[]` 已把每个可计算指标的当前值、解读和训练/随访
建议放在同一个对象中。`reference_range` 只保留证据类型、适用性、方向和来源 ID 等审计
元数据，并明确 `display=false`；网页和 PDF 不把它显示为临床参考范围。数据不足或当前
采集格式暂不支持的指标不会生成临床解读，只会进入
`biomarker_coverage.missing_keys`。

`biomarker_interpretation_policy` 规定：单次设备特异量不得用于判断正常或异常，只有同一
患者在相同设备、相同采集流程下的连续复测才可用于趋势观察。当前接口不计算队列排名或
百分位。

`knowledge_evidence` 用于审计本次报告实际采用的 RAG 知识，包含
`used_in_report`、`clinical_review_status`、`notice`、`entries` 和去重后的
`references`。每个条目会给出 `knowledge_id`、知识状态、审核说明和 `source_ids`。
当前内部试运行知识会明确返回 `clinical_review_status=demo_unreviewed`，设备端不得把它
显示为“专家已审核”或“临床指南”。RAG 未参与报告时，设备端应接受
`used_in_report=false` 和空数组，不应将其视为接口错误。

报告不再输出 EMG、EEG、IMU 各自的模态亚型，只在
`subtype_classification_and_treatment_strategy.subtype_classification.overall_subtype`
保留综合亚型。`overall_strategies` 只包含策略名称、训练剂量、反馈/调整原则和安全注意，
不包含“具体方法”；具体动作和设备配合由 `next_week_training_plan` 承载，避免重复。

也可分别下载：

```bash
curl -L "https://<cloud-host>/api/device/v1/jobs/${JOB_ID}/result.json" \
  -H "Authorization: Bearer ${DEVICE_API_TOKEN}" \
  -o result.json

curl -L "https://<cloud-host>/api/device/v1/jobs/${JOB_ID}/report.pdf" \
  -H "Authorization: Bearer ${DEVICE_API_TOKEN}" \
  -o report.pdf
```

## 6. ACK 确认收到

设备端成功保存结果后调用：

```bash
curl -X POST "https://<cloud-host>/api/device/v1/jobs/${JOB_ID}/ack" \
  -H "Authorization: Bearer ${DEVICE_API_TOKEN}"
```

云端会把任务状态更新为 `delivered` 并记录 `delivered_at`。
ACK 成功后，云端会删除用于任务恢复的原始上传副本；结构化评估记录和
`result.json` / `report.pdf` / `export.zip` 仍然保留。

若设备始终未 ACK，已完成或失败任务的原始上传副本也会在
`DEVICE_INPUT_TTL_HOURS`（默认 168 小时）后清理，避免磁盘无限增长；数据库记录和
导出结果不受影响。因此设备端仍应在校验并持久化三种结果文件后及时 ACK。

## 7. 联调建议

1. 先用 curl 上传一个已验证的设备端测试 zip。
2. 每 5-10 秒轮询一次状态，不建议高频轮询。
3. 首版只下载 `export.zip`，设备端从中读取 `result.json` 并保存 `report.pdf`。
4. 下载后校验 `manifest.json` 中的 sha256。
5. 成功保存后再 ACK。
6. 在本地持久化 `job_id`；设备程序重启后继续轮询原任务。
7. 上传超时但未拿到响应时，使用相同 `Idempotency-Key` 重传。

云端会持久化设备 ZIP 和任务快照。后端重启后，尚未完成的 `queued/running`
任务会按原创建顺序重新排队；`attempt_count` 会记录实际启动处理的次数。

## 8. 常见错误

| HTTP 状态 | 场景 |
|---:|---|
| 401 | 未提供 Bearer token |
| 403 | token 错误 |
| 403 | 独立设备 token 与请求 `device_id` 不一致，或访问了其他设备的任务 |
| 413 | zip 或解压后数据超过限制 |
| 422 | zip 无效、缺少 manifest、缺少可用 active trial 或患者字段非法 |
| 409 | 同一个 `Idempotency-Key` 被用于不同 ZIP，或任务尚未完成就请求下载 |
| 425/409 | 任务尚未完成，暂不能下载 |
| 500/503 | 后端、MySQL、模型或导出服务异常 |
