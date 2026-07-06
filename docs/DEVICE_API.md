# 设备端 HTTPS 对接接口

本文档用于训练设备端与云端康复评估系统对接。第一版采用：

```text
设备端上传评估 zip
→ 云端后台分析
→ 设备端按 job_id 轮询状态
→ 设备端下载 export.zip / result.json / report.pdf
→ 设备端 ACK 确认收到
```

不要求训练设备有公网 IP，所有通信均由设备端主动通过 HTTPS 请求云端。

## 1. 鉴权

所有 `/api/device/v1/*` 接口都需要设备端 token：

```http
Authorization: Bearer <DEVICE_API_TOKEN>
```

云端在 `backend/.env` 中配置：

```env
DEVICE_API_TOKEN=generate-a-different-long-random-token
```

设备端 token 应与页面管理员 token `APP_AUTH_TOKEN` 分开。

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
    "imu-sampling_rate_hz": 50
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
  "job_id": "devjob_9f1c0d44e3a741df",
  "device_id": "device_001",
  "session_id": "7b0bb3d9f1a2",
  "assessment_id": "EVAL_20260629_001",
  "patient_id": "P001",
  "package_hash": "sha256...",
  "status": "queued",
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

完成后返回会包含文件地址：

```json
{
  "job_id": "devjob_9f1c0d44e3a741df",
  "status": "completed",
  "assessment_db_id": 8,
  "files": {
    "json": "/api/device/v1/jobs/devjob_9f1c0d44e3a741df/result.json",
    "pdf": "/api/device/v1/jobs/devjob_9f1c0d44e3a741df/report.pdf",
    "zip": "/api/device/v1/jobs/devjob_9f1c0d44e3a741df/export.zip"
  }
}
```

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

## 7. 联调建议

1. 先用 curl 上传一个已验证的设备端测试 zip。
2. 每 5-10 秒轮询一次状态，不建议高频轮询。
3. 首版只下载 `export.zip`，设备端从中读取 `result.json` 并保存 `report.pdf`。
4. 下载后校验 `manifest.json` 中的 sha256。
5. 成功保存后再 ACK。

## 8. 常见错误

| HTTP 状态 | 场景 |
|---:|---|
| 401 | 未提供 Bearer token |
| 403 | token 错误 |
| 413 | zip 或解压后数据超过限制 |
| 422 | zip 无效、缺少 manifest、缺少可用 active trial 或患者字段非法 |
| 425/409 | 任务尚未完成，暂不能下载 |
| 500/503 | 后端、MySQL、模型或导出服务异常 |
