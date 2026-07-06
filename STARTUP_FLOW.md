# 启动与验证流程

本文档用于已经部署好的环境。首次部署请看 `SERVER_DEPLOY.md`，本地开发请看 `LOCAL_DEPLOY.md`。

## 1. 云服务器生产启动

标准部署会把仓库根目录的 `start_rehab_system.sh` 复制到 `/root/autodl-tmp/rehab_project/start_rehab_system.sh`。

```bash
bash /root/autodl-tmp/rehab_project/start_rehab_system.sh
```

脚本会启动或检查：

```text
MySQL      127.0.0.1:3306
Qwen3 HF   由 FastAPI 启动时尝试加载，用于当前报告生成
GGUF LLM   127.0.0.1:6007（可选回退/对照）
FastAPI    127.0.0.1:8000
Nginx      0.0.0.0:6006 -> frontend/dist + /api proxy
```

生产环境不启动 Vite `5173`。

## 2. 云服务器验证

```bash
curl -I http://127.0.0.1:6006/
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:6007/health   # 仅检查 GGUF 回退服务
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

业务数据接口必须登录后访问。裸访问应返回 `401 Bearer`：

```bash
curl -i http://127.0.0.1:8000/api/stats/summary
```

## 3. 公网访问

打开云平台提供的公网 HTTPS 地址，例如：

```text
https://<instance-id>.bjb2.seetacloud.com:8443
```

页面登录使用 `backend/.env` 中的：

```env
APP_ADMIN_USER
APP_ADMIN_PASSWORD
```

当前不启用 Nginx Basic Auth，因此浏览器不应弹出原生用户名/密码框。

## 4. 演示流程

```text
1. 打开公网地址
2. 页面内登录
3. 进入“任务一与任务三对接接口页面”
4. 选择医院端或设备端数据来源
5. 上传患者评估 zip 数据包
6. 点击“解析数据包”
7. 查看患者基本信息自动回填
8. 点击“开始分析”
9. 等待评分和报告生成
10. 查看 FMA-UE / BI / 手部肌张力 / Brunnstrom 结果
11. 查看 biomarker 表格和临床解读
12. 在记录详情中查看 trial、biomarker、报告和 prediction JSON
13. 下载 JSON / PDF / ZIP 结果文件，演示设备端交付格式
```

## 5. 结果文件

页面详情中的 `JSON`、`PDF`、`ZIP` 按钮会调用后端导出接口，并在服务器保存文件：

```text
/root/autodl-tmp/rehab_project/exports/assessments/{assessment_id}/
```

设备端推荐使用 `export.zip`，其中包含：

```text
result.json
report.pdf
manifest.json
```

## 6. 本地开发启动

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-gguf-llm.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start-local.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\check-local.ps1
```

访问：

```text
http://localhost:5173
```

## 7. 停止本地开发服务

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-local.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\stop-gguf-llm.ps1
```

## 8. 常见问题

### 仍然弹浏览器登录框

检查 Nginx 配置中是否还有 `auth_basic`：

```bash
grep auth_basic /etc/nginx/conf.d/rehab_demo.conf || true
nginx -t && nginx -s reload
```

### 页面登录失败

检查后端 `.env`：

```bash
grep '^APP_ADMIN_USER\\|^APP_ADMIN_PASSWORD\\|^APP_AUTH_TOKEN' backend/.env
```

修改后必须重启后端。

### 脚本显示后端已运行但页面接口失败

用 `/api/health` 判断后端是否活着；业务接口返回 `401` 表示服务正常但未登录。

### 报告生成慢

当前云端默认使用 Qwen3-8B HF 原版权重，FastAPI 启动时会尝试加载报告模型，因此启动耗时会比 remote/GGUF 模式更长。检查：

```bash
curl http://127.0.0.1:8000/api/health
tail -n 120 /root/autodl-tmp/rehab_project/backend_run.log
```

如果设置页切回了 `qwen25_7b_gguf`，再检查 GGUF 回退服务：

```bash
curl http://127.0.0.1:6007/health
tail -n 100 /root/autodl-tmp/rehab_project/gguf_server.log
```
