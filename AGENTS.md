# 康复评估系统协作规范

本文件适用于项目根目录及其全部子目录。目标是让 Codex、Claude 及其他开发助手在不破坏生产系统、临床数据和既有研究成果的前提下继续工作。

## 1. 基本原则

- 先检查真实代码、数据和运行状态，再作判断；README、历史对话和旧报告只能作为线索。
- 不确定时明确说明不确定，不编造测试结果、文献、医学阈值、患者信息或部署状态。
- 只修改用户本次授权范围内的内容，保留其他人的未提交修改。
- 优先做可回滚、范围小、可验证的修改。
- 事实由程序和真实数据提供；LLM 只负责被授权的解释与文本生成。

## 2. 项目与仓库边界

- GitHub 主仓库：`https://github.com/cty2489/Rehabilitation-Assessment-System-main`。
- 默认开发目标为该仓库的 `main`；发布前必须重新获取远端最新 `main`。
- 本地项目根目录同时保存研究资料、输出和历史中间文件，不应直接视为可发布源码快照。
- 根目录当前可能存在大量已删除、未跟踪或迁移后的文件。不得清理、恢复或整体暂存这些内容，除非用户明确指定范围。
- 根目录的 `origin` 可能仍指向历史仓库；发布前必须核对远端 URL、默认分支和写权限。
- 向 GitHub 发布时，从最新目标 `main` 建立干净分支或工作树，再逐项复制经确认的文件。
- 禁止使用 `git add .`、`git add -A` 或 `git add --all`；只显式暂存本次文件。
- 不得用旧仓库的 README、配置或目录整体覆盖新版文件。合并前必须审阅与目标 `main` 的完整差异。

## 3. 信息安全与数据保护

以下内容不得提交到公开 GitHub，也不得写入本文件、README、日志或测试夹具：

- SSH 密码、数据库密码、API token、Cookie、`.env` 内容和私钥；
- 患者姓名、身份证明、医院原始表格、医生评分表及可重新识别患者的数据；
- 原始 EEG/EMG/IMU 数据、患者 ZIP、真实报告和实验输出；
- 原始论文/指南/教材 PDF、OCR 全文及扫描页；
- 模型权重、LoRA adapter、Qdrant 数据目录、数据库文件和运行缓存。

处理真实患者资料时遵循最小读取、最小复制和去标识化原则。对外输出只使用匿名病例编号。发现凭据已进入文件或 Git 历史时立即停止发布并报告，不要在聊天中复述凭据。

## 4. 上下文与事实来源

- `CODEZ_CONTEXT.md` 是历史交接资料，可能过时且可能含敏感部署信息；只用于定位线索，不得提交到公开仓库。
- 当前行为以真实代码、数据库 schema、运行进程、API 响应和测试结果为准。
- 核查服务器版本时同时确认进程 PID、命令行、`/proc/<pid>/cwd`、Python executable 和实际加载文件，不能只看 `current` 符号链接或 `/api/ready` 的版本文本。
- 不得根据文件名、目录名或旧时间戳推断“已部署”“已入库”或“已验收”。

## 5. 生产、候选与实验隔离

- 生产流程、候选 RAG 和大模型实验必须保持目录、配置、日志及输出隔离。
- 未经明确授权，不停止、重启或替换生产后端、Nginx、MySQL、Qdrant、RAG 服务或模型进程。
- 不直接改写服务器 `current` 指向的 release；新版本先进入独立 release 或实验目录并验收。
- 生产模型缓存策略不得为实验便利而修改。
- 六模型实验必须串行加载：生成并保存结果后释放 model/tokenizer，执行垃圾回收与 `torch.cuda.empty_cache()`，确认显存后再加载下一模型。
- 生产 Qwen 占用显存时不得直接运行正式六模型实验。停服实验流程必须由用户明确授权。

## 6. 受保护的核心逻辑

除非用户明确要求，不修改以下内容：

- 深度学习推理与模型结构；
- 26 项 biomarker 算法及字段口径；
- clinical contracts 和正式 pipeline 核心逻辑；
- knowledge graph 核心规则与医学关系；
- 生产默认模型、生产 RAG 默认版本和数据库迁移。

可以在明确授权下修改 RAG ingest、知识来源管理、chunk schema、来源展示、实验基础设施和测试代码，但必须保持既有接口兼容。

## 7. 临床与医学口径

- 当前实验口径区分 `FMA腕` 与 `FMA手`；不得擅自合并为完整 `FMA上肢`。
- `FMA_UE 0–20` 只能按项目实际手部任务解释，不得写成完整 FMA-UE 0–66。
- MAS、Brunnstrom、FMA、BI 等字段必须按代码中的实际名称、类型和合法范围使用。
- PAI、前额 theta/beta、半球间相干和 CMC 必须严格区分。
- CMC、mu/beta 功率变化等 deferred 指标不得升级为已验证机制证据。
- 不把相关关系写成确定因果，不自行创造疾病诊断、患者亚型或医学阈值。
- 未采集 ADL、认知、情绪、疼痛等数据时，不得作确定性判断。
- 教材证据保持 `source_type=textbook`、`evidence_level=textbook_reference`，不得伪装成研究证据。

## 8. OCR 与原始证据

- 原始 PDF 不修改、不覆盖；OCR 必须支持断点续跑并保留完整页面产物。
- `raw_ocr_text` 永久保留，任何自动纠错只写入独立 `corrected_text`。
- 页面数据应保留 `ocr_confidence`、`page_number`、`bbox`、`page_image` 和 `source_pdf`。
- 失败页、低质量页和空白页必须显式标记；不得为追求整库完成而伪造文本。
- 每个正式 chunk 必须可追溯：`chunk_id/uid -> source_file -> page_number/page_start/page_end -> original PDF`。
- 无真实页码的来源显示“页码不可用”，不得伪造页码。

## 9. RAG 版本与治理

- 旧生产 RAG 与新版候选 RAG 必须并存，未经授权不得覆盖或删除旧 collection。
- 新教材必须合入统一候选库，不另建互不兼容的教材 RAG。
- 所有 chunk 显式包含：`uid`、`source_file`、`page_start`、`page_end`、`source_type`、`evidence_level`、`knowledge_role`。
- `core_direct`、指南/共识、教材、training、core_background、method_reference 和 deferred 按现有治理规则分别准入，不得简单全部设为 `clinical_ready=true`。
- training 可服务训练建议，但不得充当 biomarker 机制证据。
- deferred 可保留在知识库，但不得进入正式临床解释。
- 向量构建必须核对合格 chunks、embeddings 和 Qdrant points 数量一致，并核对维度、距离度量和 collection 名称。
- 来源卡片必须显示标题、UID、页码和“查看原文”，并实际验证 HTTP 状态与 PDF 页面定位。

## 10. 大模型评测实验

- 正式流程与实验流程分开：`dl_prediction` 不受实验改动影响；`manual_clinical_scores` 只绕过临床评分预测，除非实验阶段明确禁用 biomarker。
- Stage1 临床基线输入只使用已锁定字段与固定 Prompt；不得临时增加患者信息、RAG、KG、DL 或 biomarker。
- RAG 与 KG 开关相互独立。关闭时不得检索、注入或显示虚假来源。
- 六个模型必须使用同一患者事实、Prompt 版本、报告 schema 和推理参数口径。
- 医生盲评页面不得显示模型名称、厂商、RAG/KG 状态或推理参数。
- Gold Reference 必须来自医生真实填写；复制参考答案只能用于流程连通性测试，不能用于正式统计结论。
- ROUGE/BLEU 是文本重合度指标，不等同于医学正确性。正式结论必须同时保留医生评分和事实错误标记。
- 未经明确授权，不启动真实患者乘六模型批量生成，不改 Gold，不覆盖既有实验输出。

## 11. 修改与验证流程

修改前：

1. 检查当前工作树、分支、远端和目标文件已有修改；
2. 找到真实入口、调用链、schema 和现有测试；
3. 明确是否触及生产、患者数据、核心算法或外部服务。

修改后按风险选择验证：

- Python：运行相关测试；完整后端基线可用 `PYTHONPATH=backend:. python3 -m pytest backend tests -q`；
- 前端：在 `frontend/` 运行 `npm run build`；
- RAG：运行候选库 validator、相关单元测试和真实检索验收；
- 文档和配置：至少运行 `git diff --check` 并核对路径、命令和版本名称；
- 部署：验证健康接口、真实进程路径、关键页面、来源跳转和回滚点。

不得把“导入成功”“mock 通过”或“文件存在”描述成真实模型生成、临床验收或生产部署完成。

## 12. 提交与汇报

- 提交前查看 staged diff，确认没有患者数据、凭据、PDF、权重、数据库、OCR 和运行输出。
- GitHub CI 通过后再合并主线；保留安全回滚点，不擅自删除旧 release、旧 collection 或发布分支。
- 汇报应包含实际修改文件、测试命令与结果、部署状态、未完成项和风险。
- 如果发现本次请求与现有医学口径或生产安全冲突，先给出具体证据，再请求用户决定。
