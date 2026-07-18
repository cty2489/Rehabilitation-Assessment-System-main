# Changelog

## cloud-server-v1.1.19 - 2026-07-18

- 新增 RAG 精确查找接口 `/v1/lookup`：26 项固定 biomarker 不再依赖 Top-K 向量召回，而是按唯一 `system_key` 一一匹配；重复键在入库阶段直接拒绝。
- 结构化知识切块完整保留 `proposed_claim`、允许/禁止解读、采集算法要求、参考范围策略和实现要求；Assist 报告逐项使用对应知识生成不同解读并附 `[KB-*]`，数据不足项仍不解析。
- 只有 26 项全部命中且通过治理门禁时才进入完整接地模式；部分命中继续要求大模型补足，Shadow 仍只记录轨迹、不改变报告。
- 完整接地模式下，大模型只生成不重复数值的定性摘要与高层策略，不再重复生成随后会被覆盖的 26 项 `marker_text`。Qwen3-8B 真实记录实测由 156.97 秒降至 27.84 秒，约减少 82%。
- Brunnstrom、FMA 和 MAS 数值前缀以及证据不足时的综合状态界定改由代码确定；一致性校验覆盖带空格写法，并新增 MAS 校验，修复模型把“Brunnstrom 6期”误写为“FMA 6分”仍可能漏检的问题。
- 完整接地报告会过滤依据单次 EMG/EEG/IMU 直接调整刺激、判断中枢驱动或关节活动度的策略；预警固定为试运行知识状态、单次设备指标边界和不适时停止训练三类可验证内容。
- `result.json` 新增 `knowledge_evidence`，包含知识条目、状态、来源 ID、审核状态和去重参考文献；PDF 同步增加知识索引与参考文献，网页、JSON 和 PDF 的来源信息保持一致。

云端验收使用 MySQL 评估记录的 26/26 实测 biomarker：精确查找 26/26 命中、解读 26 条互不重复、旧通用句出现 0 次、知识引用 26/26；最终 JSON 收录 28 个实际采用的知识条目和 28 条去重文献，PDF 为 10 页 A4，逐页渲染检查无重叠、裁切或缺字。后端 91 个 unittest、RAG 13 个 unittest 和 4 个服务函数测试全部通过；试运行知识仍为 `clinical_ready=false`，只允许内部技术验证。

## cloud-server-v1.1.18 - 2026-07-17

- 新增结构化审阅 JSON 入库器：强制保留来源哈希、知识状态、试运行版本和专家审核状态；未审核资料必须显式使用 `--allow-internal-trial`，且不会被提升为 `clinical_ready`。
- 检索评测支持一个问题对应多个正确知识 ID，并把可回答问题与无答案/对抗问题分开统计，避免用 Hit@K 冒充拒答能力。
- Assist 报告新增 `rag_citations` 强校验：至少引用一条本次检索证据，未知或编造 ID 会触发重试/保守回退；最终证据表只展示实际采用的来源及知识状态。
- 新增去标识化 `rag_assist_smoke.py`，用于真实本地大模型、检索、结构化报告、引用白名单和未审核警示的端到端验收。

云端内部试运行集合包含 35 条知识和 82 个评测问题。70 个可回答问题实测 `Hit@1=1.0`、`Hit@3=1.0`、`MRR=1.0`；12 个无答案问题单列且未计入召回指标。Qwen3-8B Assist 冒烟耗时 37.518 秒，检索 5 条、实际引用 3 条，引用及试运行警示均通过验证。生产后端随后恢复 Shadow 稳定配置。

## cloud-server-v1.1.17 - 2026-07-16

- 将受治理的 DOCX 知识入库、BGE-M3 向量化和 Qdrant 检索迁移到 v1.1.16 稳定基线，并新增只监听 `127.0.0.1:8010` 的独立 RAG 服务；Embedding 固定使用 CPU，不占用报告大模型 GPU 显存。
- 报告后端新增 `off`、`shadow`、`assist` 三种模式。当前生产只启用 `shadow`：检索结果写入权限为 `0600` 的去标识化轨迹，不进入提示词，不改变网页、JSON 或 PDF 报告。
- Demo 证据采用服务端 `RAG_ALLOW_DEMO` 与后端 `RAG_SHADOW_INCLUDE_DEMO` 双重开关；Assist 还要求显式审批，并默认过滤全部 `clinical_ready=false` 条目。
- 知识切块补齐原文 SHA-256、条目号、参考资料、专家与审核时间等来源元数据；提示词明确把知识正文视为不可信参考数据，防止文档中的命令性文字改变系统任务或安全边界。
- RAG 服务、索引或轨迹失败时采用 fail-open，原报告流程继续执行；Shadow 轨迹使用内部 `session_id` 关联数据库评估，不记录患者姓名、患者编号或知识正文。
- CI 扩展到 RAG 入库、批量检索、治理过滤、HTTP 服务和报告门禁测试；完整部署、回退与 Assist 上线条件见 `docs/RAG_GROUNDING.md`。

已在 RTX 4090D、MySQL 8.0.46、Qwen3-8B HF、BGE-M3 CPU 和 Qdrant Local 环境完成验证：21 个语义改写问题 `Hit@1=0.8571`、`Hit@3=1.0`、`MRR=0.9286`；6 个并发检索请求全部成功；使用 125 MB、6 trial 医院数据包完成评分、大模型报告、MySQL 落库及 JSON/PDF/ZIP 校验。Shadow 命中 6 条 Demo 证据但 `used_in_prompt=false`，验收数据与导出文件随后已清理。

## cloud-server-v1.1.16 - 2026-07-15

- 浏览器登录改为 HMAC 签名、短时有效的 HttpOnly Cookie；长期管理员 Bearer 和旧共享设备码默认关闭，仅保留显式迁移开关。
- 修正评分推理与训练的数据抽样、任务/试次嵌入逻辑；设备端结果明确标注为“仅完成工程链路验证”，不冒充临床验证。
- 新增信号采样率、时长、同步降级等质量记录；网页、JSON 和 PDF 显示报告 fallback、信号质控与验证状态。
- 评估、trial 和 biomarker 改为单事务落库；数据库保存失败不再向网页或设备端返回“完成”。患者主档优先，历史评估保留不可变患者快照。
- 设备任务幂等键按设备隔离，支持重启恢复、取消、失败码和完成状态自动校正；ZIP 解压和直传增加路径、体积、文件数、压缩比及磁盘保护。
- SSE 支持事件重放与断线续传；浏览器和设备评估共用可取消的单 GPU FIFO 队列。
- 启动脚本改为 PID 管理并等待 `/api/ready`；新增后端 pytest、前端构建和脚本语法 CI。
- 精简报告展示：明确“手部肌张力（MAS）”，移除无标准依据的参考范围列、单模态亚型和冗余治疗方法，只保留综合亚型与治疗策略。
- 修复 PDF 中英文、数字和单位混排重叠，部署模板增加可配置的 TrueType 拉丁字体路径。

已在 RTX 4090D、MySQL 8.0.46 和 Qwen3-8B HF 环境，使用设备组提供的数据包完成上传、FIFO 排队、深度模型评分、26 项 biomarker、大模型报告、重启恢复、JSON/PDF/ZIP 下载和幂等 ACK 验收。测试凭证、患者、任务与导出文件均在验收后清理。

## cloud-server-v1.1.15 - 2026-07-11

- 新增 MySQL `device_credentials` 安全凭证表：只保存 SHA-256 哈希和掩码，不保存可还原的设备码明文。
- 启动时把旧共享码及 `device_002`、`device_003` 一次性迁移为数据库凭证；数据库存在凭证后即成为唯一鉴权来源，网页停用/撤销不会被 `.env` 绕过。
- 新增管理员设备凭证接口：列表、生成、启用/停用、轮换和撤销；新码只在生成或轮换响应中返回一次。
- “系统管理”新增设备凭证管理界面，显示凭证掩码、状态、最近使用时间、任务数和管理操作。
- 被撤销设备码立即失效，但历史设备任务和评估结果继续保留。

## cloud-server-v1.1.14 - 2026-07-11

- 新增 `DEVICE_API_TOKENS_JSON` 多设备独立凭证配置，同时保留 `DEVICE_API_TOKEN` 兼容已有设备。
- 独立 token 自动绑定对应 `device_id`，只能上传、查询、下载和ACK本设备任务；跨设备访问或请求中的设备ID不一致返回 403。
- 新增纯逻辑鉴权测试，校验旧凭证兼容、独立凭证匹配、错误token、非法JSON和重复token配置。

## cloud-server-v1.1.13 - 2026-07-11

- 将原“仅大模型报告排队”升级为完整评估 FIFO 队列：网页和设备任务从信号处理、深度模型、biomarker、大模型报告到持久化全程串行，避免单卡 GPU 并发和 OOM。
- 设备任务响应升级为 `rehab.device_job.v1`，新增 `phase`、`queue_position`、`queue_ahead`、`progress_percent`、`poll_after_seconds`、结构化 `error` 和 `attempt_count`。
- 设备 ZIP、患者快照和任务阶段持久化到 MySQL/`DEVICE_JOB_ROOT`；服务重启后按创建顺序恢复 `queued/running` 任务，若评估已落库则直接修正为 `completed`。
- 上传接口支持 `Idempotency-Key`；同一次评估网络重传返回原 `job_id`，相同 key 对应不同 ZIP 返回 409。未提供 key 时，同设备/患者/assessment/package hash 的未失败任务也会去重。
- ACK 改为幂等操作；新增机器可校验的 `docs/schemas/device-job-v1.schema.json` 和完整设备对接说明。
- ACK 后清理已交付任务的持久化输入副本，避免设备长期运行耗尽磁盘；数据格式错误标记为不可重试，云端临时故障标记为可重试。
- 前端排队提示改为完整“评估任务排队”，不再错误声称评分已经完成。

## cloud-server-v1.1.12 - 2026-07-09

本次迭代在 v1.1.11 基础上：修复报告生成的静默降级隐患、统一 MySQL 存储层、新增报告并发队列，并补充手势库配置流程、BF16/离线/InternLM3 等接手说明。**接手须知集中在每条的「注意」里，末尾附各模型现状速查表。**

### 报告生成质量（backend/report.py, report_builder.py）

- **动态 token 预算，修复满负载静默降级**：`report.py` 新增 `_dynamic_report_max_new_tokens(context, cfg)`，按可用 biomarker 数动态设 `max_new_tokens`（区间 2048–4096），在 `_reason_local` 非分段路径生效，替代原固定 1536。
  - 为什么：实测 qwen3 单次生成真实 26-marker 报告需 2400–2700 token；固定 1536 会截断 → JSON 解析失败 → 静默回退保守模板，而使用者以为是模型真实分析。医院端真实数据 coverage 就是 26/26，必然触发。
  - 注意：显式 `LLM_MAX_NEW_TOKENS` 环境变量、或模型 cfg 的 `max_new_tokens`（分段模型都设了）仍优先——helper 检测到就返回 None 不覆盖。
- **fallback 显式标注**：`_fallback_clinical` 的 `overall_interpretation` 加前缀「⚠️ 本报告由保守规则后备生成，非大模型结构化分析」，便于区分真实分析与降级模板。

### 手势库（backend/config/gestures_26.example.json, gestures.py, report_builder.py）

- **提供 26 手势库示例，但默认不启用处方**：新增 `backend/config/gestures_26.example.json` 和 `backend/config/README.md`，作为临床团队审核 schema 与候选动作的起点；`backend/config/gestures_26.json` 被加入 `.gitignore`，只有临床确认后手动复制为该运行态文件，`library_ready()` 才会变 True。
  - 注意：example/seed **不是临床确认库**，不要直接当作正式处方库提交或启用。未启用时报告第四节保持“手势库待补充”占位，不要求大模型生成具体手势。
- **手势校验硬要求改软要求（关键）**：`report_builder.validate_clinical` 原本在库 ready 时**硬要求** ≥6 手势 + 7 天计划，产不出就 `raise` → 整份报告 fallback。改为**软要求**：产出合规手势就用，产不出就只跳过手势段（`gesture_skipped=True`，渲染说明"该模型未生成手势，建议用 Qwen3/InternLM3/Mistral 重新生成"），保留报告其余内容。
  - 为什么：分段模型（baichuan2/deepseek/glm）的分段路径根本不产手势，硬要求会让它们全部 fallback。

### 报告模型（llm/model_registry.py）

- **分段粒度调优**：`deepseek_r1_distill_qwen7b` 与 `glm4_9b` 的 `segment_marker_chunk_size` 由 5 降到 3，缓解满负载下分段 JSON 失败（baichuan2 每段 1 marker 本就稳）。
- **单次生成实验结论（勿重复踩坑）**：本次实测把 deepseek/glm 改单次（去 `generation_mode`，像 qwen3）**均失败**——deepseek 漏 marker、glm 拼不成 JSON、baichuan2 上下文仅 4096 装不下完整 prompt，已回滚。**结论：这三个模型必须分段，勿再改单次。** 能单次的只有通用指令模型 qwen3 / internlm3 / mistral。

### 数据层（backend/main.py, db.py）

- **双存储统一到 MySQL**：删除失效的 SQLite 分支——`main.py` 移除 `import db` / `db.init_db()` / `_worker` 的 SQLite else 分支；`SessionState.persist_target` 默认由 `"sqlite"` 改 `"mysql"`；`backend/db.py` 从代码库移除。
  - 为什么：所有评估流早已传 `persist_target="mysql"`，SQLite 分支从不触发、`rehab.db` 为空。云端部署从此要求 MySQL 正常运行；若 MySQL 不可用，业务 API 返回明确 503。

### 并发（backend/main.py + frontend）

- **报告生成全局队列**：`main.py` 新增模块级 `_report_gate`(Lock) + `_report_depth`(排队+在途计数)。`_worker` 生成报告前入队，report 一次只跑一个，避免多用户同时评估抢单块 GPU 导致互相拖慢/OOM。新增 SSE 事件 `{"type":"report_queued","ahead":N}`。
- **前端排队进度**：`types.ts` 加 `report_queued` 类型；`useAssessmentStream.ts` 加 `queueAhead`；`AssessmentPage.tsx` 与 `TaskInterfacePage.tsx` 显示「报告排队中，前面还有 N 份」黄色横幅；`styles.css` 加 `.queue-banner`。
  - 注意：**AssessmentPage 未用 `useAssessmentStream` hook，而是内联了自己的一套 SSE 处理**——两处都已改，但这俩重复逻辑是已知技术债，未来应合并到 hook。

### 性能与环境（本次会话补记）

- **BF16 替代 4bit**：云端 `backend/.env` 设 `LLM_LOAD_4BIT=0`。24GB 单卡显存充足（此前仅用 8.3GB），4bit 反而拖慢。实测 qwen3 生成 26-marker 报告 **227s → 104s（提速约 2.2 倍）**；显存 16.4GB/24GB。
- **强制离线**：云端 `.env` 加 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`。服务器无外网，不设会在加载本地模型时联网校验卡死。
- **接入 InternLM3-8B**：`model_registry.py` 加 `internlm3_8b`（ChatML、`trust_remote_code=True`、`extra_eos_tokens=["<|im_end|>"]`——其 tokenizer eos 是 `</s>` 但轮次结束靠 `<|im_end|>`）。已验证单次生成通过端到端校验。

### 各报告模型现状速查（给接手者）

| 模型 | 生成方式 | 满负载 26-marker | 备注 |
|---|---|---|---|
| qwen3_8b_hf | 单次 | ✅ 稳定 ~104s | 当前默认，推荐 |
| internlm3_8b | 单次 | ✅ 稳定 ~100s | |
| mistral7b_v03 | 单次 | ✅ 稳定 ~84s | 国外 baseline |
| baichuan2_7b_chat | 分段 | ✅ 出报告（无手势段） | 文本偏模板化 |
| deepseek_r1_distill_qwen7b | 分段 | ⚠️ 偶失败→fallback | 推理模型 |
| glm4_9b | 分段 | ⚠️ 偶失败→fallback | |
| qwen25_7b_gguf | GGUF 回退 | 手动启 start_gguf_fallback.sh | 已从页面候选移除 |

> 部署环境：单卡 RTX 4090D 24GB、无外网。BF16 主力模型约占 16GB，与 GGUF 回退服务(~5GB)可勉强共存(~21GB)，但切到 GLM-4-9B(18.8GB)+GGUF 会 OOM，注意别同时常驻。

## cloud-server-v1.1.11 - 2026-07-08

### 改进

- 生产启动链路默认不再启动 GGUF 服务，`start_rehab_system.sh` 只负责 MySQL、FastAPI、前端生产包和 Nginx。
- 新增 `start_gguf_fallback.sh`，需要临时回退/对照时可手动启动 Qwen2.5-7B-Instruct GGUF 服务，默认端口为 `127.0.0.1:6008`。
- “模型设置”默认候选列表移除 `qwen25_7b_gguf` 和未准备的 `llama3_8b_instruct`，避免网页端出现不该切换的模型。
- 默认报告模型保持 `qwen3_8b_hf`；HF 原版权重候选保留 Qwen3、DeepSeek、Baichuan2、GLM、Mistral 和 InternLM3，便于后续 baseline、知识库增强和微调实验。
- 旧运行态配置若保存过 `qwen25_7b_gguf` 或 `llama3_8b_instruct`，后端读取时会自动过滤并回落到 `qwen3_8b_hf`。
- `README.md` 与 `backend/.env.example` 更新为 HF 本地模型优先的云端部署方案。

## cloud-server-v1.1.10 - 2026-07-07

### 改进

- `mistral7b_v03` 已通过真实 26 biomarker 报告 JSON 结构校验，可在“模型设置”页切换为国外 baseline 对照。
- `baichuan2_7b_chat` 接入更保守的单 biomarker 分段结构化报告生成，修复分号数组分隔、单字符串 marker 解读、未闭合单 marker 字符串和复制输入行等低质量输出的结构兼容问题；已通过真实 `mysql_assessment_33`、VI 期、26 biomarker 报告 JSON 结构校验。
- Mistral 真实输出中出现的 `treatation_advice` 拼写漂移会规范化为 `treatment_advice`，治疗策略对象列表也会规范化为报告可渲染的字符串列表。
- `glm4_9b` 增加模型专属 `repetition_penalty=1.2`，避免 summary 分段在当前真实样例上重复输出 `group_subtypes/overall_subtype` 而不生成治疗策略。
- 保持 `qwen3_8b_hf` 为默认线上报告模型；Mistral/GLM 可作为 baseline 对照，Baichuan2 仅作为低阶国产 baseline 开放，当前临床文本存在模板化、占位化和复制输入行倾向，不推荐作为默认报告模型。

## cloud-server-v1.1.9 - 2026-07-07

### 改进

- `glm4_9b` 接入通用分段结构化报告生成，避免 GLM-4-9B-Chat 一次性输出 26 biomarker 大 JSON 时截断。
- 分段 JSON 解析新增 GLM 近似 JSON 修复：可从 `["解读","建议"]"]` 这类多余括号/重复代码块输出中提取 required marker key，并继续交由最终临床 schema 严格校验。
- GLM-4-9B-Chat 已在云服务器真实 `mysql_assessment_27`、VI 期、26 biomarker 数据上通过端到端报告 JSON 结构校验，可在“模型设置”页切换为 baseline 对照。
- 当前默认报告模型仍推荐 `qwen3_8b_hf`；GLM 原版权重生成约 4 分钟/份，临床文本偏模板化，仍需后续知识库增强或微调优化。

## cloud-server-v1.1.8 - 2026-07-07

### 改进

- `deepseek_r1_distill_qwen7b` 采用分段结构化生成：关闭 R1 `<think>` 输出后，将 26 项 biomarker 拆成小段生成，再合并为统一临床 JSON。
- 新增 DeepSeek 分段 JSON 修复与校验逻辑：支持紧凑 `[解读, 建议]` marker 输出、长 key 顺序映射、单段缺失外层闭括号修复，并保持最终 `report_builder.validate_clinical` 严格校验。
- DeepSeek-R1-Distill-Qwen-7B 已在云服务器真实 `mysql_assessment_21`、VI 期、26 biomarker 数据上通过端到端报告 JSON 结构校验，可在“模型设置”页切换为 baseline 对照。
- 当前默认报告模型仍推荐 `qwen3_8b_hf`；DeepSeek 原版权重生成约 2 分钟/份，临床文本质量仍需后续知识库增强或微调优化。

## cloud-server-v1.1.7 - 2026-07-07

### 改进

- 停止补下载 Mistral，按当前服务器空间策略删除 `/root/autodl-tmp/Qwen_data/Mistral-7B-Instruct-v0.3`，释放约 13G 数据盘空间。
- 报告 JSON 解析器改为读取第一个完整的顶层临床 JSON 对象，避免 GLM 等模型重复输出代码块或输出截断时误解析内层 biomarker 小对象。
- 新增 Mistral tokenizer 慢速加载配置；但当前云服务器 Mistral 目录缺少 tokenizer 文件且已删除，因此仍不作为可切换模型。
- 验证结论更新：GLM-4-9B-Chat 能加载，小样例 JSON 可通过；但真实 26 biomarker 报告在当前 1536 token 预算内输出截断，暂不开放线上切换。Baichuan2 当前受 PyTorch 2.6 `torch.load` 安全限制影响，暂不能加载。

## cloud-server-v1.1.6 - 2026-07-07

### 改进

- 大模型候选路径识别扩展到 `/root/autodl-tmp/Qwen_data` 下的 Baichuan2、GLM-4、Mistral 和 Llama 候选目录。
- 运行态配置中保存过的旧本地权重路径如果已不存在，而当前默认候选路径已存在，后端会自动采用新的有效路径。
- 新下载的候选模型仍默认保持“候选待验证”，只有端到端报告 JSON 结构验证通过后才允许切为线上报告模型。

## cloud-server-v1.1.5 - 2026-07-07

### 改进

- “模型设置”改为独立页面，仅用于切换已验证的报告大模型；本地权重路径、远程服务地址和 adapter 目录不再暴露在业务页面。
- “系统管理”页恢复为账户与服务状态管理，不再与“模型设置”重复。
- `deepseek_r1_distill_qwen7b` 标记为候选待验证：权重可存在，但因端到端报告 JSON 结构未通过校验，暂不允许切为线上报告模型。
- 当前线上报告模型恢复为 `qwen3_8b_hf`，避免误切换到未验证候选模型导致报告慢或结构失败。
- BI/改良 Barthel 指数从当前上肢手功能在线推理、页面结果卡、统计、入组表单和导出报告中移除；数据库字段仅保留旧记录兼容。
- 在线推理不再加载 BI 评分模型，减少启动和分析阶段的模型工作量。

## cloud-server-v1.1.4 - 2026-07-06

### 改进

- 左侧菜单新增“模型设置”入口，直接定位到大模型选择区域。
- 右上角用户菜单新增“模型设置”入口。
- 保留“系统管理”页面原有账户、系统状态和大模型设置功能。

## cloud-server-v1.1.3 - 2026-07-06

### 改进

- 当前云端报告模型从 `qwen25_7b_gguf` 切换为 `qwen3_8b_hf`，后者已通过端到端报告链路测试。
- 本地 HF 报告模型生成时会尽量关闭 Qwen3/DeepSeek-R1 风格的 thinking 输出，并在解析 JSON 前剥离 `<think>...</think>` 或孤立 `</think>` 前缀。
- 文档补充 HF 原版权重推理依赖的已验证版本组合，避免部署时误升级服务器现有 PyTorch/CUDA 环境。
- 明确 `deepseek_r1_distill_qwen7b` 当前仅作为候选对照：权重可加载、可生成，但报告 JSON 结构尚不稳定，不建议设为默认。
- `qwen25_7b_gguf` 保留为可用回退/对照服务。

## cloud-server-v1.1.2 - 2026-07-06

### 改进

- Qwen3-8B 和 DeepSeek-R1-Distill-Qwen-7B 默认优先识别 `/root/autodl-tmp/Qwen_data` 下的原版 HF 权重目录。
- 新增 `LLM_ORIGINAL_MODEL_ROOT` 配置项，用于存放后续 baseline 和微调使用的原版模型。
- 文档明确区分：GGUF 用于当前远程报告服务，HF 原版目录用于模型对比和后续微调。

## cloud-server-v1.1.1 - 2026-07-06

### 改进

- 系统管理页支持直接保存每个报告模型的本地权重路径或远程服务地址。
- Qwen3-8B 自动识别服务器已有路径 `/root/autodl-tmp/Qwen_data/Qwen3-8B`。
- 未放置权重或服务不可用的模型会显示为未就绪，后端拒绝将其设为当前报告模型。
- `llm/model_registry.py` 补齐 7 个 baseline 候选模型短名，训练、生成和页面配置使用同一套 `model_id`。
- `llm/README.md` 更新为 7 候选 baseline 说明。

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
