# RAG 第一步：康复知识入库前处理

当前阶段只完成知识治理和结构化，不加载 Embedding 模型，也不改变现有评估、报告或设备接口。

## 为什么先做这一步

RAG 不是“把 Word 扔给大模型”。文档必须先变成稳定、可追踪、可审核的知识记录：

```text
私有 Word 原文
  -> 解析条目、图片和表格
  -> 补充稳定 knowledge_id
  -> 标记来源与专家审核状态
  -> 生成 entries.jsonl
  -> 生成待向量化 chunks.jsonl
  -> 通过质量门禁
```

`demo_ready` 只表示条目可用于技术检索实验；`clinical_ready` 表示来源和专家审核均完成。两者不能混用。

## 目录约定

```text
rag/ingest/                  DOCX 解析与结构化代码
knowledge_base/config/      可提交的字段映射、人工转录和治理配置
knowledge_base/eval/        可提交的检索测试问题
knowledge_base/raw/         私有原文，不提交 Git
knowledge_base/runtime/     生成数据和图片，不提交 Git
scripts/rag_prepare_knowledge.py
scripts/rag_prepare_review_json.py
scripts/rag_verify_knowledge.py
```

## 生成第一批知识

```bash
mkdir -p knowledge_base/raw
cp /path/to/康复知识条目结构化整理.docx knowledge_base/raw/

python scripts/rag_prepare_knowledge.py \
  --input knowledge_base/raw/康复知识条目结构化整理.docx
```

默认输出到：

```text
knowledge_base/runtime/rehab_knowledge_demo_v0_1/
├── entries.jsonl
├── chunks.jsonl
├── manifest.json
├── quality_report.json
└── assets/
```

`manifest.json` 保存原文 SHA-256。原文发生变化后，哈希也会变化，因此可以追溯报告使用了哪一份知识源。

如果专家沟通稿已经整理为包含 `sources`、`entries`、`evaluation_questions` 和 `trial_release` 的结构化 JSON，使用 `scripts/rag_prepare_review_json.py`。未审核资料必须显式传入 `--allow-internal-trial`，并使用独立集合名；转换器不会因此把 `expert_verified` 或 `clinical_ready` 改为真。完整命令见 [`RAG_TRIAL_ASSIST.md`](RAG_TRIAL_ASSIST.md)。

## 验证

```bash
python scripts/rag_verify_knowledge.py \
  --knowledge-dir knowledge_base/runtime/rehab_knowledge_demo_v0_1
```

当前验证使用确定性的术语匹配，只检查“解析和治理是否正确”，不是最终的向量检索效果。下一阶段才会加入 BGE-M3、Qdrant、Top-K 召回和重排评测。

## 当前资料的发布边界

- 允许：RAG 教学、切块实验、检索 Demo、内部技术验证。
- 不允许：作为正式临床证据、自动诊断依据或未经审核的治疗处方来源。
- 进入正式报告前必须补齐参考来源、版本、页码或章节、审核专家和审核日期。
- EMG、IMU、EEG 条目的算法、单位和解释条件必须与系统真实计算代码逐项对齐。
