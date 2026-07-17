# RAG 内部试运行：结构化审阅 JSON 与 Assist 验收

本流程用于在专家正式签字前验证完整 RAG 工程链路。它允许未审核知识进入隔离的内部试运行报告，但不会把这些知识标记为已完成专家审核，也不能用于正式临床决策。

## 当前试运行基线

截至 2026-07-17，云端验证基线为：

| 项目 | 结果 |
|---|---:|
| 集合 | `rehab_knowledge_trial_v0_1` |
| 知识条目 / 切块 | 35 / 35 |
| 正式临床可用条目 | 0 |
| 可回答评测问题 | 70 |
| 知识库无答案/对抗问题 | 12 |
| 可回答问题 Hit@1 / Hit@3 / MRR | 1.0000 / 1.0000 / 1.0000 |
| 真实 Qwen3-8B Assist 冒烟 | 通过，37.518 秒 |
| 检索 / 实际引用 | 5 条 / 3 条，引用均在检索白名单内 |

上述召回指标只评价 70 个有标准答案的问题。12 个无答案问题目前只作为后续拒答器测试集，不能用这组 Hit@K 证明系统已经具备可靠拒答能力。

## 不能改变的边界

试运行源 JSON 必须保留：

```json
{
  "clinical_ready": false,
  "trial_release": {
    "expert_verified": false,
    "clinical_ready": false
  }
}
```

`--allow-internal-trial` 只是显式允许建立隔离的试验集合，不会把条目升级为临床知识。转换器会把 `expert_verified=false`、知识状态、来源 ID、原文件 SHA-256 和试运行版本写入每个切块。

## 1. 准备知识

私有 JSON 放在稳定数据目录，不提交 Git：

```bash
BASE=/root/autodl-tmp/rehab_project
APP=$BASE/current
RAG_PY=/root/autodl-tmp/envs/rag_env/bin/python
RAW=$BASE/knowledge_base/raw/rehab_knowledge_trial_v0.1.json
RUNTIME=$BASE/knowledge_base/runtime/rehab_knowledge_trial_v0_1

$RAG_PY $APP/scripts/rag_prepare_review_json.py \
  --input "$RAW" \
  --output-dir "$RUNTIME" \
  --collection rehab_knowledge_trial_v0_1 \
  --allow-internal-trial
```

输出包括 `entries.jsonl`、`chunks.jsonl`、`evaluation_queries.jsonl`、`quality_report.json` 和 `manifest.json`。检查 `quality_report.json` 中 `clinical_ready_entries` 必须仍为 `0`。

## 2. 建立独立索引并评测

Qdrant Local 同一时刻只能由一个进程打开。先停止 RAG 服务，再建立或替换试运行集合：

```bash
test ! -f "$BASE/rag_service.pid" || kill "$(cat "$BASE/rag_service.pid")" || true
rm -f "$BASE/rag_service.pid"

set -a
. "$BASE/rag.env"
set +a

$RAG_PY $APP/scripts/rag_index.py \
  --chunks "$RUNTIME/chunks.jsonl" \
  --collection rehab_knowledge_trial_v0_1 \
  --manifest-out "$RUNTIME/index_manifest.json" \
  --allow-demo

$RAG_PY $APP/scripts/rag_eval_retrieval.py \
  --queries "$RUNTIME/evaluation_queries.jsonl" \
  --collection rehab_knowledge_trial_v0_1 \
  --top-k 5 \
  --qdrant-path "$BASE/knowledge_base/vector_store/qdrant_local" \
  > "$RUNTIME/retrieval_eval.json"
```

把 `$BASE/rag.env` 的 `RAG_COLLECTION` 改为 `rehab_knowledge_trial_v0_1`，再启动服务并确认 `/health` 返回该集合名。

## 3. 先运行 Shadow

后端生产基线仍建议：

```env
RAG_MODE=shadow
RAG_SERVICE_URL=http://127.0.0.1:8010
RAG_SHADOW_INCLUDE_DEMO=1
RAG_ASSIST_APPROVED=0
RAG_ALLOW_DEMO_IN_PROMPT=0
```

Shadow 会记录命中 ID、分数、来源哈希和知识状态，但不会改变网页、JSON 或 PDF 报告。

## 4. 隔离验证 Assist

只在内部验证窗口临时设置：

```bash
export RAG_MODE=assist
export RAG_ASSIST_APPROVED=1
export RAG_ALLOW_DEMO_IN_PROMPT=1
export RAG_SHADOW_INCLUDE_DEMO=1

/root/autodl-tmp/envs/rehab_backend/bin/python \
  "$APP/scripts/rag_assist_smoke.py" \
  --output "$RUNTIME/assist_smoke_result.json"
```

冒烟脚本使用去标识化的合成患者，只在以下条件全部满足时返回成功：

1. 检索证据实际进入提示词。
2. 大模型返回非空 `rag_citations`。
3. 每个引用 ID 都属于本次检索结果，编造 ID 会触发重试或保守回退。
4. 最终 Markdown 明确显示“内部技术验证”和“未完成正式专家审核”。

若需要在网页中短期体验内部试运行 Assist，可把上述三个 Assist 变量写入 `backend/.env` 后重启后端。此状态只能用于内部演示；演示完成后应恢复 Shadow。

## 5. 回退

最快回退不需要删除索引：

```env
RAG_MODE=off
RAG_ASSIST_APPROVED=0
RAG_ALLOW_DEMO_IN_PROMPT=0
```

重启后端后，报告链路不再调用 RAG。若只想保留观测能力，将 `RAG_MODE` 改回 `shadow`。

## 6. 专家审核后的正式升级

不要修改旧试运行集合。应复制源 JSON，逐条补充专家决定、专家姓名、审核日期和意见；只有审核通过的条目才能同时设为 `clinical_ready=true`。随后使用新的版本号和集合名重新转换、索引和冻结评测结果。

正式 Assist 上线前还需要完成：无答案拒答器或证据充分性判断、固定病例集 `off`/`assist` 盲评、处方越界检查、数据保护审查和回滚演练。完整门禁见 [`RAG_GROUNDING.md`](RAG_GROUNDING.md)。
