# ROUGE + BLEU 自动评价模块 v1

本目录是离线评测工具，不被正式患者评估、DL、biomarker、RAG 或知识图谱路径调用。

## 评价对象

只读取已物化报告 JSON 中的 `parsed_model_output`：

```text
benchmark_runs/<batch_id>/patients/<patient_id>/reports/<model>.json
```

评分字段固定为：

```text
biomarker_interpretation
integrated_assessment
rehabilitation_plan
overall_generated_content
```

患者事实、临床原始评分、26项 biomarker、Markdown、来源卡片和 KG 元数据不进入评分。

## Gold Reference

每个患者一个 JSON，要求 `schema_version=rehab.llm-benchmark-reference.v1`、`review_status=approved`，并包含三个生成区域。测试 fixture 明确标记为 `TEST ONLY`，不能当作临床 Gold。

## 固定指标

- ROUGE-1/2/L：`rouge-score==0.1.2`，`ChineseMedicalTokenizer_v1`，不 stemming，保存 precision/recall/F1，主汇总为固定病例分母下的 per-case F1 macro mean。
- BLEU-4：`sacrebleu==2.6.0`，`tokenize=zh`，`lowercase=false`，`smooth_method=exp`，主结果为 corpus BLEU，报告 0--100 分数和 signature。
- 中文规范化：`medical_text_normalizer_v1`，只删除排版差异，不改医学语义。

## 固定分母

有效、已批准的 Gold 患者集合定义 `n_expected`。每个模型都使用同一集合。缺失 Candidate 保留病例，使用空文本并记 0 分；无效 Gold 不进入正式评价，也不记作模型 0 分。`valid-output-only` 不是主结果。

## CLI

```bash
python3 -m backend.llm_benchmark.evaluation.cli validate-reference \
  --reference-root <gold_root>

python3 -m backend.llm_benchmark.evaluation.cli score-case \
  --reference <gold.json> \
  --candidate <reports/model.json> \
  --patient-id P001 \
  --model-id ModelA \
  --output-root <evaluation_output>

python3 -m backend.llm_benchmark.evaluation.cli score-batch \
  --batch-root <benchmark_run> \
  --reference-root <gold_root> \
  --output-root <evaluation_output> \
  --model-id ModelA --model-id ModelB
```

输出包括 `per_case_metrics.csv/jsonl`、`per_model_summary.csv`、`corpus_bleu.json`、`evaluation_errors.jsonl` 和 `metric_manifest.json`。

## Benchmark v2（正式第一阶段方案）

v1 内容和 `synthetic_v1_cli_output/` 为历史版本，不能覆盖或混用。v2 使用：

```text
python3 -m backend.llm_benchmark.evaluation.v2.cli
```

v2 的正式评价 section 只有：

```text
integrated_assessment
rehabilitation_plan
overall_generated_content
```

`biomarker_interpretation` 和 `clinical_subtype` 不进入 v2 Gold、Candidate 评分或 overall。biomarker 仍作为单次采集的辅助输入，但没有可靠参考范围时不得据其绝对值做正常/异常判断；当前没有纵向数据，也不得产生纵向结论。

v2 Gold 必须使用 `schema_version=rehab.llm-benchmark-reference.v2`；Candidate 报告必须同时使用 `rehab.llm-benchmark-report.v2` 和 `rehab_llm_benchmark_v2`。v1 输入遇到 v2 evaluator 会返回 `VERSION_MISMATCH`，不会静默兼容。

v2 汇总字段明确区分：

```text
n_expected       有效approved Gold病例数
n_scored         实际固定分母评分数，正常等于n_expected
n_schema_valid   完全符合v2输出Schema的Candidate数
n_invalid        无法解析或严重Schema失败数
n_missing        完全没有Candidate文件数
generation_failure_rate = (n_invalid + n_missing) / n_expected
```

Stage1/v2 Gold和Candidate要求恰好3条动作；输出2条或4条仍参与文本评分，只记录 `plan_count_violation`，不改变固定分母。v2 输出写入新的评测目录，不修改 v1 结果。
