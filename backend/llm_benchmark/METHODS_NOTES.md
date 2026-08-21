# 康复大模型自动文本评价 v1（研究记录）

本文件仅记录评价方法，不代表最终论文 Methods。

## 评价文本

从现有 Benchmark 物化报告的 `parsed_model_output` 提取 `biomarker_interpretation`、`integrated_assessment` 和结构化 `rehabilitation_plan`。程序固定患者事实、临床评分、原始 biomarker、标题、模板标签、引用和知识图谱元数据均排除。整体文本按上述三个区域的固定顺序拼接，不加入章节标题或字段标签。

## Normalization 和 Tokenizer

使用 `medical_text_normalizer_v1`：Unicode NFKC、统一换行、删除 Markdown code fence、行首标题和纯项目符号、去除表格控制符、折叠空白。保留数字、单位、医学缩写、左右方向和否定词，不做术语纠错、同义词替换或改写。

ROUGE 使用纯 Python 的 `ChineseMedicalTokenizer_v1`：中文汉字逐字切分，连续拉丁医学词和数字结构保持整体，标点作为分隔符不计入 token。BLEU 主结果遵循锁定口径，使用 SacreBLEU `tokenize=zh`。

## ROUGE

计算 ROUGE-1、ROUGE-2、ROUGE-L 的 precision、recall、F1；关闭 stemming。每病例单独计算，模型主结果使用固定 Gold 病例集合上的 per-case F1 macro mean，同时保存标准差。

## BLEU

使用 SacreBLEU corpus BLEU-4，`tokenize=zh`、大小写敏感、`smooth_method=exp`、单一 Reference。病例顺序按 patient_id 锁定，不能用 sentence BLEU 的平均值替代 corpus BLEU。保存 SacreBLEU signature、版本、BP、系统/Reference长度和 n-gram precision。

## Gold 和失败处理

Gold 由康复专家形成并审批后版本化。无效 Gold 不参与正式分母。Candidate 解析失败或病例缺失时保留固定分母、使用空 Candidate、记录错误并记零分；额外字段只记日志，不评分；康复动作数量异常照常评分并记录 `plan_count_violation`。

---

# 康复大模型正式第一阶段 v2（两段式）

v1 的三段式结果和指标保持历史可复现，不被覆盖。正式第一阶段改用：

```text
rehab_llm_benchmark_v2
rehab_llm_metrics_v2
```

LLM 生成内容只有：

```text
integrated_assessment
rehabilitation_plan
```

整体文本只拼接这两部分。`biomarker_interpretation`、`clinical_subtype` 和其他额外章节不进入 v2 评价。

biomarker 仍由现有 EEG/EMG/IMU 流程计算并提供给模型作为客观辅助输入，但当前是单次采集。没有经过验证的参考范围、既往同患者结果或纵向数据时，模型不得根据绝对数值下正常/异常、轻重程度、阈值、预后或纵向变化结论。当前不设计 longitudinal schema，也不建立亚型规则。

v2 使用与 v1 相同的 ROUGE-1/2/L、ChineseMedicalTokenizer_v1、无 stemming、SacreBLEU 2.6.0 `tokenize=zh`、Corpus BLEU-4、`lowercase=false`、`smooth_method=exp` 和固定病例分母。v2 manifest 额外记录：

```text
prompt_version=rehab_llm_benchmark_v2
reference_schema_version=rehab.llm-benchmark-reference.v2
metrics_version=rehab_llm_metrics_v2
biomarker_policy
excluded_sections
n_scored
n_schema_valid
n_invalid
n_missing
```

v2 evaluator 拒绝 v1 report/reference 的版本混用；动作数量异常保留文本评分并记录 violation。真实专家 Gold、6模型实验和医生盲评不属于本轮。
