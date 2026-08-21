# 康复大模型评测实验基础设施

本目录承载评测模式的版本化契约、固定 Prompt、报告渲染、批量输入、实验日志和医生盲评投影。它不改变生产评估默认行为，也不会在服务启动时调用模型、RAG或知识图谱；只有显式调用`generate_benchmark_report`时才会调用传入的模型/检索/图谱适配器。

## 入口与数据边界

- 正常入口仍是 `backend/main.py` 的 `/api/assess`：原始 EEG/EMG/IMU → biomarker → DL 临床评分 → 现有 LLM 报告。
- 手动入口仍是 `backend/main.py` 的 `/api/llm-control-test/sessions`：原始 EEG/EMG/IMU → biomarker → 医生输入的 FMA/MAS/Brunnstrom → 现有 LLM 报告。
- 评测准备包以 `BenchmarkRunConfig` 统一表达 `assessment_input_mode`、`rag_enabled`、`knowledge_graph_enabled` 和对应版本。默认是 `manual_clinical_scores + RAG OFF + KG OFF`。
- `build_evaluation_input` 只复制现有患者、临床评分、26 项 biomarker 和 QualityGate 结果，不计算、不覆盖任何临床事实。
- Benchmark实验字段统一为`fma_wrist`、`fma_hand`、`hand_mas`、`brunnstrom_hand`；旧的`FMA_UE`、`hand_tone`、`hand_function`只作为输入兼容别名，不会序列化到Benchmark Prompt。
- `batch.py`复用`eval_package.safe_extract_zip`和`read_eval_package`，严格匹配多个患者manifest与一个`clinical_scores.xlsx`。

## 固定输出

`rehab_llm_benchmark_v1.txt` 要求模型只返回三个顶层字段：

- `biomarker_interpretation`
- `integrated_assessment`
- `rehabilitation_plan`，每项为 `action / goal / reason / precaution`

`render_report` 由程序填写患者事实、四项临床评分、全部 biomarker 和数据质量，只插入上述三个模型字段；综合亚型不属于Benchmark输出。RAG开启时才渲染来源卡片；关闭时不会产生虚假来源。

## 日志与盲评

- `build_experiment_log` 保留 `raw_model_output`，解析结果单独放入 `parsed_model_output`。
- `storage.py` 以追加式 JSONL 写入，避免覆盖原始模型输出。
- `anonymous_report_id("CASE001", 1)` 生成 `CASE001-R01`。
- `build_blind_review_packet` 只暴露匿名编号、固定患者信息、报告和医生评分。
- `build_registry_mapping` 单独保存模型、RAG/KG 状态和版本，医生页面不读取该映射。

## 应用入口

- `POST /api/llm-benchmark/single/prepare`：单病例信号与医生真值准备；复用现有backend biomarker extractor，不调用模型。
- `POST /api/llm-benchmark/batch/prepare`：批量ZIP与`clinical_scores.xlsx`校验，逐患者复用现有biomarker extractor，并落盘`benchmark_runs/<batch_id>/`；不调用模型。
- 前端“LLM评测准备”页调用上述两个prepare入口；原“LLM对照测试”页仍是独立调试入口。

正式模型调用应由实验运行器显式提供`model_generate`、`retrieval_provider`和`graph_provider`，从而保证RAG OFF/KG OFF时对应Provider调用次数为零。生成结果可用`materialize_benchmark_generation`写入`patients/<patient_id>/reports/`及`logs/`。

## 当前不做的事

本阶段不启动 6 个模型，不执行正式横向比较，不执行 RAG/KG 消融，不改变 DL、26 项 biomarker、生产 Pipeline、生产知识图谱或生产默认入口。接入真实模型前，应由独立调用层明确选择 `BenchmarkRunConfig`，获得输入后再调用固定 Prompt，并把原始输出和解析结果分别写入日志。

## v2正式第一阶段

v1 三段式 Benchmark 保留为历史版本。正式第一阶段新增 `llm_benchmark.v2`：

- preset：`benchmark_stage1_v2`；
- Prompt：`prompts/rehab_llm_benchmark_v2.txt`；
- 生成字段只有 `integrated_assessment` 和 `rehabilitation_plan`；
- biomarker 继续由现有流程计算并作为单次辅助输入，但没有可靠参考范围时不得据绝对值下正常/异常或纵向结论；
- 不提供 `clinical_subtype`，不建立未经验证的亚型规则；
- v2 report、log、reference 和 metrics 均有独立版本标记，v1 不会被覆盖。

v2 入口模块位于 `llm_benchmark/v2/`，ROUGE/BLEU evaluator 位于 `llm_benchmark/evaluation/v2/`。本轮仍不调用真实模型、患者、RAG 或 KG。
