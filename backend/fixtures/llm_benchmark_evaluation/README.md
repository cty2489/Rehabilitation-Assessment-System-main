# Synthetic evaluation fixtures

本目录仅用于自动评价模块测试。测试数据为虚构内容，标记为 **TEST ONLY**，不是临床 Gold Reference，不得用于患者报告或论文结果。

`test_llm_benchmark_evaluation.py` 会在临时目录中生成 3 个虚拟病例和 3 个虚拟模型：

- `ModelPerfect`
- `ModelPartial`
- `ModelFailed`

它们用于验证固定分母、缺失 Candidate、格式异常、计划动作数量异常以及多模型汇总。
