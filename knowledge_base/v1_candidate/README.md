# rehab_knowledge_v1_candidate

这是新版统一知识库的代码与发布契约。当前已验收的运行时版本为
`v1_candidate_next`，管理员配置别名为 `v1_candidate`，对应 Qdrant collection：
`rehab_knowledge_v1_candidate_next`。

当前验收清单：

- sources：269
- chunks：15355
- embeddings：15355（bge-m3，1024维，Cosine）
- Qdrant points：15355
- 论文/指南分类保持原有 core_direct、core_background、training、method_reference
- U178/U179/U180/U181 已进入同一个候选知识库；U179 第510页按失败页策略排除
- `chunk -> uid -> source_file_id -> PDF -> page` 原文追溯契约保留在
  `source_view_contract.json`

本仓库只发布可审查的源码、脚本、schema 和小型发布清单。原始论文/指南/教材 PDF、
OCR全文、chunk全文和向量数据不进入 GitHub，需要在受控服务器上按
`source_file_manifest.jsonl` 单独部署。不要把患者数据、医生评分表、模型权重或实验输出
复制到本仓库。

构建与检查：

```bash
python3 knowledge_base/v1_candidate/scripts/build_v1_candidate.py --include-method-reference
python3 knowledge_base/v1_candidate/scripts/materialize_source_files.py
python3 knowledge_base/v1_candidate/scripts/validate_v1_candidate.py
python3 -m unittest discover -s knowledge_base/v1_candidate/tests -p 'test_*.py'
```

向量构建和 Qdrant 写入由受控运行环境执行，不能用空的本地清单冒充已完成 embedding。
`knowledge_base/rag_version_config.json` 负责把管理员可见的 `v1_candidate` 映射到实际运行时
目录；生产环境的来源文件库和 Qdrant collection 不由 Git checkout 自动创建。
