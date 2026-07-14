# RAG 第二步：BGE-M3 语义检索

本阶段把第一步生成的 `chunks.jsonl` 转换成 1024 维稠密向量，并保存到 Qdrant。它仍是独立实验，不调用报告大模型、不修改 Brunnstrom/FMA/MAS 结果，也不接入正式 JSON/PDF。

## 当前部署选择

- Embedding：`BAAI/bge-m3`，使用 Sentence Transformers，CPU 推理。
- 向量库：Qdrant Client 本地持久化模式。
- 数据范围：7 个 `demo_ready` 知识块。
- 临床状态：全部 `clinical_ready=false`，索引时必须显式使用 `--allow-demo`。
- 正式报告开关：`RAG_ENABLED=0`。

Qdrant 本地模式适合当前小规模教学和调试，数据保存在磁盘上，不监听端口。知识量、并发或服务数量增加后，再把 `RAG_BACKEND` 改成 `server`，连接只监听 `127.0.0.1:6333` 的独立 Qdrant 服务。

本地持久化目录同一时间只应由一个进程打开，不能直接供多个 FastAPI worker 共用。正式接入后端前必须迁移到独立 Qdrant Server，或把检索封装成单独的单进程服务。

## 1. 创建独立环境

不要把 RAG 依赖装进生产后端环境：

```bash
/root/miniconda3/bin/python -m venv /root/autodl-tmp/envs/rag_env
/root/autodl-tmp/envs/rag_env/bin/pip install \
  -r /root/autodl-tmp/rehab_project/Rehabilitation-Assessment-System-main/requirements-rag.txt
```

依赖文件会在这个独立环境中安装 `torch==2.6.0+cpu`。BGE-M3 官方权重为 PyTorch BIN 格式，Transformers 会因为 `CVE-2025-32434` 拒绝使用低于 2.6 的 PyTorch 加载。不要关闭这项安全检查，也不要升级生产后端共用的 PyTorch 2.1.2。

## 2. 下载 BGE-M3

只下载稠密检索需要的文件，跳过 ONNX、图片、稀疏和 ColBERT 附件：

```bash
HF_ENDPOINT=https://hf-mirror.com \
/root/autodl-tmp/envs/rag_env/bin/python scripts/rag_download_model.py \
  --endpoint https://hf-mirror.com
```

默认目录：

```text
/root/autodl-tmp/rag_models/BAAI/bge-m3
```

## 3. 建立索引

```bash
/root/autodl-tmp/envs/rag_env/bin/python scripts/rag_index.py --allow-demo
```

不加 `--allow-demo` 时，程序会拒绝把未通过专家审核的条目写入向量库。这是故意设置的临床治理门禁。

## 4. 语义检索

```bash
/root/autodl-tmp/envs/rag_env/bin/python scripts/rag_search.py \
  "患者肌肉疲劳时中位频率通常怎么变化？" --top-k 3
```

输出包含排名、余弦相似度、`knowledge_id`、原始片段和治理元数据。BGE-M3 官方模型不要求给查询添加特殊 instruction，因此查询保持自然中文即可。

## 5. 检索评测

```bash
/root/autodl-tmp/envs/rag_env/bin/python scripts/rag_eval_retrieval.py
```

- `Hit@1`：正确条目排在第一名的比例。
- `Hit@3`：正确条目出现在前三名的比例。
- `MRR`：正确条目排名倒数的平均值，越接近 1 越好。

初始验收集只有 7 道测试题，作用是验证工程链路，不足以证明临床检索质量。后续至少扩充到 50 至 100 道经过人工标注的问题，并加入同义表达、错误前提、跨指标问题和“知识库无答案”问题。

当前扩展评测还包含：

```text
knowledge_base/eval/semantic_queries_v0_1.jsonl   21个同义改写问题
knowledge_base/eval/no_answer_queries_v0_1.jsonl 11个知识库无答案问题
```

21 个改写问题的实测结果为 `Hit@1=0.8571`、`Hit@3=1.0`、`MRR=0.9286`。无答案测试发现，同属康复领域但知识库未覆盖的问题最高相似度达到 `0.6270`，因此不能仅靠固定 score 阈值决定是否回答。第三阶段必须加入 reranker、证据充分性判断和明确的拒答路径。

## 数据流

```text
chunks.jsonl
  -> BGE-M3编码知识片段
  -> Qdrant保存向量和metadata
  -> BGE-M3编码用户问题
  -> 余弦相似度Top-K
  -> 返回知识片段与来源状态
```

这一阶段只有召回，没有重排，也没有让 Qwen3 生成答案。完成检索评测后，第三步才加入 reranker 和证据包。

## 官方资料

- BGE-M3 模型说明：<https://huggingface.co/BAAI/bge-m3>
- Qdrant Local Quickstart：<https://qdrant.tech/documentation/quick-start/>
