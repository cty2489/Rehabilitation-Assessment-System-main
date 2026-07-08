# LLM 微调子模块：报告模型 baseline + QLoRA 康复评估文本生成

把每位患者的 **(人口学信息 + 4 项临床指标)** 作为输入，由 LLM 生成约 200 字的中文康复评估与建议文本。本子模块用于对报告模型做 QLoRA 微调、3-fold 交叉验证和 baseline 对比，按 `evaluate.py` 给出的 `all.char_bleu4` 自动挑出最优模型作为后续上线候选。

> 训练与评估输入均为**真实标签**；DL 推理路径 ([src/predict.py](../predict.py)) 保持独立不受影响。

---

## 1. 候选模型对比

模型设置页与训练脚本共用同一套 `model_id`。当前网页默认只展示已准备/已验证的 HF 原版权重报告模型；训练实验仍可在注册表中保留更多研究用短名。进入 sweep 前，优先挑权重已准备且 chat template 验证通过的模型。

| model_id (短名) | HF 仓库 | 参数量 | 架构 / LoRA 注入点 | 显存 (bs=1) | max_seq_length |
|---|---|---|---|---|---|
| `qwen25_7b` | `Qwen/Qwen2.5-7B-Instruct` | 7 B | Llama-style: `q/k/v/o_proj` + `gate/up/down_proj` | ~14 GB cache / 4-bit 运行 | 1024 |
| `qwen3_8b` | `Qwen/Qwen3-8B` | 8 B | 同 Llama-style | 视本地权重格式而定 | 1024 |
| `deepseek_r1_distill_qwen7b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7 B | 同 Llama-style | 视本地权重格式而定 | 1024 |
| `baichuan2_7b_chat` | `baichuan-inc/Baichuan2-7B-Chat` | 7 B | Baichuan: `W_pack` + MLP/O projection | 视本地权重格式而定 | 1024 |
| `glm4_9b` | `unsloth/GLM-4-9B-0414-bnb-4bit` | 9 B | Llama-style (`Glm4ForCausalLM`, transformers ≥ 4.52) | ~19 GB | **768** |
| `mistral7b_v03` | `mistralai/Mistral-7B-Instruct-v0.3` | 7 B | 同 Llama-style | ~5 GB (NF4) | 1024 |
| `internlm3_8b` | `internlm/internlm3-8b-instruct` | 8 B | 同 Llama-style | 视本地权重格式而定 | 1024 |

保留研究用短名：

| model_id | 用途 |
|---|---|
| `qwen25_3b` | 小模型快速调试 |
| `llama3_8b_instruct` | 需 HF 授权和本地权重，暂不作为网页默认候选 |
| `yi15_6b` | 早期 Yi-1.5 对照实验 |

> **关于预量化变体**：`qwen25_3b` 与 `glm4_9b` 改用 Unsloth 维护的 bnb-NF4 预量化仓库（tokenizer / 模块名与原版一致，可直接接 PEFT LoRA），把 HF cache 占用从 ~6 GB / ~18 GB 压到 ~1.8 GB / ~5 GB，匹配 25 GB 云盘预算。其它 HF 权重如使用官方 fp16/bf16 仓库，建议靠运行时 bnb-4bit 量化压显存。

> **关于 Mistral-7B-Instruct-v0.3**：替换早期版本的 `deepseek_r1_distill_qwen_7b`——R1-Distill 是 think-then-answer 模型，会把 `max_new_tokens` 预算全部消耗在 chain-of-thought 上，无法在固定句法骨架的 SFT 场景下产出可对比的答案；Mistral-7B-Instruct-v0.3 (Apache 2.0、无需 HF gating，`LlamaForCausalLM` 架构、与现有 `_LLAMA_STYLE_TARGETS` 完全兼容) 作为英文/通用基线接入对比。
> **关于 GLM-4-9B**：9B 4-bit 显存占用接近 20 GB，必须把 `max_seq_length` 限制到 768；本数据集所有样本 token 长度 < 700，不会被截断。
> **关于 Yi-1.5 的 EOS**：Yi-1.5-Chat 的 `tokenizer.eos_token` 并不是 `<|im_end|>`，因此 `generate.py` 必须把 `<|im_end|>` 作为额外停止 token 传入；这一点通过 `MODEL_REGISTRY['yi15_6b']['extra_eos_tokens']` 声明，由 `generate.py:_resolve_eos_ids` 在 `model.generate(eos_token_id=...)` 中拼成多 EOS 列表。否则生成会跑到 `max_new_tokens` 并伪造下一轮 user/assistant，污染 BLEU/ROUGE。

完整模型配置见 [model_registry.py](model_registry.py)，包含每个模型族的 `response_template`（用于 `DataCollatorForCompletionOnlyLM` 屏蔽 prompt token 的特殊串）和 LoRA `target_modules`。

---

## 2. 文件结构

| 文件 | 作用 |
|---|---|
| [prompts.py](prompts.py) | SYSTEM_PROMPT 与 user 模板，统一训练/推理 |
| [data_builder.py](data_builder.py) | `patient_rehab_suggestions_*.json` → ChatML JSONL；自动生成 3-fold split |
| [model_registry.py](model_registry.py) | 候选模型的 `hf_id` / `response_template` / `target_modules` / `max_seq_length` |
| [train_lora.py](train_lora.py) | TRL SFTTrainer + PEFT QLoRA + bnb 4-bit；通过 `--model-id` 切换模型族 |
| [generate.py](generate.py) | 加载 base+adapter，按 fold/partition 或 subject 列表批量生成 |
| [evaluate.py](evaluate.py) | sacrebleu(zh) / nltk(jieba) BLEU-1/2/3/4 + rouge-chinese ROUGE-1/2/L |
| [select_best_model.py](select_best_model.py) | 聚合多模型 × 3 fold 的 eval 报告，按 `all.char_bleu4` 选最优 |
| [run_all_models.sh](run_all_models.sh) | 多模型 × 3 fold 一键全流程脚本 |

---

## 3. 环境

```bash
pip install -r ../../requirements-llm.txt
# CUDA 11.8 旧镜像请改 pip install bitsandbytes==0.43.3
```

> Apple Silicon / 纯 CPU 不支持 bitsandbytes 4-bit，请用云 GPU（推荐 RTX 4090D 24GB / A100 / L4）。

云服务器推荐配置：**GPU RTX 4090D (24GB) × 1, 16 vCPU Xeon 8352V, 60 GB RAM**。

---

## 4. 一键复现 baseline 对比（推荐）

```bash
# 数据文件已就位：patient_rehab_suggestions_100subjects.json
bash src/llm/run_all_models.sh
```

整条流水线包括：

1. **构建 ChatML JSONL**（3 个 fold，模型无关，只跑一次）
   - 输出：`data/llm/fold{1,2,3}/{train,val}.jsonl`
   - 同时在 `splits/3fold_patient_split_llm_100subjects.json` 写出 subject-level 划分（S1–S5 真实样本 round-robin 进 val_test）
2. **QLoRA 微调**：对 `MODELS` 中的 model_id × 3 fold 训练
   - 输出：`checkpoints/llm/<model_id>/fold{k}/`（含 LoRA adapter + tokenizer + `model_id.txt`）
3. **test 集生成**：每个 model_id × 3 fold 推理
   - 输出：`outputs/llm/<model_id>/fold{k}_test.json`
4. **BLEU / ROUGE 评估**
   - 输出：`outputs/llm/<model_id>/eval_report_fold{k}.{json,csv}`
5. **决胜**：按 `all.char_bleu4` 跨 fold 取均值，输出 leaderboard
   - 输出：`outputs/llm/comparison/{summary.csv, mean.csv, best_model.json}`

预期单卡 4090D 总耗时约 **6 小时**（3B≈12 min/fold、7B≈30 min/fold、9B≈50 min/fold）。中间任何一步崩了都可以重跑：脚本会检测已有 checkpoint / prediction / report 并跳过。

环境变量可控选项：

```bash
SUGG=patient_rehab_suggestions_100subjects.json \
MODELS="qwen25_7b qwen3_8b glm4_9b mistral7b_v03" \
FOLDS="1 2" \
EPOCHS=3 \
RANK=16 \
DEEP_CLEAN=1 \                 # 每折训练+评估完成立即清 HF cache（25 GB 云盘兜底）
bash src/llm/run_all_models.sh
```

---

## 5. 决胜机制：选出最终 LLM

```bash
python -m src.llm.select_best_model \
  --reports-glob "outputs/llm/*/eval_report_fold*.json" \
  --metric all.char_bleu4 \
  --out outputs/llm/comparison/best_model.json
```

控制台会打印类似的 leaderboard：

```
# LLM leaderboard (metric=all.char_bleu4)

| model_id | all.char_bleu4 mean ± std | all.rougeL_f mean | real_S1_S5_only.char_bleu4 mean |
|---|---|---|---|
| `qwen25_3b`     | **42.13** ± 2.31 | 61.84 | 23.55 |
| `mistral7b_v03` | **41.20** ± 2.10 | 60.50 | 22.80 |
| `glm4_9b`       | **45.62** ± 1.42 | 64.89 | 26.71 |
| `yi15_6b`       | **43.55** ± 2.07 | 62.40 | 24.85 |

→ winner: **glm4_9b** (unsloth/GLM-4-9B-0414-bnb-4bit)
```

`best_model.json` 内容示例：

```json
{
  "metric": "all.char_bleu4",
  "best_model_id": "glm4_9b",
  "best_hf_id": "unsloth/GLM-4-9B-0414-bnb-4bit",
  "best_all.char_bleu4_mean": 45.62,
  "best_all.char_bleu4_std": 1.42,
  "runner_up_id": "yi15_6b",
  "margin_over_runner_up": 2.07,
  "n_folds": 3
}
```

**选用其他指标**：把 `--metric` 改为 `real_S1_S5_only.char_bleu4`（更看重真实金标准）、`all.rougeL_f`（结构相似度）等任意 `<group>.<metric>` 组合即可。

---

## 6. 单模型 / 单 fold 走查（调试用）

下面以 `qwen25_3b` fold-1 为例（其它模型替换 `--model-id` 即可）：

```bash
# 1) 构建 fold-1 的 ChatML JSONL
python -m src.llm.data_builder \
  --suggestions patient_rehab_suggestions_100subjects.json \
  --fold 1 --out data/llm/fold1

# 2) QLoRA 训练（4090D ≈ 12 分钟 / 7B 模型 ≈ 30 分钟）
python -m src.llm.train_lora \
  --model-id qwen25_3b \
  --train data/llm/fold1/train.jsonl \
  --val   data/llm/fold1/val.jsonl \
  --out   checkpoints/llm/qwen25_3b/fold1 \
  --epochs 3 --rank 16

# 3) test 集生成
python -m src.llm.generate \
  --model-id qwen25_3b \
  --adapter checkpoints/llm/qwen25_3b/fold1 \
  --suggestions patient_rehab_suggestions_100subjects.json \
  --split splits/3fold_patient_split_llm_100subjects.json \
  --fold 1 --partition test \
  --out outputs/llm/qwen25_3b/fold1_test.json

# 4) BLEU + ROUGE 评估
python -m src.llm.evaluate \
  --pred outputs/llm/qwen25_3b/fold1_test.json \
  --out  outputs/llm/qwen25_3b/eval_report_fold1.json
```

> `--model-id` 写入 `model_id.txt` 到 adapter 目录，`generate.py` 之后可以不带 `--model-id`，自动从 adapter dir 解析 base 模型。

---

## 7. 评估指标

`evaluate.py` 输出 JSON 与 CSV，按三组分别报告：

| 组 | 含义 |
|---|---|
| `all` | 整个 test 集（**决胜指标默认取这一组的 char_bleu4**） |
| `real_S1_S5_only` | 仅 S1–S5 真实金标准（人写文本，最关键） |
| `synthetic_only` | 仅模板生成的合成样本（学习上限） |

每组的指标：

- `char_bleu{1,2,3,4}`：字符级 BLEU-1/2/3/4，nltk sentence_bleu + smoothing
- `sacrebleu_zh`：sacrebleu `tokenize="zh"` 的 corpus BLEU
- `word_bleu{1,2,3,4}`：jieba 分词后 nltk corpus_bleu
- `rouge1_f` / `rouge2_f` / `rougeL_f`：rouge-chinese F 值

> **达标线（模板化输出）**：`all` 组 `char_bleu4 ≥ 70` 且 `rougeL_f ≥ 70`（评估器以 0–100 标度报数，对应 0.7 阈值）。100 例 rehab_text 经 `data_builder.normalize_rehab_text` 统一成同一句法骨架（病程一律 `XX天`、首段对齐模板）后，`SYSTEM_PROMPT` 内置的 `REPORT_TEMPLATE` 充当生成约束，模型只需正确填槽即可命中阈值。
> 报告 JSON 末尾的 `meets_threshold.passed` 给出布尔结果；未达标只打 stderr 警告，不会中断 sweep。
> 真实 S1–S5 子组语言更自然，分数会略低，但只占 5%，对 `all` 组均值影响有限。

---

## 8. 排错

- **hyp 里出现 chat-template 残留**（如 `<|im_start|>`、`<|assistant|>`、`<｜Assistant｜>`）
  → 两种成因：(1) tokenizer 的 `eos_token` 不等于 chat 模板的回合边界 token（典型如 Yi-1.5-Chat 的 `<|im_end|>`），导致 `model.generate` 不停而伪造下一轮——在 `MODEL_REGISTRY[<id>]['extra_eos_tokens']` 里登记该 token，`generate.py` 会自动拼到 `eos_token_id` 列表里；(2) `skip_special_tokens=True` 没识别这些 token 为 special，`generate.py:_strip_trailing_chat_tags` 会做最后兜底裁剪。若 tokenizer 未内置 `chat_template`，可在 `MODEL_REGISTRY[<id>]['chat_template']` 中显式补充，Baichuan2 已按该方式兜底。
- **hyp 复读 user prompt / train_loss 不下降**
  → `response_template` 与 tokenizer 实际渲染不一致。SentencePiece tokenizer（Yi-1.5、Mistral 等）会因为前驱字节不同把 marker 切成不同的 token 序列，单纯字符串子串匹配可以通过但 `DataCollatorForCompletionOnlyLM` 内部按 id-子序列查找会失败，loss mask 错位，模型于是学会先复读 user 再作答。`train_lora._resolve_response_template_ids` 已改为在真实渲染序列里定位 token-id 子序列并直接把 id list 喂给 collator，任何漂移都会在训练前 fail-fast。
- **OOM**
  → 9B 模型务必 `--max-seq-length 768`（已是 GLM-4 注册表默认值）；其它模型可降 `--max-seq-length` 或 `--grad-accum`；如必要切 `--fp16`。
- **bitsandbytes 报 CUDA 版本错**
  → 改 pin `bitsandbytes==0.43.3`（CUDA 11.8）。
- **GLM-4 报 `trust_remote_code` 相关错**
  → GLM-4-0414 自 transformers 4.52 起改为标准 `Glm4ForCausalLM`，但 Unsloth 的 bnb-4bit 预量化仍打包了 modeling 文件，`trust_remote_code=True` 已在注册表里写死并向下传递。Yi-1.5 / Qwen2.5 / Mistral-7B-Instruct-v0.3 均为标准 LlamaForCausalLM，无需 remote code。
- **真实样本（S1–S5）误落入训练集**
  → `data_builder.make_split()` 已 round-robin S1–S5 到三个折的 val_test，可直接看 `splits/3fold_patient_split_llm_*.json` 验证。

---

## 9. 上线推理（可选）

用 DL 模型的预测代替真实标签：

```bash
python src/predict.py --all-tasks                 # 生成 predictions.json
# 把 predictions.json 中的 *_pred 字段当作 labels 字段
# 传给 generate.py 的 --suggestions（按格式包好 demographics 即可），
# --model-id 选用 best_model.json 里的 best_model_id。
```

不在本子模块默认交付范围。
