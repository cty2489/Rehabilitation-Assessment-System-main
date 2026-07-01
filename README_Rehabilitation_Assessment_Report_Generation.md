# CMK-AGN: Tri-Modal Clinical Rehabilitation Assessment + Report Generation

A tri-modal deep learning system that turns synchronized **EEG · EMG · IMU**
recordings into four clinical rehabilitation scores, and then hands those
scores to a QLoRA-fine-tuned LLM that drafts a Chinese rehabilitation report —
a single **Assessment → Report** clinical closed-loop.

> **CMK-AGN** is the model's public name. Internal code identifiers in
> `Deeplearning/` (the backbone class `ADKMDFANTriBackbone`, env vars
> `ADK_MDFAN_*`, filenames `adk_mdfan*.py`) are kept as-is because they are
> bound to the trained `.pth` checkpoints.

| Stage | What it does |
|---|---|
| **Sensing** | Synchronized EEG (.bdf) + EMG (.csv) + IMU streams per trial |
| **Alignment** | Weighted-Bayes DTW on EMG↔IMU + linear-resampled EEG (Tri-ADK-Knot) |
| **Assessment** | 4 independent CMK-AGN heads → FMA-UE / BI / Hand MAS / Brunnstrom |
| **Report** | 4-model QLoRA bake-off (Qwen2.5-3B / Mistral-7B / GLM-4-9B / Yi-1.5-6B) → auto-selected winner (**Yi-1.5-6B-Chat**) emits ~200-char Chinese rehab report |

---

## Clinical Tasks

Each task is trained as an **independent** model — no shared head, no joint loss.

| Task key | Clinical scale | Type | Range / Classes | Primary metric |
|---|---|---|---|---|
| `FMA_UE` | FMA-UE hand subscore | Regression | 0 – 20 (integer) | MAE / Rounded Acc (±1) |
| `BI` | Barthel Index | Regression | 0 – 100 (step 5) | MAE / Rounded Acc (±5) |
| `hand_tone` | **Hand MAS** (Modified Ashworth) | 6-class ordinal | `0, 1, 1+, 2, 3, 4` | Accuracy / Weighted κ |
| `hand_function` | **Brunnstrom stage** (hand) | 5-class ordinal | `2, 3, 4, 5, 6` | Accuracy / Weighted κ |

> Note: the code-level identifiers `hand_tone` / `hand_function` are kept for
> backwards compatibility with earlier manifests; clinically they correspond to
> Hand MAS and Brunnstrom hand-stage respectively.

---

## Project Structure

```
ADK-MDFAN/
├── simulate_data.py                  # Synthetic data generator (extends real → 100 patients)
├── samples_manifest_tri_4tasks_100subjects.csv
├── splits/
│   └── 3fold_patient_split_tri_4tasks_100subjects.json
├── BJH/                              # Raw EEG/EMG recordings (not committed)
├── bjh_labels.json                   # Per-patient clinical labels
├── patient_rehab_suggestions_100subjects.json
├── src/
│   ├── train.py                      # Single-task trainer entry point
│   ├── predict.py                    # Inference / batch prediction
│   ├── clinical_model.py             # Unified model wrapper (all 4 tasks)
│   ├── task_config.py                # Task specs: ranges, loss types, label encoders
│   ├── data_indexer_tri_modified.py  # Tri-modal sample indexer
│   ├── patient_splits.py             # K-fold patient splitter
│   ├── subject_aggregation.py        # Bag-of-trials → subject-level aggregation
│   ├── aggregate_ablation.py         # Cross-fold / cross-ablation summary
│   ├── alignment/
│   │   ├── wby_dtw.py                # Weighted-Bayes DTW
│   │   └── tri_strategies.py         # EEG/EMG/IMU tri-modal alignment
│   ├── bjh_io/
│   │   ├── bjh_loader.py             # Load EEG (.bdf) and EMG (.csv)
│   │   └── eeg_cache.py              # Disk-backed EEG preprocessing cache
│   ├── models/
│   │   ├── adk_mdfan_tri.py          # Tri-modal MDFAN backbone (main architecture)
│   │   └── adk_mdfan.py              # Bi-modal baseline
│   ├── baselines/                    # Baseline / ablation architectures
│   └── llm/
│       ├── README.md                 # LLM pipeline commands
│       ├── data_builder.py           # ChatML JSONL generation + 3-fold splits
│       ├── train_lora.py             # QLoRA SFT trainer (Qwen2.5-3B-Instruct)
│       ├── generate.py               # Inference with LoRA adapter
│       ├── evaluate.py               # BLEU / ROUGE evaluation
│       ├── compute_bertscore_zh.py   # Chinese BERTScore
│       ├── select_best_model.py      # Cross-model checkpoint selection
│       ├── model_registry.py         # Registered LLM backbones
│       ├── run_all_models.sh         # Sweep launcher
│       └── prompts.py                # System + user prompt templates
├── RESULT_newdata_baseline/          # Trained baseline checkpoints + logs
├── RESULT_newdata_CMK-AGN(Ours)/     # Our full-model results
├── RESULT_newdata_ablation/          # Ablation runs
├── outputs_llm_final/                # Generated reports + LLM eval results
├── figures/  Result_composite_figure/  analysis/
├── requirements.txt                  # DL pipeline dependencies
└── requirements-llm.txt              # LLM fine-tuning dependencies
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt          # EEG/EMG/IMU training pipeline
pip install -r requirements-llm.txt      # LLM QLoRA fine-tuning (optional)
```

### 2. Generate synthetic data

```bash
python simulate_data.py
```

Produces `samples_manifest_tri_4tasks_100subjects.csv` and the matching 3-fold
patient-disjoint split under `splits/`.

### 3. Train a single task

```bash
# Train FMA-UE on fold 1 only:
python src/train.py --task FMA_UE --fold 1

# Train all 3 folds (fold=0 means "all"):
python src/train.py --task FMA_UE --fold 0

# Custom output directory:
python src/train.py --task BI --fold 1 --out-dir experiments/BI/run1
```

**Default output layout** (`--out-dir` defaults to `RESULT_newdata_baseline/<task>/baseline/`):

```
RESULT_newdata_baseline/FMA_UE/baseline/
├── FMA_UE_fold1.pth                  # best model checkpoint
├── FMA_UE_fold1_logs/
│   ├── training_history.csv          # epoch-by-epoch metrics
│   ├── val_predictions.csv           # per-subject predictions
│   ├── metrics.json                  # final evaluation metrics
│   └── bland_altman_data.csv         # agreement analysis (regression)
├── FMA_UE_3fold_summary.{csv,json}   # cross-fold aggregation
└── config.json                       # full experiment config snapshot
```

### 4. Run inference

```bash
python src/predict.py \
    --task FMA_UE \
    --checkpoint RESULT_newdata_baseline/FMA_UE/baseline/FMA_UE_fold1.pth \
    --manifest samples_manifest_tri_4tasks_100subjects.csv
```

---

## LLM Pipeline (Assessment → Report)

Once the four clinical scores have been predicted, a QLoRA-fine-tuned LLM takes
`(demographics, FMA-UE, BI, Hand-MAS, Brunnstrom)` as input and emits a
~200-character Chinese rehabilitation assessment report. The submodule runs a
**4-model bake-off** with 3-fold cross-validation and automatically picks the
winner by `all.char_bleu4`, closing the **Assessment → Report** clinical loop.

All entry points live in [`src/llm/`](src/llm/) and are runnable as modules
(e.g. `python -m src.llm.train_lora ...`). Full details, ablations, threshold
discussion and trouble-shooting are in [`src/llm/README.md`](src/llm/README.md).

### Candidate models (QLoRA NF4 + LoRA r=16)

| `--model-id` | HF repo | Params | LoRA targets | `max_seq_length` |
|---|---|---|---|---|
| `qwen25_3b`     | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` | 3 B | Llama-style q/k/v/o + gate/up/down | 1024 |
| `mistral7b_v03` | `mistralai/Mistral-7B-Instruct-v0.3`   | 7 B | Llama-style                        | 1024 |
| `glm4_9b`       | `unsloth/GLM-4-9B-0414-bnb-4bit`       | 9 B | Llama-style (`Glm4ForCausalLM`)    | **768** |
| `yi15_6b`       | `01-ai/Yi-1.5-6B-Chat`                 | 6 B | Llama-style                        | 1024 |

Per-model hyper-params (`response_template`, `target_modules`, EOS overrides)
are declared in [`src/llm/model_registry.py`](src/llm/model_registry.py).
Tested on a single RTX 4090D (24 GB).

### Entry points (`src/llm/`)

| Module | Role |
|---|---|
| [`prompts.py`](src/llm/prompts.py) | `SYSTEM_PROMPT` + user template (shared by train and infer) |
| [`data_builder.py`](src/llm/data_builder.py) | `patient_rehab_suggestions_*.json` → ChatML JSONL + 3-fold subject-disjoint split |
| [`model_registry.py`](src/llm/model_registry.py) | Per-model `hf_id` / `response_template` / `target_modules` / `extra_eos_tokens` |
| [`train_lora.py`](src/llm/train_lora.py) | TRL `SFTTrainer` + PEFT QLoRA + bnb-4bit; switch base via `--model-id` |
| [`generate.py`](src/llm/generate.py) | Load base + adapter, batch generate by fold / partition / subject list |
| [`evaluate.py`](src/llm/evaluate.py) | sacrebleu (zh) + nltk (jieba) BLEU-1/2/3/4 + rouge-chinese ROUGE-1/2/L |
| [`compute_bertscore_zh.py`](src/llm/compute_bertscore_zh.py) | Chinese BERTScore (optional semantic metric) |
| [`select_best_model.py`](src/llm/select_best_model.py) | Aggregate `4 × 3` reports, rank by `all.char_bleu4`, emit `best_model.json` |
| [`run_all_models.sh`](src/llm/run_all_models.sh) | One-click `4 models × 3 folds` end-to-end sweep |
| [`clean_hf_cache.py`](src/llm/clean_hf_cache.py) | Free disk between folds when running on 25 GB cloud volumes |

### One-click reproduction

```bash
# 1) install LLM deps (assumes CUDA GPU, bitsandbytes 4-bit)
pip install -r requirements-llm.txt

# 2) run the full 4-model × 3-fold bake-off (~6 h on a single 4090D)
bash src/llm/run_all_models.sh

# 3) inspect the leaderboard / winner
cat outputs/llm/comparison/best_model.json
```

Environment knobs (subset; see `src/llm/README.md` §4):

```bash
SUGG=patient_rehab_suggestions_100subjects.json \
MODELS="qwen25_3b yi15_6b" \
FOLDS="1 2" \
EPOCHS=3 RANK=16 DEEP_CLEAN=1 \
bash src/llm/run_all_models.sh
```

### Single-model / single-fold walkthrough

```bash
# (1) Build fold-1 ChatML JSONL (model-agnostic, only needs to run once)
python -m src.llm.data_builder \
    --suggestions patient_rehab_suggestions_100subjects.json \
    --fold 1 --out data/llm/fold1

# (2) QLoRA SFT — switch base model via --model-id
python -m src.llm.train_lora \
    --model-id qwen25_3b \
    --train data/llm/fold1/train.jsonl \
    --val   data/llm/fold1/val.jsonl \
    --out   checkpoints/llm/qwen25_3b/fold1 \
    --epochs 3 --rank 16

# (3) Generate on the test partition (4-beam decoding by default)
python -m src.llm.generate \
    --model-id qwen25_3b \
    --adapter checkpoints/llm/qwen25_3b/fold1 \
    --suggestions patient_rehab_suggestions_100subjects.json \
    --split splits/3fold_patient_split_llm_100subjects.json \
    --fold 1 --partition test \
    --out outputs/llm/qwen25_3b/fold1_test.json

# (4) BLEU + ROUGE evaluation (per-group: all / real_S1_S5_only / synthetic_only)
python -m src.llm.evaluate \
    --pred outputs/llm/qwen25_3b/fold1_test.json \
    --out  outputs/llm/qwen25_3b/eval_report_fold1.json

# (5) Cross-model winner selection
python -m src.llm.select_best_model \
    --reports-glob "outputs/llm/*/eval_report_fold*.json" \
    --metric all.char_bleu4 \
    --out outputs/llm/comparison/best_model.json
```

### Outputs

```
data/llm/fold{1,2,3}/{train,val}.jsonl            # ChatML training data
splits/3fold_patient_split_llm_100subjects.json   # subject-disjoint LLM split
checkpoints/llm/<model_id>/fold{k}/               # LoRA adapter + tokenizer + model_id.txt
outputs/llm/<model_id>/fold{k}_test.json          # generated reports
outputs/llm/<model_id>/eval_report_fold{k}.json   # per-fold BLEU / ROUGE
outputs/llm/comparison/best_model.json            # winner + leaderboard
outputs_llm_final/                                # frozen final results
```

### Closing the loop end-to-end

Replace the LLM input's real labels with `src/predict.py`'s predictions to
chain raw EEG/EMG/IMU → 4 clinical scores → Chinese rehab report:

```bash
python src/predict.py --all-tasks                 # → predictions.json
# repackage predictions.json into the suggestions JSON schema (demographics +
# *_pred fields as labels), then feed it to:
python -m src.llm.generate \
    --model-id $(jq -r .best_model_id outputs/llm/comparison/best_model.json) \
    --adapter  checkpoints/llm/<winner>/fold<best> \
    --suggestions predictions_as_suggestions.json \
    --partition all --out outputs/llm/end2end.json
```

---

## Key Arguments (`src/train.py`)

| Argument | Default | Description |
|---|---|---|
| `--task` | *(required)* | `FMA_UE`, `BI`, `hand_tone`, `hand_function` |
| `--out-dir` | `RESULT_newdata_baseline/<task>/baseline/` | All outputs: checkpoints, logs, summary |
| `--checkpoint` | `None` | Override exact `.pth` path (supports `{fold}`) |
| `--fold` | `0` | Fold to train; `0` trains all folds |
| `--manifest` | `samples_manifest_tri_4tasks_100subjects.csv` | Trial-level sample manifest |
| `--split-json` | `splits/3fold_patient_split_tri_4tasks_100subjects.json` | Patient-disjoint 3-fold split |
| `--alignment-mode` | `adk` | `adk` = Tri-ADK-Knot (WBy-DTW on EMG↔IMU + EEG linear resample) |
| `--modalities` | `eeg+emg+imu` | Modality subset for ablation (e.g. `emg+imu`) |
| `--head` | `auto` | Classification head: `auto`/`ce`/`corn` |
| `--epochs` | `120` | Training epochs |
| `--patience` | `25` | Early stopping patience |
| `--lr` | `1e-4` | Learning rate |
| `--bag-size` | `4` | Trials per bag (MIL) |
| `--device` | auto | `cuda` / `cpu` |

---

## Citation

If you use this code, please cite the corresponding paper (link TBD).
