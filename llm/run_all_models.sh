#!/usr/bin/env bash
# 4-LLM × 3-fold sweep: build ChatML data → QLoRA SFT → generate → BLEU/ROUGE.
# After all 12 (model, fold) runs finish, pick the winner by all.char_bleu4.
#
# Hardware target: RTX 4090D 24 GB. End-to-end wall-clock ≈ 6 h.
#
# Override via env:
#   SUGG=patient_rehab_suggestions_100subjects.json   (default; 100-subject set)
#   MODELS="qwen25_3b mistral7b_v03"                  (subset)
#   FOLDS="1 2 3"
#   EPOCHS=3
#   RANK=16
#   DEEP_CLEAN=1                                       (drop HF cache after every fold,
#                                                       not just after every model;
#                                                       re-downloads on the next fold —
#                                                       useful for tight 25 GB disks)
set -euo pipefail

# libgomp requires a numeric value; unset or defaulting to "" causes the warning.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

SUGG=${SUGG:-patient_rehab_suggestions_100subjects.json}
MODELS=${MODELS:-"qwen25_3b mistral7b_v03 glm4_9b yi15_6b"}
FOLDS=${FOLDS:-"1 2 3"}
EPOCHS=${EPOCHS:-3}
RANK=${RANK:-16}
DEEP_CLEAN=${DEEP_CLEAN:-0}

if [ ! -f "$SUGG" ]; then
  echo "ERROR: suggestions file not found: $SUGG" >&2
  exit 1
fi

N=$(python -c "import json; print(json.load(open('$SUGG'))['n_subjects'])")
SPLIT="splits/3fold_patient_split_llm_${N}subjects.json"
echo "[run_all] suggestions=$SUGG  n_subjects=$N  split=$SPLIT"

# 1) Build the ChatML JSONL once per fold (model-agnostic).
for k in $FOLDS; do
  if [ ! -f "data/llm/fold${k}/train.jsonl" ]; then
    python -m src.llm.data_builder \
      --suggestions "$SUGG" --fold "$k" --out "data/llm/fold${k}"
  else
    echo "[run_all] data/llm/fold${k} already built, skipping"
  fi
done

# 2) For each model × fold: train → generate → evaluate.
for m in $MODELS; do
  for k in $FOLDS; do
    CKPT="checkpoints/llm/${m}/fold${k}"
    PRED="outputs/llm/${m}/fold${k}_test.json"
    REPORT="outputs/llm/${m}/eval_report_fold${k}.json"

    echo
    echo "================================================================"
    echo "[run_all] model=$m  fold=$k"
    echo "================================================================"

    if [ ! -f "${CKPT}/adapter_config.json" ]; then
      python -m src.llm.train_lora \
        --model-id "$m" \
        --train "data/llm/fold${k}/train.jsonl" \
        --val   "data/llm/fold${k}/val.jsonl" \
        --out   "$CKPT" \
        --epochs "$EPOCHS" --rank "$RANK"
    else
      echo "[run_all] adapter exists at $CKPT, skipping training"
    fi

    if [ ! -f "$PRED" ]; then
      python -m src.llm.generate \
        --model-id "$m" \
        --adapter "$CKPT" \
        --suggestions "$SUGG" \
        --split "$SPLIT" \
        --fold "$k" --partition test \
        --out "$PRED"
    else
      echo "[run_all] predictions exist at $PRED, skipping generation"
    fi

    if [ ! -f "$REPORT" ]; then
      python -m src.llm.evaluate --pred "$PRED" --out "$REPORT"
    else
      echo "[run_all] eval report exists at $REPORT, skipping evaluate"
    fi

    if [ "$DEEP_CLEAN" = "1" ]; then
      echo "[run_all] DEEP_CLEAN=1: dropping HF cache for $m after fold $k"
      python -m src.llm.clean_hf_cache --model-id "$m" || \
        echo "[run_all] WARN: per-fold HF cache cleanup failed, continuing"
    fi
  done

  # All folds for $m succeeded (set -e would have aborted otherwise).
  # Drop this model's HF hub cache to keep the 24 GB disk from filling up
  # before the next model downloads its base weights.
  echo "[run_all] all folds done for $m, cleaning HF cache"
  python -m src.llm.clean_hf_cache --model-id "$m" || \
    echo "[run_all] WARN: HF cache cleanup for $m failed, continuing"
done

# 3) Decide the winner.
echo
echo "================================================================"
echo "[run_all] selecting best model by all.char_bleu4"
echo "================================================================"
python -m src.llm.select_best_model \
  --reports-glob "outputs/llm/*/eval_report_fold*.json" \
  --metric all.char_bleu4 \
  --out outputs/llm/comparison/best_model.json
