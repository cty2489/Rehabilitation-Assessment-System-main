# Stage 1 Clinical Baseline

`stage1_clinical_baseline` is the first-stage clinical-only benchmark path.
It reads the fixed workbook sheet `医生填写` and maps only columns B–J into
the medical Prompt. Column A is metadata; K is draft audit text; L is the only
source for approved Gold Reference and neither K nor L is sent to a model.

## Input

The Prompt contains exactly:

```text
sex, age, diagnosis, disease_course, paralysis_side,
fma_wrist, fma_hand, hand_mas, brunnstrom_hand
```

The path does not load signal packages and does not call biomarker, DL, RAG or
KG code. Model IDs are supplied through `Stage1RunConfig.model_ids`; no model
names are hard-coded.

## Commands

```bash
PYTHONPATH=backend python3 -m llm_benchmark.stage1.cli \
  validate-doctor-sheet \
  --excel 康复医生_极简完形填空表_32例.xlsx \
  --output doctor_progress.csv

PYTHONPATH=backend python3 -m llm_benchmark.stage1.cli \
  build-gold-references \
  --excel 康复医生_极简完形填空表_32例.xlsx \
  --output-root benchmark_references/gold_v2
```

Reference parsing is deterministic (`doctor_reference_parser_v1`). A row with
`【填写】`, missing assessment text, or an incomplete recommendation is not
`reference_ready` and is not written to Gold.

## Output

Stage 1 Candidates use the existing v2 report contract and therefore feed the
existing v2 ROUGE/BLEU evaluator. Their prompt version is
`rehab_llm_benchmark_stage1_v1`; this is distinct from the v2 prompt while
using the same v2 output/report contract. The generated payload contains exactly two
sections and exactly three recommendations. A model output with two or four
recommendations is retained and scored, with `plan_count_violation` recorded.

Each experiment writes `stage1_manifest.json`, Candidate JSON files under
`patients/<patient_id>/reports/`, and raw/parsed generation records under
`logs/experiment_stage1.jsonl`.
