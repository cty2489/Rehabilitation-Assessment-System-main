"""BERTScore-zh evaluation for generated Chinese rehab text (Figure 17, §4.6.5).

Reads each LLM's per-fold prediction JSON (subject_id / source / hyp / ref) and
produces three artifacts so we can both draw a per-sample violin plot and a
per-group summary table:

  1) {model_dir}/bertscore_fold{k}.csv  - per-sample P / R / F1
  2) {model_dir}/bertscore_summary.csv  - per-group (all / real / synthetic)
                                          mean +/- std across all available folds
  3) outputs_llm_final/llm_result/comparison/bertscore_mean.csv
                                        - cross-model mean +/- std for Table 8

Encoder: hfl/chinese-roberta-wwm-ext (HFL whole-word-masked RoBERTa, 12 layers).
Layer 8 is used (BERTScore paper recommends a mid-upper layer; bert-score's
default for unsupported models is the last layer, which over-smooths).
Baseline rescaling is disabled because hfl/chinese-roberta-wwm-ext has no
official baseline file shipped with bert-score; the paper text must disclose
that raw F1 (typically 0.6-0.9 for Chinese text) is reported.

Examples:
    # Single model, one fold (smoke test)
    python -m src.llm.compute_bertscore_zh \
        --model-dir outputs_llm_final/llm_result/yi15_6b --folds 1

    # All 4 models, all folds
    python -m src.llm.compute_bertscore_zh --all
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_MODELS = ("yi15_6b", "glm4_9b", "qwen25_3b", "mistral7b_v03")
REAL_SUBJECT_IDS = {"1", "2", "3", "4", "5"}
ENCODER = "hfl/chinese-roberta-wwm-ext"
NUM_LAYERS = 8


@dataclass
class SampleScore:
    subject_id: str
    source: str
    group: str
    p: float
    r: float
    f1: float


def _select_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _classify_group(row: Dict[str, object]) -> str:
    sid = str(row.get("subject_id"))
    if row.get("source") == "real" or sid in REAL_SUBJECT_IDS:
        return "real_S1_S5_only"
    return "synthetic_only"


def _load_fold(path: Path) -> List[Dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"Empty or malformed prediction file: {path}")
    return rows


def score_fold(
    rows: Sequence[Dict[str, object]],
    device: str,
    batch_size: int,
) -> List[SampleScore]:
    from bert_score import score as bertscore

    hyps = [str(r["hyp"]) for r in rows]
    refs = [str(r["ref"]) for r in rows]
    P, R, F1 = bertscore(
        cands=hyps,
        refs=refs,
        model_type=ENCODER,
        num_layers=NUM_LAYERS,
        lang="zh",
        rescale_with_baseline=False,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )
    out: List[SampleScore] = []
    for row, p, r, f in zip(rows, P.tolist(), R.tolist(), F1.tolist()):
        out.append(
            SampleScore(
                subject_id=str(row.get("subject_id")),
                source=str(row.get("source")),
                group=_classify_group(row),
                p=float(p),
                r=float(r),
                f1=float(f),
            )
        )
    return out


def _write_per_sample_csv(scores: Sequence[SampleScore], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["subject_id", "source", "group", "bertscore_p", "bertscore_r", "bertscore_f1"]
        )
        for s in scores:
            w.writerow(
                [s.subject_id, s.source, s.group, f"{s.p:.6f}", f"{s.r:.6f}", f"{s.f1:.6f}"]
            )


def _agg(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (float(values[0]), 0.0)
    return (float(mean(values)), float(stdev(values)))


def _summarize_model(
    per_fold_scores: Dict[int, List[SampleScore]],
) -> List[Dict[str, object]]:
    """Per-group mean +/- std across folds, plus the union 'all' group."""
    rows: List[Dict[str, object]] = []
    folds = sorted(per_fold_scores)
    for group in ("all", "real_S1_S5_only", "synthetic_only"):
        fold_p, fold_r, fold_f1, fold_n = [], [], [], 0
        for k in folds:
            scores = per_fold_scores[k]
            if group == "all":
                subset = scores
            else:
                subset = [s for s in scores if s.group == group]
            if not subset:
                continue
            fold_n += len(subset)
            fold_p.append(mean(s.p for s in subset))
            fold_r.append(mean(s.r for s in subset))
            fold_f1.append(mean(s.f1 for s in subset))
        p_mean, p_std = _agg(fold_p)
        r_mean, r_std = _agg(fold_r)
        f_mean, f_std = _agg(fold_f1)
        rows.append(
            {
                "group": group,
                "n": fold_n,
                "P_mean": p_mean, "P_std": p_std,
                "R_mean": r_mean, "R_std": r_std,
                "F1_mean": f_mean, "F1_std": f_std,
            }
        )
    return rows


def _write_summary_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["group", "n", "P_mean", "P_std", "R_mean", "R_std", "F1_mean", "F1_std"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(
                [
                    r["group"], r["n"],
                    f"{r['P_mean']:.6f}", f"{r['P_std']:.6f}",
                    f"{r['R_mean']:.6f}", f"{r['R_std']:.6f}",
                    f"{r['F1_mean']:.6f}", f"{r['F1_std']:.6f}",
                ]
            )


def _process_one_model(
    model_dir: Path,
    folds: Iterable[int],
    device: str,
    batch_size: int,
) -> List[Dict[str, object]]:
    per_fold: Dict[int, List[SampleScore]] = {}
    for k in folds:
        pred_path = model_dir / f"fold{k}_test.json"
        if not pred_path.exists():
            print(f"[bertscore] skip missing: {pred_path}")
            continue
        rows = _load_fold(pred_path)
        print(
            f"[bertscore] {model_dir.name} fold{k}: scoring {len(rows)} samples "
            f"on device={device}, encoder={ENCODER}, layer={NUM_LAYERS}"
        )
        scores = score_fold(rows, device=device, batch_size=batch_size)
        per_fold[k] = scores
        _write_per_sample_csv(scores, model_dir / f"bertscore_fold{k}.csv")
        print(
            f"  -> wrote {model_dir / f'bertscore_fold{k}.csv'} "
            f"(F1 mean={mean(s.f1 for s in scores):.4f}, n={len(scores)})"
        )

    summary = _summarize_model(per_fold)
    _write_summary_csv(summary, model_dir / "bertscore_summary.csv")
    print(f"[bertscore] {model_dir.name} summary -> {model_dir / 'bertscore_summary.csv'}")
    return summary


def _write_cross_model_csv(
    model_to_summary: Dict[str, List[Dict[str, object]]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "model_id", "group", "n",
        "F1_mean", "F1_std", "P_mean", "P_std", "R_mean", "R_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for model_id in DEFAULT_MODELS:
            if model_id not in model_to_summary:
                continue
            for row in model_to_summary[model_id]:
                w.writerow(
                    [
                        model_id, row["group"], row["n"],
                        f"{row['F1_mean']:.6f}", f"{row['F1_std']:.6f}",
                        f"{row['P_mean']:.6f}", f"{row['P_std']:.6f}",
                        f"{row['R_mean']:.6f}", f"{row['R_std']:.6f}",
                    ]
                )
    print(f"[bertscore] cross-model summary -> {path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_result_dir = repo_root / "outputs_llm_final" / "llm_result"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-dir", type=Path, default=default_result_dir,
                    help="Root dir holding per-model subdirs (default: outputs_llm_final/llm_result)")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="Single model directory (overrides --all / --models)")
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                    help="Comma-separated model subdir names under --result-dir")
    ap.add_argument("--all", action="store_true",
                    help="Process every model in --models (default 4 LLMs)")
    ap.add_argument("--folds", type=str, default="1,2,3",
                    help="Comma-separated folds to process")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=str, default=None,
                    help="Override device (cuda/mps/cpu); auto-detect if omitted")
    args = ap.parse_args()

    folds = tuple(int(x) for x in args.folds.split(",") if x.strip())
    device = args.device or _select_device()
    print(f"[bertscore] device={device}, folds={folds}, batch_size={args.batch_size}")

    if args.model_dir is not None:
        targets: List[Path] = [args.model_dir]
        model_ids: List[str] = [args.model_dir.name]
    else:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        targets = [args.result_dir / m for m in model_ids]

    model_to_summary: Dict[str, List[Dict[str, object]]] = {}
    for model_id, model_dir in zip(model_ids, targets):
        if not model_dir.exists():
            print(f"[bertscore] skip missing model dir: {model_dir}")
            continue
        summary = _process_one_model(
            model_dir=model_dir,
            folds=folds,
            device=device,
            batch_size=args.batch_size,
        )
        model_to_summary[model_id] = summary

    if len(model_to_summary) >= 2:
        cross_path = args.result_dir / "comparison" / "bertscore_mean.csv"
        _write_cross_model_csv(model_to_summary, cross_path)


if __name__ == "__main__":
    main()
