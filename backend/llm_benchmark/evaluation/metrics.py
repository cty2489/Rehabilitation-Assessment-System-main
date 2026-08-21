"""ROUGE and SacreBLEU metric adapters for metrics v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sacrebleu
from rouge_score import rouge_scorer

from .tokenizer import ChineseMedicalTokenizer

METRICS_VERSION = "rehab_llm_metrics_v1"
ROUGE_NAMES = ("rouge1", "rouge2", "rougeL")


@dataclass(frozen=True)
class RougeValues:
    precision: float
    recall: float
    f1: float


def score_rouge(reference_text: str, candidate_text: str, *, tokenizer: ChineseMedicalTokenizer | None = None) -> dict[str, RougeValues]:
    """Score one normalized text pair using the versioned medical tokenizer."""

    tokenizer = tokenizer or ChineseMedicalTokenizer()
    if not tokenizer.tokenize(reference_text):
        raise ValueError("reference text has no metric tokens")
    if not tokenizer.tokenize(candidate_text):
        return {name: RougeValues(0.0, 0.0, 0.0) for name in ROUGE_NAMES}
    scorer = rouge_scorer.RougeScorer(
        list(ROUGE_NAMES),
        tokenizer=tokenizer,
        use_stemmer=False,
    )
    result = scorer.score(reference_text, candidate_text)
    return {
        name: RougeValues(
            precision=float(result[name].precision),
            recall=float(result[name].recall),
            f1=float(result[name].fmeasure),
        )
        for name in ROUGE_NAMES
    }


def _bleu_metric(*, effective_order: bool = False) -> sacrebleu.metrics.BLEU:
    return sacrebleu.metrics.BLEU(
        tokenize="zh",
        lowercase=False,
        smooth_method="exp",
        effective_order=effective_order,
    )


def sentence_bleu(reference_text: str, candidate_text: str) -> float:
    """Return auxiliary sentence BLEU on SacreBLEU's 0--100 scale."""

    if not candidate_text.strip():
        return 0.0
    # SacreBLEU recommends effective-order handling for short sentence-level
    # diagnostics.  The primary corpus BLEU below keeps the locked default.
    score = _bleu_metric(effective_order=True).sentence_score(candidate_text, [reference_text])
    return float(score.score)


def corpus_bleu(references: Sequence[str], candidates: Sequence[str]) -> dict[str, object]:
    """Return one corpus BLEU-4 result for aligned reference/candidate lists."""

    if len(references) != len(candidates):
        raise ValueError("reference/candidate corpus lengths differ")
    if not references:
        raise ValueError("cannot score an empty corpus")
    metric = _bleu_metric()
    score = metric.corpus_score(list(candidates), [list(references)])
    return {
        "bleu": float(score.score),
        "bleu_normalized": float(score.score) / 100.0,
        "bp": float(score.bp),
        "sys_len": int(score.sys_len),
        "ref_len": int(score.ref_len),
        "precisions": [float(value) for value in score.precisions],
        "signature": str(metric.get_signature()),
        "sacrebleu_version": str(sacrebleu.__version__),
        "tokenizer": "zh",
        "lowercase": False,
        "smooth_method": "exp",
        "nrefs": 1,
    }


__all__ = [
    "METRICS_VERSION",
    "ROUGE_NAMES",
    "RougeValues",
    "corpus_bleu",
    "sacrebleu",
    "score_rouge",
    "sentence_bleu",
]
