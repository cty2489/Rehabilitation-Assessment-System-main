"""Offline ROUGE/BLEU evaluation for the rehabilitation LLM benchmark.

This package is deliberately independent from the production assessment path.
It reads materialized benchmark JSON and writes evaluation artifacts only.
"""

from .aggregate import (
    EvaluationRun,
    evaluate_batch,
    evaluate_single_case,
    write_evaluation_outputs,
)
from .candidate_reader import CandidateRecord, read_candidate_report
from .canonicalize import CanonicalPlan, canonicalize_rehabilitation_plan
from .reference_schema import (
    GoldReference,
    ReferenceValidationResult,
    load_reference_set,
    validate_reference_payload,
)
from .tokenizer import ChineseMedicalTokenizer

__all__ = [
    "CanonicalPlan",
    "CandidateRecord",
    "ChineseMedicalTokenizer",
    "EvaluationRun",
    "GoldReference",
    "ReferenceValidationResult",
    "canonicalize_rehabilitation_plan",
    "evaluate_batch",
    "evaluate_single_case",
    "load_reference_set",
    "read_candidate_report",
    "validate_reference_payload",
    "write_evaluation_outputs",
]
