"""ROUGE/BLEU metrics v2 for the two-section benchmark."""

from .aggregate import EvaluationRun, evaluate_batch, evaluate_single_case, write_evaluation_outputs
from .candidate_reader import CandidateRecord, read_candidate_report
from .reference_schema import GoldReference, load_reference_set, validate_reference_payload

__all__ = [
    "CandidateRecord",
    "EvaluationRun",
    "GoldReference",
    "evaluate_batch",
    "evaluate_single_case",
    "load_reference_set",
    "read_candidate_report",
    "validate_reference_payload",
    "write_evaluation_outputs",
]
