"""Create the doctor-facing blind-review projection and restricted mapping."""
from __future__ import annotations

import re

from .schemas import (
    BenchmarkEvaluationInput,
    BenchmarkRegistryMapping,
    BlindReviewPacket,
    DoctorReviewScore,
)


_CASE_ID = re.compile(r"^CASE\d{3,}$", re.IGNORECASE)


def anonymous_report_id(case_id: str, report_index: int) -> str:
    """Return a stable, human-readable blind-review ID such as CASE001-R01."""
    normalized = str(case_id).strip().upper()
    if not _CASE_ID.fullmatch(normalized):
        raise ValueError("case_id必须使用CASE加至少三位数字，例如CASE001")
    if report_index < 1:
        raise ValueError("report_index必须从1开始")
    return f"{normalized}-R{report_index:02d}"


def build_blind_review_packet(
    *,
    report_id: str,
    evaluation_input: BenchmarkEvaluationInput,
    report_markdown: str,
    doctor_score: DoctorReviewScore | None = None,
) -> BlindReviewPacket:
    """Expose only fixed patient facts, anonymous ID, report and doctor score."""
    return BlindReviewPacket(
        anonymous_report_id=report_id,
        patient=evaluation_input.patient,
        report_markdown=report_markdown,
        doctor_score=doctor_score,
    )


def build_registry_mapping(
    *,
    report_id: str,
    evaluation_input: BenchmarkEvaluationInput,
    llm_model_id: str,
) -> BenchmarkRegistryMapping:
    """Keep model and ablation metadata outside the doctor-facing packet."""
    config = evaluation_input.config
    return BenchmarkRegistryMapping(
        anonymous_report_id=report_id,
        patient_id=evaluation_input.patient.patient_id,
        llm_model_id=llm_model_id,
        assessment_input_mode=evaluation_input.config.assessment_input_mode,
        clinical_score_source=evaluation_input.clinical_score_source,
        rag_enabled=config.rag_enabled,
        rag_version=config.rag_version,
        knowledge_graph_enabled=config.knowledge_graph_enabled,
        kg_version=config.kg_version,
    )


__all__ = [
    "anonymous_report_id",
    "build_blind_review_packet",
    "build_registry_mapping",
]
