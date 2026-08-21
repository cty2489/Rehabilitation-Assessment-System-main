"""Read structured Benchmark Candidates without parsing Markdown."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonicalize import canonicalize_rehabilitation_plan
from .normalization import normalize_medical_text

CANONICAL_SECTIONS = (
    "biomarker_interpretation",
    "integrated_assessment",
    "rehabilitation_plan",
)


@dataclass(frozen=True)
class CandidateRecord:
    patient_id: str
    model_id: str
    source_path: str | None
    parsed_output: dict[str, Any] | None
    parse_success: bool
    schema_valid: bool
    invalid_candidate: bool
    missing_sections: tuple[str, ...]
    unexpected_sections: tuple[str, ...]
    plan_count_violation: bool
    missing_plan_fields: tuple[str, ...]
    errors: tuple[dict[str, Any], ...]

    @property
    def missing_any_section(self) -> bool:
        return bool(self.missing_sections)


def missing_candidate_record(*, patient_id: str, model_id: str, message: str = "candidate file missing") -> CandidateRecord:
    error = {
        "error_code": "CANDIDATE_MISSING",
        "error_message": message,
    }
    return CandidateRecord(
        patient_id=patient_id,
        model_id=model_id,
        source_path=None,
        parsed_output=None,
        parse_success=False,
        schema_valid=False,
        invalid_candidate=True,
        missing_sections=CANONICAL_SECTIONS,
        unexpected_sections=(),
        plan_count_violation=False,
        missing_plan_fields=(),
        errors=(error,),
    )


def _empty_failure(
    *,
    patient_id: str,
    model_id: str,
    source_path: str | None,
    error_code: str,
    error_message: str,
) -> CandidateRecord:
    return CandidateRecord(
        patient_id=patient_id,
        model_id=model_id,
        source_path=source_path,
        parsed_output=None,
        parse_success=False,
        schema_valid=False,
        invalid_candidate=True,
        missing_sections=CANONICAL_SECTIONS,
        unexpected_sections=(),
        plan_count_violation=False,
        missing_plan_fields=(),
        errors=({"error_code": error_code, "error_message": error_message},),
    )


def read_candidate_report(
    path: str | Path,
    *,
    patient_id: str,
    model_id: str | None = None,
) -> CandidateRecord:
    """Read ``parsed_model_output`` from an existing materialized report.

    A direct structured Candidate JSON is accepted for fixtures and offline
    tooling.  Markdown and ``raw_model_output`` are never used as fallback.
    """

    source = Path(path)
    resolved_model_id = model_id or source.stem
    if not source.exists():
        return missing_candidate_record(patient_id=patient_id, model_id=resolved_model_id)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - parser implementation detail
        return _empty_failure(
            patient_id=patient_id,
            model_id=resolved_model_id,
            source_path=str(source),
            error_code="CANDIDATE_PARSE_FAILED",
            error_message=f"{type(exc).__name__}: {exc}",
        )
    if not isinstance(document, Mapping):
        return _empty_failure(
            patient_id=patient_id,
            model_id=resolved_model_id,
            source_path=str(source),
            error_code="CANDIDATE_NOT_OBJECT",
            error_message="candidate JSON must be an object",
        )

    embedded_patient_id = document.get("patient_id")
    payload = document.get("parsed_model_output")
    if payload is None and any(key in document for key in CANONICAL_SECTIONS):
        payload = document
    if embedded_patient_id is not None and str(embedded_patient_id) != patient_id:
        return _empty_failure(
            patient_id=patient_id,
            model_id=resolved_model_id,
            source_path=str(source),
            error_code="PATIENT_ID_MISMATCH",
            error_message=f"expected={patient_id}; found={embedded_patient_id}",
        )
    if not isinstance(payload, Mapping):
        return _empty_failure(
            patient_id=patient_id,
            model_id=resolved_model_id,
            source_path=str(source),
            error_code="CANDIDATE_NO_PARSED_OUTPUT",
            error_message="parsed_model_output is absent or not an object; Markdown is not a fallback",
        )

    parsed = {key: value for key, value in payload.items() if key in CANONICAL_SECTIONS}
    unexpected = tuple(sorted(str(key) for key in payload.keys() if key not in CANONICAL_SECTIONS))
    errors: list[dict[str, Any]] = []
    missing_sections: list[str] = []
    structural_errors: list[str] = []

    for section in CANONICAL_SECTIONS[:2]:
        value = payload.get(section)
        if not isinstance(value, str):
            if section in payload:
                structural_errors.append(f"{section}_not_string")
            missing_sections.append(section)
        elif not normalize_medical_text(value):
            missing_sections.append(section)

    plan_result = canonicalize_rehabilitation_plan(payload.get("rehabilitation_plan"))
    if "rehabilitation_plan" not in payload or plan_result.missing:
        missing_sections.append("rehabilitation_plan")
    if plan_result.structural_error:
        structural_errors.append(plan_result.structural_error)

    if unexpected:
        errors.append({
            "error_code": "UNEXPECTED_SECTION",
            "error_message": ",".join(unexpected),
        })
    if missing_sections:
        errors.append({
            "error_code": "SECTION_MISSING",
            "error_message": ",".join(sorted(set(missing_sections))),
        })
    if plan_result.missing_fields:
        errors.append({
            "error_code": "PLAN_FIELD_MISSING",
            "error_message": ",".join(plan_result.missing_fields),
        })
    if plan_result.plan_count_violation:
        errors.append({
            "error_code": "PLAN_COUNT_VIOLATION",
            "error_message": str(plan_result.count),
        })
    for error in structural_errors:
        errors.append({"error_code": "CANDIDATE_SCHEMA_ERROR", "error_message": error})

    # Missing sections and plan cardinality remain scoreable under the fixed
    # denominator.  Only parse failure, patient mismatch, or structural
    # corruption makes the whole Candidate invalid.
    invalid = bool(structural_errors)
    schema_valid = not missing_sections and not structural_errors and not plan_result.missing_fields and not plan_result.plan_count_violation
    return CandidateRecord(
        patient_id=patient_id,
        model_id=resolved_model_id,
        source_path=str(source),
        parsed_output=parsed,
        parse_success=True,
        schema_valid=schema_valid,
        invalid_candidate=invalid,
        missing_sections=tuple(sorted(set(missing_sections))),
        unexpected_sections=unexpected,
        plan_count_violation=plan_result.plan_count_violation,
        missing_plan_fields=plan_result.missing_fields,
        errors=tuple(errors),
    )


__all__ = [
    "CANONICAL_SECTIONS",
    "CandidateRecord",
    "missing_candidate_record",
    "read_candidate_report",
]
