"""Strict version-aware Candidate reader for Benchmark v2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..canonicalize import canonicalize_rehabilitation_plan
from ..normalization import normalize_medical_text

PROMPT_VERSION = "rehab_llm_benchmark_v2"
STAGE1_PROMPT_V1_VERSION = "rehab_llm_benchmark_stage1_v1"
STAGE1_PROMPT_VERSION = "rehab_llm_benchmark_stage1_v2"
STAGE1_PROMPT_V3_VERSION = "rehab_llm_benchmark_stage1_v3"
STAGE1_PROMPT_ADVICE_VERSION = "rehab_llm_benchmark_stage1_advice_v1"
STAGE1_PROMPT_ADVICE_V2_VERSION = "rehab_llm_benchmark_stage1_advice_v2"
SUPPORTED_PROMPT_VERSIONS = (PROMPT_VERSION, STAGE1_PROMPT_V1_VERSION, STAGE1_PROMPT_VERSION, STAGE1_PROMPT_V3_VERSION, STAGE1_PROMPT_ADVICE_VERSION, STAGE1_PROMPT_ADVICE_V2_VERSION)
REPORT_SCHEMA_VERSION = "rehab.llm-benchmark-report.v2"
CANONICAL_SECTIONS = ("integrated_assessment", "rehabilitation_plan")
EXPECTED_PLAN_COUNT = 3
FORBIDDEN_SECTIONS = (
    "biomarker_interpretation",
    "clinical_subtype",
    "subtype",
    "evidence_summary",
    "limitations",
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
    candidate_missing: bool
    missing_sections: tuple[str, ...]
    unexpected_sections: tuple[str, ...]
    plan_count_violation: bool
    missing_plan_fields: tuple[str, ...]
    errors: tuple[dict[str, Any], ...]
    prompt_version: str | None = None


def missing_candidate_record(*, patient_id: str, model_id: str) -> CandidateRecord:
    return CandidateRecord(
        patient_id=patient_id,
        model_id=model_id,
        source_path=None,
        parsed_output=None,
        parse_success=False,
        schema_valid=False,
        invalid_candidate=False,
        candidate_missing=True,
        missing_sections=CANONICAL_SECTIONS,
        unexpected_sections=(),
        plan_count_violation=False,
        missing_plan_fields=(),
        errors=({"error_code": "CANDIDATE_MISSING", "error_message": "candidate file missing"},),
        prompt_version=None,
    )


def _failure(*, patient_id: str, model_id: str, source_path: str, code: str, message: str) -> CandidateRecord:
    return CandidateRecord(
        patient_id=patient_id,
        model_id=model_id,
        source_path=source_path,
        parsed_output=None,
        parse_success=False,
        schema_valid=False,
        invalid_candidate=True,
        candidate_missing=False,
        missing_sections=CANONICAL_SECTIONS,
        unexpected_sections=(),
        plan_count_violation=False,
        missing_plan_fields=(),
        errors=({"error_code": code, "error_message": message},),
        prompt_version=None,
    )


def read_candidate_report(path: str | Path, *, patient_id: str, model_id: str | None = None) -> CandidateRecord:
    source = Path(path)
    resolved_model = model_id or source.stem
    if not source.exists():
        return missing_candidate_record(patient_id=patient_id, model_id=resolved_model)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        return _failure(patient_id=patient_id, model_id=resolved_model, source_path=str(source), code="CANDIDATE_PARSE_FAILED", message=f"{type(exc).__name__}: {exc}")
    if not isinstance(document, Mapping):
        return _failure(patient_id=patient_id, model_id=resolved_model, source_path=str(source), code="CANDIDATE_NOT_OBJECT", message="candidate JSON must be an object")
    if document.get("schema_version") != REPORT_SCHEMA_VERSION or document.get("prompt_version") not in SUPPORTED_PROMPT_VERSIONS:
        return _failure(
            patient_id=patient_id,
            model_id=resolved_model,
            source_path=str(source),
            code="VERSION_MISMATCH",
            message=f"expected report={REPORT_SCHEMA_VERSION}, prompt in {SUPPORTED_PROMPT_VERSIONS}; found report={document.get('schema_version')}, prompt={document.get('prompt_version')}",
        )
    embedded_patient = document.get("patient_id")
    if embedded_patient is not None and str(embedded_patient) != patient_id:
        return _failure(patient_id=patient_id, model_id=resolved_model, source_path=str(source), code="PATIENT_ID_MISMATCH", message=f"expected={patient_id}; found={embedded_patient}")
    payload = document.get("parsed_model_output")
    if not isinstance(payload, Mapping):
        return _failure(patient_id=patient_id, model_id=resolved_model, source_path=str(source), code="CANDIDATE_NO_PARSED_OUTPUT", message="v2 requires parsed_model_output; Markdown is not a fallback")

    parsed = {key: value for key, value in payload.items() if key in CANONICAL_SECTIONS}
    unexpected = tuple(sorted(str(key) for key in payload.keys() if key not in CANONICAL_SECTIONS))
    errors: list[dict[str, Any]] = []
    missing: list[str] = []
    structural: list[str] = []
    for section in ("integrated_assessment",):
        value = payload.get(section)
        if not isinstance(value, str):
            if section in payload:
                structural.append(f"{section}_not_string")
            missing.append(section)
        elif not normalize_medical_text(value):
            missing.append(section)
    plan_result = canonicalize_rehabilitation_plan(
        payload.get("rehabilitation_plan"), expected_count=EXPECTED_PLAN_COUNT
    )
    if "rehabilitation_plan" not in payload or plan_result.missing:
        missing.append("rehabilitation_plan")
    if plan_result.structural_error:
        structural.append(plan_result.structural_error)
    if unexpected:
        forbidden = [key for key in unexpected if key in FORBIDDEN_SECTIONS]
        errors.append({
            "error_code": "FORBIDDEN_SECTION" if forbidden else "UNEXPECTED_SECTION",
            "error_message": ",".join(unexpected),
        })
    if missing:
        errors.append({"error_code": "SECTION_MISSING", "error_message": ",".join(sorted(set(missing)))})
    if plan_result.missing_fields:
        errors.append({"error_code": "PLAN_FIELD_MISSING", "error_message": ",".join(plan_result.missing_fields)})
    if plan_result.plan_count_violation:
        errors.append({"error_code": "PLAN_COUNT_VIOLATION", "error_message": str(plan_result.count)})
    for error in structural:
        errors.append({"error_code": "CANDIDATE_SCHEMA_ERROR", "error_message": error})
    schema_valid = not missing and not structural and not unexpected and not plan_result.missing_fields and not plan_result.plan_count_violation
    return CandidateRecord(
        patient_id=patient_id,
        model_id=resolved_model,
        source_path=str(source),
        parsed_output=parsed,
        parse_success=True,
        schema_valid=schema_valid,
        invalid_candidate=bool(structural),
        candidate_missing=False,
        missing_sections=tuple(sorted(set(missing))),
        unexpected_sections=unexpected,
        plan_count_violation=plan_result.plan_count_violation,
        missing_plan_fields=plan_result.missing_fields,
        errors=tuple(errors),
        prompt_version=str(document.get("prompt_version") or ""),
    )


__all__ = [
    "CANONICAL_SECTIONS",
    "CandidateRecord",
    "EXPECTED_PLAN_COUNT",
    "STAGE1_PROMPT_VERSION",
    "STAGE1_PROMPT_V3_VERSION",
    "STAGE1_PROMPT_ADVICE_VERSION",
    "SUPPORTED_PROMPT_VERSIONS",
    "FORBIDDEN_SECTIONS",
    "missing_candidate_record",
    "read_candidate_report",
]
