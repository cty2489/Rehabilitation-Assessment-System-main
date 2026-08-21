"""Strict Stage 1 candidate contract."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..runner import _first_json_object, _json_object_candidates
from .prompt import STAGE1_PROMPT_VERSION
from ..schemas import RehabilitationAction


class Stage1CandidateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    integrated_assessment: str = Field(min_length=1, max_length=12000)
    rehabilitation_plan: list[RehabilitationAction]

    @property
    def plan_count_violation(self) -> bool:
        """Expected output is three; abnormal counts remain scoreable."""

        return len(self.rehabilitation_plan) != 3


class Stage1AdviceOutput(BaseModel):
    """Model-owned output for the advice-only Stage 1 protocol."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    rehabilitation_plan: list[RehabilitationAction]

    @property
    def plan_count_violation(self) -> bool:
        return len(self.rehabilitation_plan) != 3


class Stage1OutputParseError(ValueError):
    pass


def parse_stage1_model_output(raw_model_output: str) -> Stage1CandidateOutput:
    if not isinstance(raw_model_output, str) or not raw_model_output.strip():
        raise Stage1OutputParseError("模型输出为空")
    try:
        return Stage1CandidateOutput.model_validate(_first_json_object(raw_model_output))
    except Exception as exc:  # pydantic and JSON extraction errors
        raise Stage1OutputParseError(f"模型输出不符合Stage1两段式Schema：{exc}") from exc


def parse_stage1_advice_model_output(raw_model_output: str) -> Stage1AdviceOutput:
    if not isinstance(raw_model_output, str) or not raw_model_output.strip():
        raise Stage1OutputParseError("模型输出为空")
    candidates = _json_object_candidates(raw_model_output)
    if not candidates:
        raise Stage1OutputParseError("模型输出未找到合法JSON对象")
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            # Some local models emit an invalid preliminary object and then a
            # valid advice-only object.  Selecting the first object that
            # satisfies the common schema is deterministic parsing, not text
            # repair; raw_model_output remains unchanged for audit.
            return Stage1AdviceOutput.model_validate(candidate)
        except Exception as exc:  # pydantic validation errors
            last_error = exc
    raise Stage1OutputParseError(
        f"模型输出不符合Stage1 advice-only Schema：{last_error}"
    ) from last_error


def candidate_document(
    *,
    patient_id: str,
    anonymous_report_id: str,
    raw_model_output: str,
    parsed_model_output: Stage1CandidateOutput,
    call_trace: dict[str, Any] | None = None,
    raw_model_outputs: list[str] | tuple[str, ...] | None = None,
    format_retry_count: int = 0,
    clinical_fact_layer: dict[str, Any] | None = None,
    semantic_validation: dict[str, Any] | None = None,
    generation_protocol: str | None = None,
    llm_generated_sections: tuple[str, ...] | list[str] | None = None,
    program_generated_sections: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Create a v2-compatible Candidate without putting patient facts in generated fields."""

    return {
        "schema_version": "rehab.llm-benchmark-report.v2",
        "prompt_version": STAGE1_PROMPT_VERSION,
        "output_schema_version": "rehab.llm-benchmark-output.v2",
        "stage": "stage1_clinical_baseline",
        "patient_id": patient_id,
        "anonymous_report_id": anonymous_report_id,
        "parsed_model_output": parsed_model_output.model_dump(mode="json"),
        "raw_model_output": raw_model_output,
        "raw_model_outputs": list(raw_model_outputs or [raw_model_output]),
        "format_retry_count": int(format_retry_count),
        "model_id": (call_trace or {}).get("model_id"),
        "provider": (call_trace or {}).get("provider"),
        "actual_parameters": dict((call_trace or {}).get("actual_parameters") or {}),
        "unsupported_parameters": list((call_trace or {}).get("unsupported_parameters") or []),
        "ignored_parameters": list((call_trace or {}).get("ignored_parameters") or []),
        "response_time_ms": (call_trace or {}).get("response_time_ms"),
        "clinical_fact_layer": dict(clinical_fact_layer or {}),
        "semantic_validation": dict(semantic_validation or {}),
        "generation_protocol": generation_protocol,
        "llm_generated_sections": list(llm_generated_sections or []),
        "program_generated_sections": list(program_generated_sections or []),
    }


__all__ = [
    "Stage1CandidateOutput",
    "Stage1AdviceOutput",
    "Stage1OutputParseError",
    "candidate_document",
    "parse_stage1_model_output",
    "parse_stage1_advice_model_output",
]
