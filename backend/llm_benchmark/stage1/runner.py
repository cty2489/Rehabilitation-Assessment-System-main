"""Stage 1 clinical-only runner; model calls are injected by the caller."""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..blind_review import anonymous_report_id
from .clinical_input import Stage1Case
from .clinical_facts import build_stage1_fact_layer, render_program_integrated_assessment
from .output import (
    Stage1AdviceOutput,
    Stage1CandidateOutput,
    Stage1OutputParseError,
    candidate_document,
    parse_stage1_advice_model_output,
    parse_stage1_model_output,
)
from .semantic import validate_stage1_semantics
from .prompt import (
    STAGE1_ADVICE_SERIALIZATION_RETRY_SUFFIX,
    STAGE1_PROTOCOL,
    STAGE1_PROMPT_VERSION,
    STAGE1_SERIALIZATION_RETRY_VERSION,
    build_stage1_messages,
    build_stage1_advice_serialization_retry_messages,
    build_stage1_serialization_retry_messages,
)

STAGE1_NAME = "stage1_clinical_baseline"
INPUT_SCHEMA_VERSION = "rehab.llm-benchmark-stage1-input.v1"
REFERENCE_VERSION = "gold_v2"


class Stage1RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    stage: str = STAGE1_NAME
    model_ids: list[str] = Field(min_length=1)
    prompt_version: str = STAGE1_PROMPT_VERSION
    protocol: str = STAGE1_PROTOCOL
    input_schema_version: str = INPUT_SCHEMA_VERSION
    reference_version: str = REFERENCE_VERSION
    rag_enabled: bool = False
    knowledge_graph_enabled: bool = False
    dl_prediction_enabled: bool = False
    biomarker_injected: bool = False
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1)

    @model_validator(mode="after")
    def validate_contract(self) -> "Stage1RunConfig":
        if self.stage != STAGE1_NAME:
            raise ValueError(f"stage必须为{STAGE1_NAME}")
        if self.prompt_version != STAGE1_PROMPT_VERSION:
            raise ValueError(f"Stage1必须使用{STAGE1_PROMPT_VERSION}")
        if self.protocol != STAGE1_PROTOCOL:
            raise ValueError(f"Stage1必须使用{STAGE1_PROTOCOL}")
        if self.input_schema_version != INPUT_SCHEMA_VERSION:
            raise ValueError(f"Stage1必须使用{INPUT_SCHEMA_VERSION}")
        if self.reference_version != REFERENCE_VERSION:
            raise ValueError(f"Stage1必须使用{REFERENCE_VERSION}")
        if self.rag_enabled or self.knowledge_graph_enabled or self.dl_prediction_enabled or self.biomarker_injected:
            raise ValueError("Stage1 clinical-only必须关闭RAG、KG、DL和biomarker注入")
        normalized = [str(model_id).strip() for model_id in self.model_ids]
        if any(not model_id for model_id in normalized):
            raise ValueError("model_ids不能包含空值")
        object.__setattr__(self, "model_ids", normalized)
        return self


@dataclass(frozen=True)
class Stage1GenerationResult:
    case: Stage1Case
    model_id: str
    anonymous_report_id: str
    messages: list[dict[str, str]]
    raw_model_output: str
    raw_model_outputs: tuple[str, ...]
    parsed_model_output: Stage1CandidateOutput
    candidate_document: dict[str, Any]
    generation_time_ms: float
    call_trace: dict[str, Any]
    semantic_validation: dict[str, Any]


@dataclass(frozen=True)
class Stage1ExperimentRun:
    experiment_id: str
    output_root: Path
    manifest: dict[str, Any]
    results: tuple[Stage1GenerationResult, ...]


class Stage1GenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw_model_output: str = "",
        raw_model_outputs: Optional[Iterable[str]] = None,
        trace: Optional[dict[str, Any]] = None,
        format_retry_count: int = 0,
    ):
        super().__init__(message)
        self.raw_model_output = raw_model_output
        self.raw_model_outputs = tuple(raw_model_outputs or ([raw_model_output] if raw_model_output else []))
        self.trace = trace or {}
        self.format_retry_count = int(format_retry_count)


ModelGenerate = Callable[..., str]


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {"git_commit": commit or "unknown", "git_dirty": bool(dirty)}


def build_stage1_manifest(
    *,
    experiment_id: str,
    cases: Iterable[Stage1Case],
    config: Stage1RunConfig,
    timestamp: str | None = None,
) -> dict[str, Any]:
    case_list = tuple(cases)
    return {
        "experiment_id": experiment_id,
        "stage": STAGE1_NAME,
        "patient_ids": [case.patient_id for case in case_list],
        "model_ids": list(config.model_ids),
        "prompt_version": config.prompt_version,
        "protocol": config.protocol,
        "input_schema_version": config.input_schema_version,
        "reference_version": config.reference_version,
        "rag_enabled": False,
        "knowledge_graph_enabled": False,
        "dl_prediction_enabled": False,
        "biomarker_injected": False,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "serialization_retry": {
            "version": STAGE1_SERIALIZATION_RETRY_VERSION,
            "max_retries": 1,
            "on": "output_schema_parse_failure",
        },
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        **_git_metadata(),
    }


def generate_stage1_report(
    *,
    case: Stage1Case,
    model_id: str,
    model_generate: ModelGenerate,
    config: Optional[Stage1RunConfig] = None,
    case_id: Optional[str] = None,
    report_index: int = 1,
) -> Stage1GenerationResult:
    run_config = config or Stage1RunConfig(model_ids=[model_id])
    normalized_model_id = str(model_id or "").strip()
    if normalized_model_id not in run_config.model_ids:
        raise ValueError("model_id必须在Stage1RunConfig.model_ids中")
    messages = build_stage1_messages(case.clinical_input)
    anonymous_id = anonymous_report_id(case_id or "CASE001", report_index)
    started = time.perf_counter()
    raw_output = ""
    raw_outputs: list[str] = []
    format_retry_count = 0

    def call_model(call_messages: list[dict[str, str]]) -> str:
        nonlocal raw_outputs
        output = str(model_generate(
            call_messages,
            model_id=normalized_model_id,
            temperature=run_config.temperature,
            top_p=run_config.top_p,
            max_tokens=run_config.max_tokens,
        ))
        provider_attempts = list(getattr(model_generate, "last_raw_attempts", []) or [])
        if provider_attempts:
            raw_outputs.extend(str(item) for item in provider_attempts)
        if not provider_attempts or provider_attempts[-1] != output:
            raw_outputs.append(output)
        return output

    def current_trace() -> dict[str, Any]:
        return dict(
            getattr(model_generate, "last_trace", {})
            or getattr(model_generate, "last_error_trace", {})
            or {}
        )

    def parse_current_output(raw: str) -> Stage1CandidateOutput:
        if run_config.protocol in {
            "program_fact_layer_llm_advice_v1",
            "program_fact_layer_llm_advice_v2",
        }:
            advice: Stage1AdviceOutput = parse_stage1_advice_model_output(raw)
            return Stage1CandidateOutput(
                integrated_assessment=render_program_integrated_assessment(case.clinical_input),
                rehabilitation_plan=advice.rehabilitation_plan,
            )
        return parse_stage1_model_output(raw)

    def build_current_retry_messages() -> list[dict[str, str]]:
        if run_config.protocol in {
            "program_fact_layer_llm_advice_v1",
            "program_fact_layer_llm_advice_v2",
        }:
            return build_stage1_advice_serialization_retry_messages(messages)
        return build_stage1_serialization_retry_messages(messages)

    try:
        raw_output = call_model(messages)
        try:
            parsed = parse_current_output(raw_output)
        except Stage1OutputParseError as first_parse_error:
            format_retry_count = 1
            retry_messages = build_current_retry_messages()
            try:
                raw_output = call_model(retry_messages)
                parsed = parse_current_output(raw_output)
            except Exception as retry_error:  # noqa: BLE001 - model/provider boundary
                raise Stage1GenerationError(
                    str(retry_error),
                    raw_model_output=raw_output,
                    raw_model_outputs=raw_outputs,
                    trace=current_trace(),
                    format_retry_count=format_retry_count,
                ) from retry_error
    except Exception as exc:  # noqa: BLE001 - model/provider boundary
        if isinstance(exc, Stage1GenerationError):
            raise
        raise Stage1GenerationError(
            str(exc),
            raw_model_output=raw_output,
            raw_model_outputs=raw_outputs,
            trace=current_trace(),
            format_retry_count=format_retry_count,
        ) from exc
    elapsed = (time.perf_counter() - started) * 1000
    call_trace = current_trace()
    call_trace["format_retry_count"] = format_retry_count
    call_trace["serialization_retry_version"] = STAGE1_SERIALIZATION_RETRY_VERSION
    document = candidate_document(
        patient_id=case.patient_id,
        anonymous_report_id=anonymous_id,
        raw_model_output=raw_output,
        parsed_model_output=parsed,
        call_trace=call_trace,
        raw_model_outputs=raw_outputs,
        format_retry_count=format_retry_count,
        clinical_fact_layer=build_stage1_fact_layer(case.clinical_input),
        semantic_validation=validate_stage1_semantics(parsed, case.clinical_input).to_dict(),
        generation_protocol=run_config.protocol,
        llm_generated_sections=("rehabilitation_plan",),
        program_generated_sections=("integrated_assessment",),
    )
    semantic_validation = dict(document["semantic_validation"])
    return Stage1GenerationResult(
        case=case,
        model_id=normalized_model_id,
        anonymous_report_id=anonymous_id,
        messages=messages,
        raw_model_output=raw_output,
        raw_model_outputs=tuple(raw_outputs),
        parsed_model_output=parsed,
        candidate_document=document,
        generation_time_ms=elapsed,
        call_trace=call_trace,
        semantic_validation=semantic_validation,
    )


def generate_stage1_batch_reports(
    cases: Iterable[Stage1Case],
    *,
    model_ids: Iterable[str],
    model_generate: ModelGenerate,
    config: Optional[Stage1RunConfig] = None,
) -> list[Stage1GenerationResult]:
    case_list = tuple(cases)
    run_config = config or Stage1RunConfig(model_ids=list(model_ids))
    results: list[Stage1GenerationResult] = []
    for case_index, case in enumerate(case_list, start=1):
        for model_id in run_config.model_ids:
            results.append(generate_stage1_report(
                case=case,
                model_id=model_id,
                model_generate=model_generate,
                config=run_config,
                case_id=f"CASE{case_index:03d}",
                report_index=run_config.model_ids.index(model_id) + 1,
            ))
    return results


def _model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._") or "model"


def write_stage1_result(root: str | Path, result: Stage1GenerationResult) -> Path:
    """Write one v2-compatible Candidate under the Stage 1 batch layout."""

    patient_root = Path(root) / "patients" / result.case.patient_id / "reports"
    patient_root.mkdir(parents=True, exist_ok=True)
    report_path = patient_root / f"{_model_slug(result.model_id)}.json"
    report_path.write_text(json.dumps(result.candidate_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def write_stage1_manifest(root: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(root) / "stage1_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_stage1_experiment(
    *,
    cases: Iterable[Stage1Case],
    config: Stage1RunConfig,
    model_generate: ModelGenerate,
    output_root: str | Path,
    experiment_id: str = "stage1-clinical-baseline",
) -> Stage1ExperimentRun:
    """Future single/batch entry: persist manifest, Candidates and raw logs."""

    case_list = tuple(cases)
    root = Path(output_root)
    manifest = build_stage1_manifest(experiment_id=experiment_id, cases=case_list, config=config)
    write_stage1_manifest(root, manifest)
    results: list[Stage1GenerationResult] = []
    logs_path = root / "logs" / "experiment_stage1.jsonl"
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    with logs_path.open("w", encoding="utf-8") as logs:
        for case_index, case in enumerate(case_list, start=1):
            for model_index, model_id in enumerate(config.model_ids, start=1):
                try:
                    result = generate_stage1_report(
                        case=case,
                        model_id=model_id,
                        model_generate=model_generate,
                        config=config,
                        case_id=f"CASE{case_index:03d}",
                        report_index=model_index,
                    )
                except Stage1GenerationError as exc:
                    trace = dict(exc.trace or {})
                    log = {
                        "experiment_id": experiment_id,
                        "stage": STAGE1_NAME,
                        "patient_id": case.patient_id,
                        "anonymous_report_id": f"CASE{case_index:03d}-R{model_index:02d}",
                        "llm_model_id": model_id,
                        "prompt_version": config.prompt_version,
                        "input_schema_version": config.input_schema_version,
                        "rag_enabled": False,
                        "knowledge_graph_enabled": False,
                        "dl_prediction_enabled": False,
                        "biomarker_injected": False,
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "max_tokens": config.max_tokens,
                        "raw_model_output": exc.raw_model_output,
                        "raw_model_outputs": list(exc.raw_model_outputs),
                        "format_retry_count": exc.format_retry_count,
                        "parsed_model_output": None,
                        "parse_success": False,
                        "error": str(exc),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "provider": trace.get("provider"),
                        "actual_parameters": trace.get("actual_parameters", {}),
                        "unsupported_parameters": trace.get("unsupported_parameters", []),
                        "ignored_parameters": trace.get("ignored_parameters", []),
                        "response_time_ms": trace.get("response_time_ms"),
                    }
                    logs.write(json.dumps(log, ensure_ascii=False) + "\n")
                    continue
                write_stage1_result(root, result)
                logs.write(json.dumps({
                    "experiment_id": experiment_id,
                    "stage": STAGE1_NAME,
                    "patient_id": result.case.patient_id,
                    "anonymous_report_id": result.anonymous_report_id,
                    "llm_model_id": result.model_id,
                    "prompt_version": config.prompt_version,
                    "input_schema_version": config.input_schema_version,
                    "rag_enabled": False,
                    "knowledge_graph_enabled": False,
                    "dl_prediction_enabled": False,
                    "biomarker_injected": False,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "max_tokens": config.max_tokens,
                    "generation_time_ms": result.generation_time_ms,
                    "raw_model_output": result.raw_model_output,
                    "raw_model_outputs": list(result.raw_model_outputs),
                    "format_retry_count": result.call_trace.get("format_retry_count", 0),
                    "parsed_model_output": result.parsed_model_output.model_dump(mode="json"),
                    "parse_success": True,
                    "plan_count_violation": result.parsed_model_output.plan_count_violation,
                    "provider": result.call_trace.get("provider"),
                    "actual_parameters": result.call_trace.get("actual_parameters", {}),
                    "unsupported_parameters": result.call_trace.get("unsupported_parameters", []),
                    "ignored_parameters": result.call_trace.get("ignored_parameters", []),
                    "response_time_ms": result.call_trace.get("response_time_ms", result.generation_time_ms),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")
                results.append(result)
    return Stage1ExperimentRun(experiment_id, root, manifest, tuple(results))


__all__ = [
    "INPUT_SCHEMA_VERSION",
    "REFERENCE_VERSION",
    "STAGE1_NAME",
    "STAGE1_PROMPT_VERSION",
    "Stage1GenerationError",
    "Stage1ExperimentRun",
    "Stage1GenerationResult",
    "Stage1RunConfig",
    "build_stage1_manifest",
    "generate_stage1_batch_reports",
    "generate_stage1_report",
    "run_stage1_experiment",
    "write_stage1_manifest",
    "write_stage1_result",
]
