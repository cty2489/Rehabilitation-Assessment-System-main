"""The single shared generation core for single-case and batch benchmark runs.

This module owns the benchmark switches. Production ``/api/assess`` is not
imported here, so the four experimental combinations cannot silently change
the formal DL path. Callers provide the already-computed biomarker payload and
explicit retrieval/graph/model adapters.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

from .blind_review import anonymous_report_id, build_registry_mapping
from .prompt import build_messages
from .runner import (
    BenchmarkOutputParseError,
    build_evaluation_input,
    build_experiment_log,
    parse_model_output,
    render_report,
)
from .schemas import (
    BenchmarkEvaluationInput,
    BenchmarkExperimentLog,
    BenchmarkGraphContext,
    BenchmarkKnowledgeContext,
    BenchmarkLlmOutput,
    BenchmarkRegistryMapping,
    BenchmarkRunConfig,
)


ModelGenerate = Callable[..., str]
RetrievalProvider = Callable[[BenchmarkEvaluationInput], BenchmarkKnowledgeContext]
GraphProvider = Callable[[BenchmarkEvaluationInput], BenchmarkGraphContext]


class BenchmarkGenerationError(RuntimeError):
    """Raised when the benchmark model output cannot satisfy the fixed schema."""

    def __init__(self, message: str, *, experiment_log: Optional[BenchmarkExperimentLog] = None):
        super().__init__(message)
        self.experiment_log = experiment_log


@dataclass(frozen=True)
class BenchmarkGenerationResult:
    evaluation_input: BenchmarkEvaluationInput
    anonymous_report_id: str
    messages: list[dict[str, str]]
    raw_model_output: str
    parsed_model_output: BenchmarkLlmOutput
    report_markdown: str
    experiment_log: BenchmarkExperimentLog
    registry_mapping: BenchmarkRegistryMapping
    knowledge_context: Optional[BenchmarkKnowledgeContext]
    graph_context: Optional[BenchmarkGraphContext]


def generate_benchmark_report(
    *,
    evaluation_input: Optional[BenchmarkEvaluationInput] = None,
    patient_input: Any = None,
    clinical_scores: Any = None,
    biomarkers: Optional[dict[str, Any]] = None,
    quality: Optional[dict[str, Any]] = None,
    config: Optional[BenchmarkRunConfig] = None,
    model_id: str,
    model_name: Optional[str] = None,
    model_generate: ModelGenerate,
    retrieval_provider: Optional[RetrievalProvider] = None,
    graph_provider: Optional[GraphProvider] = None,
    case_id: str,
    report_index: int,
    batch_id: Optional[str] = None,
) -> BenchmarkGenerationResult:
    """Generate one fixed benchmark report.

    Both the single-case endpoint and batch runner call this function. No
    provider is invoked when its switch is false; there is no fake outage or
    environment-variable mutation used to represent an ablation.
    """
    if evaluation_input is None:
        if patient_input is None or clinical_scores is None or biomarkers is None:
            raise ValueError("未提供evaluation_input，必须同时提供患者、评分和biomarker")
        evaluation_input = build_evaluation_input(
            patient=patient_input,
            clinical_scores=clinical_scores,
            biomarkers=biomarkers,
            quality=quality,
            config=config,
        )
    elif config is not None and evaluation_input.config != config:
        raise ValueError("evaluation_input.config与config不一致")

    run_config = evaluation_input.config
    knowledge_context: Optional[BenchmarkKnowledgeContext] = None
    graph_context: Optional[BenchmarkGraphContext] = None
    if run_config.rag_enabled:
        if retrieval_provider is None:
            raise ValueError("RAG开启时必须提供retrieval_provider")
        knowledge_context = retrieval_provider(evaluation_input)
        if knowledge_context.rag_version != run_config.rag_version:
            raise ValueError("RAG context版本与rag_version不一致")
    if run_config.knowledge_graph_enabled:
        if graph_provider is None:
            raise ValueError("KG开启时必须提供graph_provider")
        graph_context = graph_provider(evaluation_input)
        if graph_context.kg_version != run_config.kg_version:
            raise ValueError("KG context版本与kg_version不一致")

    messages = build_messages(
        evaluation_input,
        knowledge_context=knowledge_context,
        graph_context=graph_context,
    )
    normalized_model_id = str(model_id or "").strip()
    if not normalized_model_id:
        raise ValueError("model_id不能为空")
    anonymous_id = anonymous_report_id(case_id, report_index)
    started = time.perf_counter()
    raw_output = ""
    parsed: Optional[BenchmarkLlmOutput] = None
    error: Optional[str] = None
    try:
        raw_output = str(
            model_generate(
                messages,
                model_id=normalized_model_id,
                temperature=run_config.temperature,
                max_new_tokens=run_config.max_new_tokens,
            )
        )
        parsed = parse_model_output(raw_output)
    except (BenchmarkOutputParseError, ValueError, TypeError) as exc:
        error = str(exc)
        log = build_experiment_log(
            evaluation_input=evaluation_input,
            anonymous_report_id=anonymous_id,
            llm_model_id=normalized_model_id,
            llm_model_name=model_name or normalized_model_id,
            raw_model_output=raw_output,
            parsed_model_output=None,
            temperature=run_config.temperature,
            max_new_tokens=run_config.max_new_tokens,
            generation_time=(time.perf_counter() - started) * 1000,
            batch_id=batch_id,
            error=error,
        )
        raise BenchmarkGenerationError(error, experiment_log=log) from exc

    generation_time_ms = (time.perf_counter() - started) * 1000
    assert parsed is not None
    cards: Sequence[Any] = knowledge_context.evidence_cards if knowledge_context else ()
    report = render_report(evaluation_input, parsed, evidence_cards=cards)
    log = build_experiment_log(
        evaluation_input=evaluation_input,
        anonymous_report_id=anonymous_id,
        llm_model_id=normalized_model_id,
        llm_model_name=model_name or normalized_model_id,
        raw_model_output=raw_output,
        parsed_model_output=parsed,
        temperature=run_config.temperature,
        max_new_tokens=run_config.max_new_tokens,
        generation_time=generation_time_ms,
        batch_id=batch_id,
    )
    mapping = build_registry_mapping(
        report_id=anonymous_id,
        evaluation_input=evaluation_input,
        llm_model_id=normalized_model_id,
    )
    return BenchmarkGenerationResult(
        evaluation_input=evaluation_input,
        anonymous_report_id=anonymous_id,
        messages=messages,
        raw_model_output=raw_output,
        parsed_model_output=parsed,
        report_markdown=report,
        experiment_log=log,
        registry_mapping=mapping,
        knowledge_context=knowledge_context,
        graph_context=graph_context,
    )


def generate_benchmark_batch_reports(
    evaluation_inputs: Iterable[BenchmarkEvaluationInput],
    *,
    model_id: str,
    model_name: Optional[str] = None,
    model_generate: ModelGenerate,
    case_start: int = 1,
    report_index: int = 1,
    batch_id: Optional[str] = None,
    retrieval_provider: Optional[RetrievalProvider] = None,
    graph_provider: Optional[GraphProvider] = None,
) -> list[BenchmarkGenerationResult]:
    """Run a batch through the exact same single-report core function."""
    results: list[BenchmarkGenerationResult] = []
    for offset, evaluation_input in enumerate(evaluation_inputs):
        case_id = f"CASE{case_start + offset:03d}"
        results.append(
            generate_benchmark_report(
                evaluation_input=evaluation_input,
                model_id=model_id,
                model_name=model_name,
                model_generate=model_generate,
                case_id=case_id,
                report_index=report_index,
                batch_id=batch_id,
                retrieval_provider=retrieval_provider,
                graph_provider=graph_provider,
            )
        )
    return results


__all__ = [
    "BenchmarkGenerationError",
    "BenchmarkGenerationResult",
    "GraphProvider",
    "ModelGenerate",
    "RetrievalProvider",
    "generate_benchmark_report",
    "generate_benchmark_batch_reports",
]
