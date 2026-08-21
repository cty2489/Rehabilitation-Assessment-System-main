"""Production boundary for the frozen ``planner_rag`` v0.1 pipeline.

This module only adapts existing assessment objects, assembles the already
implemented pipeline, and renders its structured report for the legacy
Markdown/SSE surface. It contains no clinical classification rules.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Mapping, Optional

import knowledge_admin
from biomarker_refs import marker_ref
from pydantic import ValidationError

from .config import (
    CoreKnowledgeConfig,
    KnowledgeGraphConfig,
    LlmRoleConfig,
    PipelineConfig,
)
from .contracts import CanonicalBiomarker, CanonicalPredictions
from .knowledge_planner import ExistingLlmClient, KnowledgePlanner
from .orchestrator import (
    ClinicalPipelineOrchestrator,
    OrchestrationResult,
    PipelineAssessmentInput,
    PipelinePatientInput,
    PipelineRunStatus,
)
from .report_generator import (
    ExistingReportLlmClient,
    LlmStrategyContractError,
    ReportGenerator,
    ReportResult,
    build_conservative_report,
    build_llm_strategy_recovery_report,
)
from .validator import ValidationResult, Validator


class ProductionAdapterError(ValueError):
    """Raised when production data cannot satisfy the pipeline input contract."""


class ProductionPipelineBlockedError(ValueError):
    """Raised when QualityGate blocks a production assessment."""


class ProductionPipelineExecutionError(RuntimeError):
    """Raised when an orchestrated production run fails or is incomplete."""


@dataclass(frozen=True)
class ProductionPipelineRequest:
    assessment_input: PipelineAssessmentInput
    report_model_id: str


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ProductionAdapterError(f"{name}不能为空")
    return text


def _optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProductionAdapterError(f"{field_name}必须是有限数值")
    number = float(value)
    if not math.isfinite(number):
        raise ProductionAdapterError(f"{field_name}必须是有限数值")
    return number


def _canonical_biomarkers(value: Optional[Mapping[str, Any]]) -> list[CanonicalBiomarker]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        raise ProductionAdapterError("biomarkers必须是对象或null")
    groups = value.get("groups")
    if groups is None:
        return []
    if not isinstance(groups, list):
        raise ProductionAdapterError("biomarkers.groups必须是列表")

    output: list[CanonicalBiomarker] = []
    seen_keys: set[str] = set()
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ProductionAdapterError(
                f"biomarkers.groups[{group_index}]必须是对象"
            )
        modality = _required_text(
            group.get("key"), f"biomarkers.groups[{group_index}].key"
        ).lower()
        if modality not in {"eeg", "emg", "imu"}:
            raise ProductionAdapterError(
                f"biomarkers.groups[{group_index}].key不支持：{modality}"
            )
        markers = group.get("markers")
        if not isinstance(markers, list):
            raise ProductionAdapterError(
                f"biomarkers.groups[{group_index}].markers必须是列表"
            )

        for marker_index, marker in enumerate(markers):
            prefix = f"biomarkers.groups[{group_index}].markers[{marker_index}]"
            if not isinstance(marker, Mapping):
                raise ProductionAdapterError(f"{prefix}必须是对象")
            metric_key = _required_text(marker.get("key"), f"{prefix}.key")
            if metric_key in seen_keys:
                raise ProductionAdapterError(f"biomarker key重复：{metric_key}")
            seen_keys.add(metric_key)

            available_raw = marker.get("available", True)
            if not isinstance(available_raw, bool):
                raise ProductionAdapterError(f"{prefix}.available必须是布尔值")
            n_valid_raw = marker.get("n_valid", 0)
            if isinstance(n_valid_raw, bool):
                raise ProductionAdapterError(f"{prefix}.n_valid必须是非负整数")
            try:
                n_valid = int(n_valid_raw)
            except (TypeError, ValueError) as exc:
                raise ProductionAdapterError(
                    f"{prefix}.n_valid必须是非负整数"
                ) from exc
            if n_valid < 0:
                raise ProductionAdapterError(f"{prefix}.n_valid必须是非负整数")

            raw_value = marker.get("value")
            number = (
                _finite_number(raw_value, f"{prefix}.value")
                if available_raw
                else None
            )
            try:
                output.append(
                    CanonicalBiomarker(
                        metric_key=metric_key,
                        name=_required_text(marker.get("name"), f"{prefix}.name"),
                        value=number,
                        unit=_optional_text(marker.get("unit")),
                        modality=modality,
                        available=available_raw,
                        n_valid=n_valid,
                    )
                )
            except ValidationError as exc:
                raise ProductionAdapterError(
                    f"{prefix}不符合CanonicalBiomarker契约：{exc}"
                ) from exc
    return output


def adapt_production_input(
    *,
    patient: Any,
    predictions_raw: Mapping[str, Any],
    biomarkers: Optional[Mapping[str, Any]],
    quality: Mapping[str, Any],
    assessment_id: Optional[str],
    patient_id: str,
    report_model_id: str,
    context_id: Optional[str] = None,
) -> ProductionPipelineRequest:
    """Convert one completed inference result into the frozen input contract."""
    if patient is None:
        raise ProductionAdapterError("SessionState.patient不能为空")
    if not isinstance(predictions_raw, Mapping):
        raise ProductionAdapterError("predictions_raw必须是对象")
    if not isinstance(quality, Mapping):
        raise ProductionAdapterError("quality必须是对象")

    requested_patient_id = _required_text(patient_id, "patient_id")
    patient_object_id = _required_text(
        _field(patient, "patient_id"), "SessionState.patient.patient_id"
    )
    if requested_patient_id != patient_object_id:
        raise ProductionAdapterError(
            "patient_id与SessionState.patient.patient_id不一致"
        )
    model_id = _required_text(report_model_id, "report_model_id")

    missing_predictions = [
        key
        for key in ("FMA_UE", "hand_tone", "hand_function")
        if key not in predictions_raw or predictions_raw.get(key) is None
    ]
    if missing_predictions:
        raise ProductionAdapterError(
            "predictions_raw缺少关键字段：" + "、".join(missing_predictions)
        )

    try:
        predictions = CanonicalPredictions(
            FMA_UE=predictions_raw.get("FMA_UE"),
            hand_tone=str(predictions_raw.get("hand_tone")),
            hand_function=predictions_raw.get("hand_function"),
        )
        pipeline_patient = PipelinePatientInput(
            patient_id=patient_object_id,
            age=_field(patient, "age"),
            sex=_optional_text(_field(patient, "sex")),
            diagnosis=_optional_text(_field(patient, "diagnosis")),
            disease_days=_field(patient, "disease_days"),
            paralysis_side=_optional_text(_field(patient, "paralysis_side")),
        )
        input_data: Dict[str, Any] = {
            "assessment_id": _optional_text(assessment_id),
            "patient": pipeline_patient,
            "predictions": predictions,
            "biomarkers": _canonical_biomarkers(biomarkers),
            "quality_metadata": dict(quality),
        }
        if context_id is not None:
            input_data["context_id"] = _required_text(context_id, "context_id")
        assessment_input = PipelineAssessmentInput(**input_data)
    except ValidationError as exc:
        raise ProductionAdapterError(
            f"生产评估数据不符合PipelineAssessmentInput契约：{exc}"
        ) from exc

    return ProductionPipelineRequest(
        assessment_input=assessment_input,
        report_model_id=model_id,
    )


def build_production_orchestrator(report_model_id: str) -> ClinicalPipelineOrchestrator:
    """Build independent Planner and ReportGenerator roles on the selected LLM."""
    model_id = _required_text(report_model_id, "report_model_id")
    collection_id = knowledge_admin.active_collection_id()
    graph_mode = os.environ.get("CLINICAL_KG_MODE", "llm_only").strip().lower()
    if graph_mode not in {"llm_only", "graph_enhanced"}:
        raise ProductionAdapterError(
            "CLINICAL_KG_MODE只支持llm_only或graph_enhanced"
        )
    config = PipelineConfig(
        config_version="planner_rag-v0.1-production",
        core_knowledge=CoreKnowledgeConfig(bundle_version=collection_id),
        planner=LlmRoleConfig(model_id=model_id),
        knowledge_graph=KnowledgeGraphConfig(mode=graph_mode),
        report_generator=LlmRoleConfig(model_id=model_id),
    )
    return ClinicalPipelineOrchestrator(
        config=config,
        knowledge_planner=KnowledgePlanner(ExistingLlmClient(model_id=model_id)),
        report_generator=ReportGenerator(
            ExistingReportLlmClient(model_id=model_id),
            prefer_compact_contract=True,
        ),
    )


def require_completed_report(
    result: OrchestrationResult,
) -> tuple[ReportResult, ValidationResult]:
    """Return a completed report or raise a production-facing explicit error."""
    if result.status == PipelineRunStatus.BLOCKED:
        reasons = "；".join(result.block_reasons) or "QualityGate未提供阻断原因"
        raise ProductionPipelineBlockedError(f"planner_rag质量门控阻断：{reasons}")
    if result.status == PipelineRunStatus.FAILED:
        if result.failure is None:
            detail = "未记录失败阶段"
        else:
            detail = f"{result.failure.module.value}：{result.failure.message}"
        raise ProductionPipelineExecutionError(f"planner_rag执行失败：{detail}")
    if result.report is None or result.validation is None:
        raise ProductionPipelineExecutionError(
            "planner_rag执行完成但缺少ReportResult或ValidationResult"
        )
    return result.report, result.validation


def recover_report_generator_failure(
    result: OrchestrationResult,
    *,
    report_model_id: str,
) -> tuple[OrchestrationResult, ValidationResult]:
    """Recover only a ReportGenerator JSON-contract failure.

    Earlier stages (including graph-enhanced non-IMU filtering, planner and
    retriever) have completed at this point.  The recovery therefore builds a
    conservative report from that exact ``ReportGenerationInput`` instead of
    routing back through the legacy report path, which could have a different
    modality scope.
    """
    failure = result.failure
    if (
        result.status != PipelineRunStatus.FAILED
        or failure is None
        or failure.module.value != "ReportGenerator"
        or result.report_input is None
    ):
        raise ProductionPipelineExecutionError(
            "仅ReportGenerator失败且保留ReportInput时允许保守报告回退"
        )

    try:
        report = build_llm_strategy_recovery_report(
            result.report_input,
            model_id=report_model_id,
            attempt=2,
        )
        recovery_mode = "llm_strategy_recovery"
    except Exception as recovery_exc:  # noqa: BLE001 - final availability guard
        report = build_conservative_report(
            result.report_input,
            model_id=report_model_id,
        )
        recovery_mode = "conservative_structured_fallback"
        result.trace.artifact_refs["llm_strategy_recovery_error_type"] = type(
            recovery_exc
        ).__name__
        if isinstance(recovery_exc, LlmStrategyContractError):
            result.trace.artifact_refs["llm_strategy_recovery_error_code"] = (
                recovery_exc.code
            )
    validation = Validator().validate(report, result.report_input)
    result.trace.artifact_refs["report_generator_recovery"] = recovery_mode
    result.trace.artifact_refs["report_generator_failure"] = failure.message
    result.trace.artifact_refs["validation_result"] = validation.validation_id
    result.trace.artifact_refs["validation_status"] = validation.status.value
    recovered = result.model_copy(
        update={
            "status": PipelineRunStatus.COMPLETED,
            "report": report,
            "validation": validation,
            "failure": None,
        }
    )
    return recovered, validation


_GRAPH_DISPLAY_PATH_LIMIT = 24
_GRAPH_INDICATOR_NODE_TYPES = {
    "EEGBiomarker",
    "EMGBiomarker",
}


def _graph_display_value(state: Mapping[str, Any]) -> str:
    value = state.get("value")
    if value is None:
        return "未获得可用值"
    if isinstance(value, float):
        value_text = format(value, ".6g")
    else:
        value_text = _one_line(value)
    unit = _one_line(state.get("unit"))
    return f"{value_text} {unit}".strip()


def _graph_path_node(
    nodes: list[Mapping[str, Any]],
    node_type: str | set[str],
) -> Mapping[str, Any]:
    types = {node_type} if isinstance(node_type, str) else node_type
    return next(
        (node for node in nodes if str(node.get("node_type")) in types),
        {},
    )


def _knowledge_graph_display(result: OrchestrationResult) -> Dict[str, Any]:
    """Build a bounded, de-identified package for the web relationship view."""
    mode = result.trace.artifact_refs.get("knowledge_graph_mode", "llm_only")
    status = result.trace.artifact_refs.get("knowledge_graph_status", "disabled")
    scope = result.trace.artifact_refs.get(
        "knowledge_graph_non_imu_scope", "not_applied"
    )
    evidence = result.knowledge_graph_evidence or {}
    indicator_states = list(evidence.get("matched_indicator_states") or [])
    state_by_id = {
        str(state.get("state_id")): state
        for state in indicator_states
        if state.get("state_id")
    }

    raw_paths = list(evidence.get("matched_graph_paths") or [])
    display_paths: list[Dict[str, Any]] = []
    for path in raw_paths[:_GRAPH_DISPLAY_PATH_LIMIT]:
        nodes = [
            node for node in (path.get("node_path") or []) if isinstance(node, Mapping)
        ]
        state_node = _graph_path_node(nodes, "IndicatorState")
        state = state_by_id.get(str(state_node.get("node_id")), {})
        indicator = _graph_path_node(nodes, _GRAPH_INDICATOR_NODE_TYPES)
        functional = _graph_path_node(nodes, "FunctionalFinding")
        dimension = _graph_path_node(nodes, "ClinicalDimension")
        topic = _graph_path_node(nodes, "EvidenceTopic")
        if not all((indicator, functional, dimension, topic)):
            continue
        relations = [
            str(relation.get("type"))
            for relation in (path.get("relation_path") or [])
            if isinstance(relation, Mapping) and relation.get("type")
        ]
        display_paths.append(
            {
                "path_id": str(path.get("path_id") or f"path:{len(display_paths) + 1}"),
                "source_field": _one_line(
                    state.get("source_field") or path.get("source_field")
                ),
                "source": {
                    "id": str(state_node.get("node_id") or path.get("source_field")),
                    "label": _one_line(
                        state.get("label") or indicator.get("label") or "本次指标"
                    ),
                    "value": _graph_display_value(state),
                    "state": _one_line(state.get("state") or state_node.get("state")),
                    "modality": _one_line(state.get("source_modality")) or "clinical_scale",
                },
                "indicator": {
                    "id": str(indicator.get("node_id") or ""),
                    "label": _one_line(indicator.get("label") or "图谱指标"),
                    "type": str(indicator.get("node_type") or ""),
                },
                "functional_finding": {
                    "id": str(functional.get("node_id") or ""),
                    "label": _one_line(functional.get("label") or "功能表现"),
                },
                "clinical_dimension": {
                    "id": str(dimension.get("node_id") or ""),
                    "label": _one_line(dimension.get("label") or "临床维度"),
                },
                "rag_topic": {
                    "id": str(topic.get("node_id") or ""),
                    "label": _one_line(topic.get("label") or "RAG检索主题"),
                },
                "relations": relations,
            }
        )

    final_topics: list[Dict[str, Any]] = []
    if result.knowledge_plan is not None:
        for topic in result.knowledge_plan.topics[:20]:
            origins = list(
                dict.fromkeys(
                    result.knowledge_plan.topic_origins.get(topic.topic_id)
                    or ["planner"]
                )
            )
            final_topics.append(
                {
                    "topic_id": topic.topic_id,
                    "label": topic.label,
                    "priority": topic.priority,
                    "origins": origins,
                }
            )

    rule_results = [
        {
            "rule_id": str(item.get("rule_id") or ""),
            "status": _one_line((item.get("result") or {}).get("status")),
            "message": _one_line((item.get("result") or {}).get("message")),
            "evidence_level": _one_line(item.get("evidence_level")),
            "expert_review_status": _one_line(item.get("expert_review_status")),
        }
        for item in (evidence.get("rule_results") or [])
        if item.get("matched")
    ]
    dimensions = [
        {
            "dimension_id": str(item.get("dimension_id") or ""),
            "label": _one_line(item.get("label") or "临床维度"),
        }
        for item in (evidence.get("analysis_dimensions") or [])
    ]
    graph_seeded_topic_count = sum(
        1
        for topic in final_topics
        if any(str(origin).startswith("graph:") for origin in topic["origins"])
    )

    # MVP：预测结果区（运行时从 CanonicalPredictions 读取，不写入静态图谱）。
    prediction_results: list[Dict[str, Any]] = []
    predictions = (
        result.canonical_context.predictions
        if result.canonical_context is not None
        else None
    )
    if predictions is not None:
        prediction_results = [
            {
                "target_id": "target:FMA_hand",
                "target_label": "手的 Fugl-Meyer 评分（手部子量表 0-20）",
                "value": predictions.FMA_UE,
                "value_text": _fma_score_text(predictions.FMA_UE),
                "range": "0-20",
                "range_note": "手部子量表，非完整 FMA-UE 0-66",
                "model_label": "FMA 手部子量表模型",
                "is_model_prediction": True,
            },
            {
                "target_id": "target:MAS_hand",
                "target_label": "手部肌张力（MAS 分级）",
                "value": predictions.hand_tone,
                "value_text": _one_line(predictions.hand_tone) if predictions.hand_tone is not None else "未获得可用值",
                "range": "0,1,1+,2,3,4",
                "range_note": "Modified Ashworth Scale 手部肌张力",
                "model_label": "MAS 手部肌张力模型",
                "is_model_prediction": True,
            },
            {
                "target_id": "target:Brunnstrom_hand",
                "target_label": "手的布氏分期（Brunnstrom 手功能）",
                "value": predictions.hand_function,
                "value_text": (
                    _one_line(str(predictions.hand_function))
                    if predictions.hand_function is not None
                    else "未获得可用值"
                ),
                "range": "2-6",
                "range_note": "Brunnstrom 量表理论分期为 1-6 期；当前模型分类范围为 2-6 期。",
                "model_label": "Brunnstrom 手功能分期模型",
                "is_model_prediction": True,
            },
        ]

    # MVP：数据质量区（独立于临床维度；沿用 inference 质量字段，无 blocked）。
    quality_meta = dict(evidence.get("patient_summary") or {}).get("quality") or {}
    data_quality = {
        "status": _one_line(quality_meta.get("status")) or "unknown",
        "trial_count": quality_meta.get("trial_count"),
        "short_trial_count": quality_meta.get("short_trial_count"),
        "sync_fallback_count": quality_meta.get("sync_fallback_count"),
        "sampling_rate_mismatch_count": quality_meta.get("sampling_rate_mismatch_count"),
        "warnings": [
            {
                "code": _one_line(item.get("code")),
                "message": _one_line(item.get("message")),
            }
            for item in (evidence.get("data_quality_warnings") or [])
            if isinstance(item, dict)
        ],
        "is_clinical_dimension": False,
        "blocked_support": False,
    }

    return {
        "schema_version": "rehab.knowledge-graph-display.v1",
        "mode": mode,
        "status": status,
        "scope": scope,
        "applied": bool(
            mode == "graph_enhanced"
            and status == "matched"
            and scope == "applied"
        ),
        "prototype_notice": "研究原型关系；表示本次分析与检索路径，不表达诊断或确定因果。",
        "expert_review_status": evidence.get("expert_review_status") or "pending",
        "summary": {
            "indicator_count": len(indicator_states),
            "path_count": len(raw_paths),
            "displayed_path_count": len(display_paths),
            "dimension_count": len(dimensions),
            "final_topic_count": len(final_topics),
            "graph_seeded_topic_count": graph_seeded_topic_count,
            "matched_rule_count": len(rule_results),
            "retrieval_status": (
                result.retrieval.status.value if result.retrieval else None
            ),
            "retrieval_evidence_count": (
                len(result.retrieval.evidence) if result.retrieval else 0
            ),
        },
        "paths": display_paths,
        "paths_truncated": len(raw_paths) > len(display_paths),
        "analysis_dimensions": dimensions,
        "final_topics": final_topics,
        "matched_rules": rule_results,
        "data_quality_warnings": list(evidence.get("data_quality_warnings") or []),
        "measurement_context_topics": [
            {
                "topic_id": str(item.get("topic_id") or ""),
                "label": _one_line(item.get("label") or "测量条件与解释边界"),
            }
            for item in (evidence.get("measurement_context_topics") or [])
        ],
        "prediction_results": prediction_results,
        "data_quality": data_quality,
    }


def orchestration_metadata(result: OrchestrationResult) -> Dict[str, Any]:
    """Small in-band audit summary used before a dedicated DB column exists."""
    report, validation = require_completed_report(result)
    return {
        "mode": "planner_rag",
        "run_id": result.trace.run_id,
        "run_status": result.status.value,
        "quality_gate": result.quality_gate.decision.value if result.quality_gate else None,
        "planner_generation_mode": (
            result.knowledge_plan.generation_mode if result.knowledge_plan else None
        ),
        "knowledge_graph_mode": result.trace.artifact_refs.get(
            "knowledge_graph_mode", "llm_only"
        ),
        "knowledge_graph_status": result.trace.artifact_refs.get(
            "knowledge_graph_status", "disabled"
        ),
        "knowledge_graph_topic_count": result.trace.artifact_refs.get(
            "knowledge_graph_topic_count", "0"
        ),
        "knowledge_graph_measurement_context_topic_count": result.trace.artifact_refs.get(
            "knowledge_graph_measurement_context_topic_count", "0"
        ),
        "knowledge_graph_non_imu_scope": result.trace.artifact_refs.get(
            "knowledge_graph_non_imu_scope", "not_applied"
        ),
        "retrieval_status": result.retrieval.status.value if result.retrieval else None,
        "report_id": report.report_id,
        "report_generation_mode": report.generation_mode,
        "report_generator_recovery": result.trace.artifact_refs.get(
            "report_generator_recovery", "none"
        ),
        "report_generator_llm_error_code": result.trace.artifact_refs.get(
            "llm_strategy_recovery_error_code"
        ),
        "validation_status": validation.status.value,
        "knowledge_graph_display": _knowledge_graph_display(result),
    }


def _dedup_recommendations(items: list[str]) -> list[str]:
    """Remove near-duplicate recommendations by word overlap."""
    if len(items) <= 1:
        return items
    keep = []
    keep_word_sets = []
    for item in items:
        words = {w for w in item if len(w) >= 2}
        is_dup = False
        for prev in keep_word_sets:
            if not words or not prev:
                continue
            overlap = len(words & prev)
            smaller = min(len(words), len(prev))
            if smaller > 0 and overlap / smaller > 0.5:
                is_dup = True
                break
        if not is_dup:
            keep.append(item)
            keep_word_sets.append(words)
    return keep



def _one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _fma_score_text(value: Any) -> str:
    """FMA 手部子量表分数按整数展示（0-20 分，不出现小数点）。"""
    if value is None:
        return "未获得可用值"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _one_line(value)
    return str(int(round(number)))


def _table_cell(value: Any) -> str:
    return _one_line(value).replace("|", "\\|") or "—"




def _result_value_text(
    finding: Any,
    clinical_score_source: str = "dl_prediction",
) -> str:
    """Keep the result cell factual and separate from the explanation."""
    if finding is None or getattr(finding, "value", None) is None:
        return "未获得可用数据"
    modality = str(getattr(getattr(finding, "modality", None), "value", ""))
    prefix = (
        "医生提供的临床评定结果"
        if clinical_score_source == "clinician_provided" and modality == "clinical_scale"
        else "模型预测值"
        if modality == "clinical_scale"
        else "本次记录值"
    )
    unit = _one_line(getattr(finding, "unit", ""))
    suffix = f" {unit}" if unit else ""
    metric_key = str(getattr(finding, "metric_key", ""))
    displayed = _fma_score_text(finding.value) if metric_key == "FMA_UE" else _one_line(finding.value)
    return f"{prefix}：{displayed}{suffix}"


def _first_reading_sentence(value: Any) -> str:
    text = _one_line(value).removeprefix("模型预测结果：")
    return re.split(r"[。；]", text, maxsplit=1)[0].strip()


_METRIC_PURPOSES = {
    "resting_emg_level": "静息时肌肉是否仍有不必要的紧张",
    "wrist_co_contraction_index": "腕屈肌和伸肌是否同时用力、动作是否协调",
    "finger_co_contraction_index": "手指屈肌和伸肌是否同时用力、动作是否协调",
    "emg_activation_rms": "动作时肌肉募集的总体强弱",
    "fcr_iemg": "桡侧腕屈肌在整个动作中的总用力量",
    "fds_iemg": "指浅屈肌在整个动作中的总用力量",
    "ecu_iemg": "尺侧腕伸肌在整个动作中的总用力量",
    "extensor_digitorum_iemg": "指伸肌在整个动作中的总用力量",
    "flexor_extensor_iemg_ratio": "屈肌和伸肌出力是否平衡",
    "emg_burst_duration": "一次动作中肌肉持续发力的时长",
    "fcr_mdf": "桡侧腕屈肌的疲劳或募集变化",
    "fds_mdf": "指浅屈肌的疲劳或募集变化",
    "ecu_mdf": "尺侧腕伸肌的疲劳或募集变化",
    "extensor_digitorum_mdf": "指伸肌的疲劳或募集变化",
    "pathological_asymmetry_index": "两侧大脑静息活动是否平衡",
    "corticomuscular_coherence_beta": "大脑运动区和肌肉发力是否同步配合",
    "prefrontal_theta_beta_ratio": "前额叶与注意、任务控制相关的脑电活动比例",
    "interhemispheric_motor_coherence": "左右运动脑区之间的协同活动",
    "movement_mu_power_change": "动作时运动脑区的μ节律反应",
    "movement_beta_power_change": "动作时运动脑区的β节律反应",
    "movement_smoothness_sparc": "动作是否连续、流畅，是否频繁停顿或抖动",
    "range_of_motion_proxy": "本次动作活动范围的大小",
    "tremor_index_3_6hz": "动作中3–6Hz震颤成分的多少",
    "wrist_flexion_peak_velocity": "腕屈动作达到的最快速度",
    "wrist_extension_peak_velocity": "腕伸动作达到的最快速度",
    "finger_extension_peak_velocity": "伸指动作达到的最快速度",
}


def _metric_purpose(finding: Any) -> str:
    key = str(getattr(finding, "metric_key", ""))
    return _METRIC_PURPOSES.get(key, _one_line(getattr(finding, "name", "该指标")))


def _plain_interpretation_text(
    finding: Any,
    clinical_score_source: str = "dl_prediction",
) -> str:
    """Explain what the indicator measures before stating comparison limits."""
    if finding is None:
        return "本次数据已记录。"
    if getattr(finding, "value", None) is None:
        return f"用于观察{_metric_purpose(finding)}；本次没有可用数据，暂不作判断。"

    metric_key = str(getattr(finding, "metric_key", ""))
    modality = str(getattr(getattr(finding, "modality", None), "value", ""))
    if modality == "clinical_scale":
        if metric_key == "FMA_UE":
            return "反映手部动作完成情况；需结合现场动作检查确认。"
        reading = _first_reading_sentence(getattr(finding, "description", ""))
        if metric_key == "hand_tone":
            return f"{reading or '反映肌肉放松和阻力情况'}；需由治疗师实际检查确认。"
        if metric_key == "hand_function":
            return f"{reading or '反映手部动作恢复阶段'}；以实际抓握和伸指观察为准。"
        if clinical_score_source == "clinician_provided":
            return "这是医生提供的临床评定结果，仍需结合现场检查确认。"
        return "这是模型预测结果，需结合现场检查确认。"

    purpose = _metric_purpose(finding)
    result = _result_value_text(finding, clinical_score_source)
    observed_value = result.replace("本次记录值：", "记录值为")
    status = str(getattr(getattr(finding, "status", None), "value", ""))
    if status == "within_reference":
        return f"用于观察{purpose}；{result}在文献参考范围内，仍需结合动作表现判断。"
    if status == "above_reference":
        return f"用于观察{purpose}；{result}高于文献参考范围，需结合动作表现和复测判断。"
    if status == "below_reference":
        return f"用于观察{purpose}；{result}低于文献参考范围，需结合动作表现和复测判断。"

    reference = marker_ref(metric_key) or {}
    direction = {"increase": "升高", "decrease": "下降"}.get(
        str(reference.get("expected_direction") or "")
    )
    if status == "direction_only":
        trend = (
            f"研究通常关注同条件下是否{direction}"
            if direction
            else "研究通常关注同条件下的变化方向"
        )
        return (
            f"用于观察{purpose}；{observed_value}。{trend}，"
            "应结合既往同条件记录比较变化。"
        )
    if status in {"not_classifiable", "missing"}:
        return (
            f"用于观察{purpose}；{observed_value}。"
            "目前没有统一的单次判断范围，应结合既往同条件记录比较变化。"
        )
    return (
        f"用于观察{purpose}；{observed_value}。"
        "应结合既往同条件记录比较变化。"
    )


_BRUNNSTROM_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}

# Transcribed from the user-provided document
# "根据布氏分期的训练手势分类20260724.docx".  This is a controlled
# presentation list for the test report rather than a new classification rule.
_BRUNNSTROM_GESTURE_ACTIONS = {
    1: (("SS-15", "五指伸展"), ("SS-16", "五指屈曲")),
    2: (
        ("SS-15", "五指伸展"),
        ("SS-16", "五指屈曲"),
        ("SS-10", "拇指屈曲"),
        ("SS-18", "柱状抓握"),
    ),
    3: (
        ("SS-15", "五指伸展"),
        ("SS-16", "五指屈曲"),
        ("SS-11", "拇指竖起"),
        ("SS-12", "食中指伸展"),
        ("SS-14", "四指伸展"),
        ("SS-22", "球体抓握"),
    ),
    4: (
        ("SS-1", "食指屈曲"),
        ("SS-3", "中指屈曲"),
        ("SS-11", "拇指竖起"),
        ("SS-12", "食中指伸展"),
        ("SS-14", "四指伸展"),
        ("SS-19", "棍状物抓握"),
    ),
    5: (
        ("SS-19", "棍状物抓握"),
        ("SS-20", "食指伸展抓握"),
        ("SS-21", "环形抓握"),
        ("SS-22", "球体抓握"),
        ("SS-24", "拇指指尖捏取"),
    ),
}

_REVIEW_ONLY_RECOMMENDATION = re.compile(
    r"^(?:建议)?(?:由)?(?:康复)?(?:专业人员|治疗师|专家).{0,16}"
    r"(?:人工)?(?:复核|审核|确认)。?$"
)


def _brunnstrom_stage_number(result: OrchestrationResult) -> Optional[int]:
    """Return the observed hand-function stage only when it is 1 through 6."""
    findings = result.interpretation.findings if result.interpretation else []
    hand_function = next(
        (finding for finding in findings if finding.metric_key == "hand_function"),
        None,
    )
    if hand_function is None:
        return None
    try:
        stage_number = int(hand_function.value)
    except (TypeError, ValueError):
        return None
    return stage_number if stage_number in _BRUNNSTROM_ROMAN else None


def _fallback_overall_recommendations(stage_number: Optional[int]) -> list[str]:
    """Keep the test report useful if an LLM returns only a review request."""
    stage_text = (
        f"Brunnstrom {_BRUNNSTROM_ROMAN[stage_number]}期"
        if stage_number in _BRUNNSTROM_ROMAN
        else "本次手功能分期"
    )
    return [
        (
            f"训练重点：围绕{stage_text}的手部动作表现，优先选择能够稳定完成的动作，"
            "先保证手指打开、抓握和放开的质量，再逐步提高任务复杂度。"
        ),
        (
            "执行原则：练习中关注是否出现明显代偿、疼痛、张力增加或动作质量持续下降；"
            "出现时应降低当前难度或暂停该动作，不把完成次数作为唯一目标。"
        ),
        (
            "记录与调整：记录各动作能否完成、需要的辅助和完成质量；"
            "后续在同一设备和相近任务条件下复测，再据变化调整动作选择。"
        ),
    ]


def _overall_rehabilitation_recommendations(
    result: OrchestrationResult,
    report: ReportResult,
) -> list[str]:
    """Return detailed overall directions, excluding a review-only placeholder."""
    recommendations = _dedup_recommendations(
        [_one_line(item) for item in report.recommendations if _one_line(item)]
    )
    usable = [
        item
        for item in recommendations
        if not _REVIEW_ONLY_RECOMMENDATION.fullmatch(item)
    ]
    return usable or _fallback_overall_recommendations(
        _brunnstrom_stage_number(result)
    )


def _overall_subtype_text(
    result: OrchestrationResult,
    clinical_score_source: str = "dl_prediction",
) -> str:
    """Create the visible test-only overall subtype from pipeline observations.

    The planner_rag ReportGenerator owns narrative and strategy text, but its
    v0.1 contract did not carry the legacy overall-subtype field. This concise
    synthesis is deterministic and stays within the available model-predicted
    observations; it does not add a diagnosis or a treatment prescription.
    """
    findings = result.interpretation.findings if result.interpretation else []
    by_metric = {finding.metric_key: finding for finding in findings}
    hand_function = by_metric.get("hand_function")
    fma_hand = by_metric.get("FMA_UE")

    stage_number = _brunnstrom_stage_number(result)
    stage_prefix = (
        f"{_BRUNNSTROM_ROMAN[stage_number]}期"
        if stage_number in _BRUNNSTROM_ROMAN
        else "手功能分期待确认"
    )
    stage_detail = _one_line(
        hand_function.description if hand_function is not None else ""
    ) or (
        "本次未获得可用的手功能临床评定结果"
        if clinical_score_source == "clinician_provided"
        else "本次未获得可用的手功能模型预测结果"
    )

    fma_detail = ""
    if fma_hand is not None and fma_hand.value is not None:
        provenance = (
            "医生提供的临床评定结果"
            if clinical_score_source == "clinician_provided"
            else "模型预测值"
        )
        fma_detail = f"FMA手部子量表{provenance}为{_fma_score_text(fma_hand.value)}分；"

    return (
        f"{stage_prefix}-手功能综合亚型（测试性归纳）：{stage_detail}；"
        f"{fma_detail}"
        "中枢驱动、协同分离和关节活动度仍需结合动作检查确认。"
    )

def _ordered_citations(report: ReportResult) -> list[str]:
    values: list[str] = []
    for source_id in report.citations + [
        source_id
        for finding in report.findings
        for source_id in finding.citations
    ]:
        source_id = source_id.strip()
        if source_id and source_id not in values:
            values.append(source_id)
    return values


def render_compatible_markdown(
    *,
    patient: Any,
    result: OrchestrationResult,
    assessment_validation_status: str,
    quality: Mapping[str, Any],
) -> str:
    """Render ``ReportResult`` for the existing Markdown/SSE frontend surface."""
    report, _validation = require_completed_report(result)
    clinical_score_source = str(quality.get("clinical_score_source") or "dl_prediction")
    source_ids = _ordered_citations(report)
    citation_numbers = {value: index for index, value in enumerate(source_ids, start=1)}

    def markers(values: list[str]) -> str:
        return "".join(
            f"【{citation_numbers[value]}】"
            for value in values
            if value in citation_numbers
        )

    finding_names = {
        finding.finding_id: finding.name
        for finding in (result.interpretation.findings if result.interpretation else [])
    }
    source_findings = {
        finding.finding_id: finding
        for finding in (result.interpretation.findings if result.interpretation else [])
    }
    finding_modalities = {
        finding_id: finding.modality.value
        for finding_id, finding in source_findings.items()
    }
    source_details: Dict[str, Dict[str, str]] = {}
    if result.core_knowledge is not None:
        for entry in result.core_knowledge.entries:
            for source_id in entry.source_ids:
                source_details.setdefault(
                    source_id,
                    {
                        "knowledge_id": entry.knowledge_id,
                        "title": entry.system_key,
                    },
                )
    if result.retrieval is not None:
        for evidence in result.retrieval.evidence:
            for source_id in evidence.source_ids:
                source_details.setdefault(
                    source_id,
                    {
                        "knowledge_id": str(evidence.metadata.get("knowledge_id") or ""),
                        "title": str(evidence.metadata.get("title") or ""),
                    },
                )

    try:
        snapshot = knowledge_admin.load_snapshot()
    except Exception:  # noqa: BLE001 - references retain contract fallback details
        snapshot = None
    if snapshot is not None:
        for source in snapshot.sources:
            source_id = str(source.get("source_id") or "").strip()
            if source_id not in source_ids:
                continue
            detail = source_details.setdefault(source_id, {})
            detail["source_title"] = str(source.get("title") or "").strip()
            detail["year"] = str(source.get("year") or "").strip()
            detail["evidence_tier"] = str(
                source.get("evidence_tier") or ""
            ).strip()
            detail["url"] = str(source.get("url") or "").strip()

    lines = ["# 智能康复评估报告", ""]

    age = _field(patient, "age")
    disease_days = _field(patient, "disease_days")
    lines.extend([
        "## 一、患者基本信息",
        "",
        f"- 患者ID：{_table_cell(_field(patient, 'patient_id'))}",
        f"- 姓名：{_table_cell(_field(patient, 'name'))}",
        f"- 年龄/性别：{_table_cell(age) if age is not None else '—'}岁/{_table_cell(_field(patient, 'sex'))}",
        f"- 病程：{_table_cell(disease_days) if disease_days is not None else '—'}天",
        f"- 诊断信息：{_table_cell(_field(patient, 'diagnosis'))}，{_table_cell(_field(patient, 'paralysis_side'))}侧",
        "",
        "## 二、综合评估结果",
        "",
        f"**临床解读：** {_one_line(report.summary)}",
    ])
    modality_groups = [
        (
            "clinical_scale",
            "临床评定结果" if clinical_score_source == "clinician_provided" else "临床任务模型预测",
        ),
        ("emg", "肌电指标"),
        ("eeg", "脑电指标"),
        ("multimodal", "脑肌多模态指标"),
        ("imu", "运动学指标"),
    ]
    rendered_ids: set[str] = set()
    for modality, label in modality_groups:
        group = [
            finding
            for finding in report.findings
            if finding_modalities.get(finding.finding_id) == modality
        ]
        if not group:
            continue
        lines.extend([
            "",
            f"### {label}",
            "",
            "| 指标 | 本次结果 | 解读 | 依据 |",
            "|---|---|---|---|",
        ])
        for finding in group:
            rendered_ids.add(finding.finding_id)
            lines.append(
                "| "
                + " | ".join(
                    [
                        _table_cell(
                            finding_names.get(finding.finding_id)
                            or finding.finding_id
                        ),
                        _table_cell(_result_value_text(source_findings.get(finding.finding_id), clinical_score_source)),
                        _table_cell(_plain_interpretation_text(source_findings.get(finding.finding_id), clinical_score_source)),
                        markers(finding.citations) or "—",
                    ]
                )
                + " |"
            )
    for finding in report.findings:
        if finding.finding_id in rendered_ids:
            continue
        lines.extend([
            "",
            "| 指标 | 本次结果 | 解读 | 依据 |",
            "|---|---|---|---|",
            "| "
            + " | ".join([
                _table_cell(finding_names.get(finding.finding_id) or finding.finding_id),
                _table_cell(_result_value_text(source_findings.get(finding.finding_id), clinical_score_source)),
                _table_cell(_plain_interpretation_text(source_findings.get(finding.finding_id), clinical_score_source)),
                markers(finding.citations) or "—",
            ])
            + " |",
        ])

    overall_subtype = _overall_subtype_text(result, clinical_score_source)
    lines.extend([
        "",
        "## 三、综合亚型界定",
        "",
        f"**综合亚型：** {_one_line(overall_subtype)}",
    ])

    recommendations = _overall_rehabilitation_recommendations(result, report)
    stage_number = _brunnstrom_stage_number(result)
    lines.extend(["", "## 四、康复策略建议", "", "### 一、总体训练方向", ""])
    lines.extend(
        f"{index}. {_one_line(value)}"
        for index, value in enumerate(recommendations, start=1)
    )
    lines.extend(["", "### 二、按 Brunnstrom 分期的训练动作", ""])
    if stage_number is None:
        lines.append("本次未获得可用的 Brunnstrom 手功能分期，暂不展示分期动作清单。")
    elif stage_number == 6:
        lines.append(
            f"手功能{'临床评定结果' if clinical_score_source == 'clinician_provided' else '模型预测'}为 Brunnstrom VI期，提示分离运动能力接近正常；"
            "仍需结合实际动作表现选择训练任务。"
        )
        lines.extend([
            "- 精细操作：扣钮扣、捏取小物体、书写或使用工具。",
            "- 双手协调：拿取、转移和放置物品，观察两手配合、准确性和速度。",
            "- 任务反馈：记录代偿、疼痛和动作质量变化，必要时降低难度或暂停。",
        ])
    else:
        lines.append(
            f"当前{'临床评定结果' if clinical_score_source == 'clinician_provided' else '模型预测'}为 Brunnstrom {_BRUNNSTROM_ROMAN[stage_number]}期；"
            "可从以下动作中选择训练。"
        )
        lines.extend(
            f"- {code}：{name}"
            for code, name in _BRUNNSTROM_GESTURE_ACTIONS[stage_number]
        )
    lines.append("以下动作是按 Brunnstrom 分期整理的训练示例，需结合实际动作表现选择。")
    lines.extend(["", "## 五、进一步个体化所需信息", ""])
    lines.extend(
        f"{index}. {_one_line(value)}"
        for index, value in enumerate(report.limitations, start=1)
    )

    lines.extend(["", "## 六、依据来源与参考文献", ""])
    if not source_ids:
        lines.append("本次报告未引用外部检索来源。")
    else:
        for source_id in source_ids:
            detail = source_details.get(source_id, {})
            title = detail.get("source_title") or detail.get("title") or source_id
            metadata = [value for value in [
                detail.get("year"),
                (
                    f"证据等级 {detail['evidence_tier']}"
                    if detail.get("evidence_tier")
                    else ""
                ),
            ] if value]
            suffix = f"（{'，'.join(metadata)}）" if metadata else ""
            url = detail.get("url") or ""
            link = f" [原文链接]({url})" if url else ""
            lines.append(
                f"【{citation_numbers[source_id]}】{title}{suffix}{link}"
                f" · {source_id}"
            )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ProductionAdapterError",
    "ProductionPipelineBlockedError",
    "ProductionPipelineExecutionError",
    "ProductionPipelineRequest",
    "adapt_production_input",
    "build_production_orchestrator",
    "orchestration_metadata",
    "render_compatible_markdown",
    "require_completed_report",
]
