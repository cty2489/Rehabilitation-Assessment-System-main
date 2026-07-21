"""Production boundary for the frozen ``planner_rag`` v0.1 pipeline.

This module only adapts existing assessment objects, assembles the already
implemented pipeline, and renders its structured report for the legacy
Markdown/SSE surface. It contains no clinical classification rules.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Mapping, Optional

import knowledge_admin
from pydantic import ValidationError

from .config import CoreKnowledgeConfig, LlmRoleConfig, PipelineConfig
from .contracts import CanonicalBiomarker, CanonicalPredictions
from .knowledge_planner import ExistingLlmClient, KnowledgePlanner
from .orchestrator import (
    ClinicalPipelineOrchestrator,
    OrchestrationResult,
    PipelineAssessmentInput,
    PipelinePatientInput,
    PipelineRunStatus,
)
from .report_generator import ExistingReportLlmClient, ReportGenerator, ReportResult
from .validator import ValidationResult


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
    config = PipelineConfig(
        config_version="planner_rag-v0.1-production",
        core_knowledge=CoreKnowledgeConfig(bundle_version=collection_id),
        planner=LlmRoleConfig(model_id=model_id),
        report_generator=LlmRoleConfig(model_id=model_id),
    )
    return ClinicalPipelineOrchestrator(
        config=config,
        knowledge_planner=KnowledgePlanner(ExistingLlmClient(model_id=model_id)),
        report_generator=ReportGenerator(
            ExistingReportLlmClient(model_id=model_id)
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
        "retrieval_status": result.retrieval.status.value if result.retrieval else None,
        "report_id": report.report_id,
        "validation_status": validation.status.value,
    }


def _one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _table_cell(value: Any) -> str:
    return _one_line(value).replace("|", "\\|") or "—"


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
    finding_modalities = {
        finding.finding_id: finding.modality.value
        for finding in (result.interpretation.findings if result.interpretation else [])
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
        ("clinical_scale", "临床任务模型预测"),
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
            "| 指标 | 本次结果与知识解读 | 依据 |",
            "|---|---|---|",
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
                        _table_cell(finding.statement),
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
            "| 指标 | 本次结果与知识解读 | 依据 |",
            "|---|---|---|",
            "| "
            + " | ".join([
                _table_cell(finding_names.get(finding.finding_id) or finding.finding_id),
                _table_cell(finding.statement),
                markers(finding.citations) or "—",
            ])
            + " |",
        ])

    lines.extend(["", "## 三、康复策略建议", ""])
    lines.extend(
        f"{index}. {_one_line(value)}"
        for index, value in enumerate(report.recommendations, start=1)
    )
    lines.extend(["", "## 四、进一步个体化所需信息", ""])
    lines.extend(
        f"{index}. {_one_line(value)}"
        for index, value in enumerate(report.limitations, start=1)
    )

    lines.extend(["", "## 五、依据来源与参考文献", ""])
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
