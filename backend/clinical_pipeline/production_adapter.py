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
from biomarker_refs import marker_ref
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


def _table_cell(value: Any) -> str:
    return _one_line(value).replace("|", "\\|") or "—"




def _result_value_text(finding: Any) -> str:
    """Keep the result cell factual and separate from the explanation."""
    if finding is None or getattr(finding, "value", None) is None:
        return "未获得可用数据"
    modality = str(getattr(getattr(finding, "modality", None), "value", ""))
    prefix = "模型预测值" if modality == "clinical_scale" else "本次记录值"
    unit = _one_line(getattr(finding, "unit", ""))
    suffix = f" {unit}" if unit else ""
    return f"{prefix}：{_one_line(finding.value)}{suffix}"


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


def _plain_interpretation_text(finding: Any) -> str:
    """Explain what the indicator measures before stating comparison limits."""
    if finding is None:
        return "本次数据已记录，建议结合同条件复测看变化。"
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
        return "这是模型预测结果，需结合现场检查确认。"

    purpose = _metric_purpose(finding)
    result = _result_value_text(finding)
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
        trend = f"研究通常看同条件下是否{direction}" if direction else "研究通常看同条件下的变化方向"
        return f"用于观察{purpose}；{result}是本次记录，{trend}，本次先作为个人基线。"
    if status in {"not_classifiable", "missing"}:
        return (
            f"用于观察{purpose}；{result}是本设备/算法算出的本次记录，"
            "目前没有统一的好坏范围，后续同条件复测看变化。"
        )
    return f"用于观察{purpose}；{result}先作为本次基线，后续同条件复测看变化。"


_BRUNNSTROM_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}


def _overall_subtype_text(result: OrchestrationResult) -> str:
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

    stage_number: Optional[int] = None
    if hand_function is not None:
        try:
            stage_number = int(hand_function.value)
        except (TypeError, ValueError):
            stage_number = None
    stage_prefix = (
        f"{_BRUNNSTROM_ROMAN[stage_number]}期"
        if stage_number in _BRUNNSTROM_ROMAN
        else "手功能分期待确认"
    )
    stage_detail = _one_line(
        hand_function.description if hand_function is not None else ""
    ) or "本次未获得可用的手功能模型预测结果"

    fma_detail = ""
    if fma_hand is not None and fma_hand.value is not None:
        fma_detail = f"FMA手部子量表模型预测值为{_one_line(fma_hand.value)}分；"

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
                        _table_cell(_result_value_text(source_findings.get(finding.finding_id))),
                        _table_cell(_plain_interpretation_text(source_findings.get(finding.finding_id))),
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
                _table_cell(_result_value_text(source_findings.get(finding.finding_id))),
                _table_cell(_plain_interpretation_text(source_findings.get(finding.finding_id))),
                markers(finding.citations) or "—",
            ])
            + " |",
        ])

    overall_subtype = _overall_subtype_text(result)
    lines.extend([
        "",
        "## 三、综合亚型界定",
        "",
        f"**综合亚型：** {_one_line(overall_subtype)}",
    ])

    deduped_recommendations = _dedup_recommendations([r for r in report.recommendations if r])
    lines.extend(["", "## 四、康复策略建议", ""])
    lines.extend(
        f"{index}. {_one_line(value)}"
        for index, value in enumerate(deduped_recommendations, start=1)
    )
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
