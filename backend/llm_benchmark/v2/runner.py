"""Pure v2 input adaptation, parsing, and program-owned report rendering."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from pydantic import ValidationError

from ..runner import _display_marker, _field, _first_json_object, _marker_rows
from .prompt import PROMPT_VERSION, build_messages
from .schemas import (
    BenchmarkEvaluationInput,
    BenchmarkExperimentLog,
    BenchmarkLlmOutput,
    BenchmarkRunConfig,
    ClinicalScoreSource,
)


class BenchmarkOutputParseErrorV2(ValueError):
    """Raised when output cannot satisfy the v2 two-section contract."""


def build_evaluation_input(
    *,
    patient: Any,
    clinical_scores: Any,
    biomarkers: Mapping[str, Any],
    quality: Optional[Mapping[str, Any]] = None,
    config: Optional[BenchmarkRunConfig] = None,
) -> BenchmarkEvaluationInput:
    run_config = config or BenchmarkRunConfig()
    patient_model = {
        "patient_id": str(_field(patient, "patient_id") or ""),
        "name": str(_field(patient, "name") or ""),
        "sex": _field(patient, "sex"),
        "age": _field(patient, "age"),
        "diagnosis": _field(patient, "diagnosis"),
        "disease_days": _field(patient, "disease_days"),
        "paralysis_side": _field(patient, "paralysis_side"),
    }
    from ..schemas import BenchmarkClinicalScores, BenchmarkPatientInfo

    scores = BenchmarkClinicalScores(
        fma_wrist=_field(clinical_scores, "fma_wrist"),
        fma_hand=(
            _field(clinical_scores, "fma_hand")
            if _field(clinical_scores, "fma_hand") is not None
            else _field(clinical_scores, "FMA_UE")
        ),
        hand_mas=(
            _field(clinical_scores, "hand_mas")
            if _field(clinical_scores, "hand_mas") is not None
            else _field(clinical_scores, "hand_tone")
        ),
        brunnstrom_hand=(
            _field(clinical_scores, "brunnstrom_hand")
            if _field(clinical_scores, "brunnstrom_hand") is not None
            else _field(clinical_scores, "hand_function")
        ),
    )
    quality_data = dict(quality or {})
    return BenchmarkEvaluationInput(
        config=run_config,
        patient=BenchmarkPatientInfo(**patient_model),
        clinical_scores=scores,
        clinical_score_source=run_config.clinical_score_source or ClinicalScoreSource.CLINICIAN_PROVIDED,
        biomarkers=_marker_rows(biomarkers),
        quality_status="review" if quality_data.get("status") == "review" else "pass",
        quality_metadata=quality_data,
    )


def parse_model_output(raw_model_output: str) -> BenchmarkLlmOutput:
    if not isinstance(raw_model_output, str) or not raw_model_output.strip():
        raise BenchmarkOutputParseErrorV2("模型输出为空")
    try:
        value = _first_json_object(raw_model_output)
        return BenchmarkLlmOutput.model_validate(value)
    except (ValidationError, ValueError, TypeError) as exc:
        raise BenchmarkOutputParseErrorV2(
            f"模型输出不符合rehab_llm_benchmark_v2：{exc}"
        ) from exc


def render_report(
    evaluation_input: BenchmarkEvaluationInput,
    model_output: BenchmarkLlmOutput,
    *,
    evidence_cards: Sequence[Any] = (),
) -> str:
    """Render fixed facts plus exactly two generated report sections."""

    scores = evaluation_input.clinical_scores
    patient = evaluation_input.patient
    lines = [
        "# 康复大模型评测报告 v2",
        "",
        "## 基本信息与临床评定",
        "",
        f"- 患者编号：{patient.patient_id}",
        f"- 姓名：{patient.name}",
        f"- 性别：{patient.sex or '—'}",
        f"- 年龄：{patient.age if patient.age is not None else '—'}",
        f"- 诊断：{patient.diagnosis or '—'}",
        f"- 病程：{patient.disease_days if patient.disease_days is not None else '—'}天",
        f"- 偏瘫侧：{patient.paralysis_side or '—'}",
        "",
        "| 临床评定 | 程序记录值 |",
        "| --- | --- |",
        f"| FMA腕（fma_wrist） | {scores.fma_wrist if scores.fma_wrist is not None else '待补充'} |",
        f"| FMA手（fma_hand） | {scores.fma_hand:g}/20 |",
        f"| 手部MAS（hand_mas） | {scores.hand_mas}级 |",
        f"| Brunnstrom手功能（brunnstrom_hand） | {scores.brunnstrom_hand}期 |",
        "",
        "| biomarker原始名称 | 程序记录值 | 可用性 |",
        "| --- | --- | --- |",
    ]
    for marker in evaluation_input.biomarkers:
        lines.append(
            f"| {marker.name} | {_display_marker(marker)} | "
            f"{'可用' if marker.available else '不可用'} |"
        )
    lines.extend(
        [
            "",
            "## 一、综合康复评估",
            "",
            model_output.integrated_assessment,
            "",
            "## 二、康复建议",
            "",
        ]
    )
    for index, action in enumerate(model_output.rehabilitation_plan, start=1):
        lines.extend(
            [
                f"### 建议动作 {index}",
                f"- 动作：{action.action}",
                f"- 训练目的：{action.goal}",
                f"- 推荐原因：{action.reason}",
                f"- 必要注意事项：{action.precaution}",
                "",
            ]
        )
    lines.append(f"数据质量：{'需复核' if evaluation_input.quality_status == 'review' else '通过'}")
    cards = list(evidence_cards)
    if evaluation_input.config.rag_enabled and cards:
        lines.extend(["", "依据来源卡片："])
        for card in cards:
            source = getattr(card, "source_file", None) or getattr(card, "source_pdf", None)
            lines.append(f"- {card.uid}｜{card.title}｜{source or '原文文件未登记'}｜查看原文")
    return "\n".join(lines).rstrip() + "\n"


def build_experiment_log(
    *,
    evaluation_input: BenchmarkEvaluationInput,
    anonymous_report_id: str,
    llm_model_id: str,
    llm_model_name: str,
    raw_model_output: str,
    parsed_model_output: Optional[BenchmarkLlmOutput],
    temperature: Optional[float],
    max_new_tokens: Optional[int],
    generation_time: Optional[float],
    batch_id: Optional[str] = None,
    error: Optional[str] = None,
) -> BenchmarkExperimentLog:
    config = evaluation_input.config
    return BenchmarkExperimentLog(
        batch_id=batch_id,
        patient_id=evaluation_input.patient.patient_id,
        anonymous_report_id=anonymous_report_id,
        assessment_input_mode=config.assessment_input_mode,
        clinical_score_source=evaluation_input.clinical_score_source,
        fma_wrist=evaluation_input.clinical_scores.fma_wrist,
        fma_hand=evaluation_input.clinical_scores.fma_hand,
        hand_mas=evaluation_input.clinical_scores.hand_mas,
        brunnstrom_hand=evaluation_input.clinical_scores.brunnstrom_hand,
        llm_model_id=llm_model_id,
        llm_model_name=llm_model_name,
        rag_enabled=config.rag_enabled,
        rag_version=config.rag_version,
        knowledge_graph_enabled=config.knowledge_graph_enabled,
        kg_version=config.kg_version,
        prompt_version=PROMPT_VERSION,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        generation_time_ms=generation_time,
        generation_time=generation_time,
        raw_model_output=raw_model_output,
        parsed_model_output=(parsed_model_output.model_dump(mode="json") if parsed_model_output else None),
        parse_success=parsed_model_output is not None,
        biomarker_version=str(evaluation_input.quality_metadata.get("biomarker_version") or "") or None,
        biomarker_metadata=dict(evaluation_input.quality_metadata.get("biomarker_metadata") or {}),
        error=error,
    )


__all__ = [
    "BenchmarkOutputParseErrorV2",
    "build_evaluation_input",
    "build_experiment_log",
    "build_messages",
    "parse_model_output",
    "render_report",
]
