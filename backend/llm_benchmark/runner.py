"""Pure benchmark preparation helpers; no model/RAG/KG call is made here."""
from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional, Sequence

from pydantic import ValidationError

from .prompt import PROMPT_VERSION, build_messages
from .schemas import (
    BenchmarkBiomarker,
    BenchmarkClinicalScores,
    BenchmarkEvaluationInput,
    BenchmarkExperimentLog,
    BenchmarkLlmOutput,
    BenchmarkRunConfig,
    ClinicalScoreSource,
)


class BenchmarkOutputParseError(ValueError):
    """Raised when a model response cannot satisfy the fixed output contract."""


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _marker_rows(biomarkers: Mapping[str, Any]) -> list[BenchmarkBiomarker]:
    rows: list[BenchmarkBiomarker] = []
    for group in biomarkers.get("groups", []) or []:
        modality = str(group.get("key") or "").lower()
        if modality not in {"eeg", "emg", "imu"}:
            raise ValueError(f"不支持的biomarker modality：{modality}")
        for marker in group.get("markers", []) or []:
            rows.append(
                BenchmarkBiomarker(
                    metric_key=str(marker.get("key") or marker.get("metric_key") or ""),
                    name=str(marker.get("name") or marker.get("marker_name") or ""),
                    value=marker.get("value", marker.get("value_num")),
                    value_text=marker.get("value_text"),
                    unit=marker.get("unit"),
                    modality=modality,
                    available=bool(marker.get("available", True)),
                    clinical_usable=bool(
                        marker.get("clinical_usable", marker.get("available", True))
                    ),
                    n_valid=int(marker.get("n_valid", 0) or 0),
                )
            )
    if not rows:
        raise ValueError("biomarkers.groups未提供任何指标")
    return rows


def build_evaluation_input(
    *,
    patient: Any,
    clinical_scores: Any,
    biomarkers: Mapping[str, Any],
    quality: Optional[Mapping[str, Any]] = None,
    config: Optional[BenchmarkRunConfig] = None,
) -> BenchmarkEvaluationInput:
    """Adapt the existing manual or DL result objects without recalculating them."""
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
    if (
        run_config.assessment_input_mode.value == "dl_prediction"
        and scores.fma_wrist is not None
    ):
        raise ValueError("dl_prediction模式不能提供FMA腕；当前FMA腕只接受医生人工真值")
    quality_data = dict(quality or {})
    quality_status = "review" if quality_data.get("status") == "needs_review" else "pass"
    return BenchmarkEvaluationInput(
        config=run_config,
        patient=patient_model,
        clinical_scores=scores,
        clinical_score_source=run_config.clinical_score_source
        or (
            ClinicalScoreSource.CLINICIAN_PROVIDED
            if run_config.assessment_input_mode == "manual_clinical_scores"
            else ClinicalScoreSource.DL_PREDICTION
        ),
        biomarkers=_marker_rows(biomarkers),
        quality_status=quality_status,
        quality_metadata=quality_data,
    )


def _json_object_candidates(text: str) -> list[Any]:
    normalized = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S)
    normalized = re.sub(r"[\x60]{3}(?:json|JSON)?", "", normalized)
    normalized = normalized.replace("\x60\x60\x60", "")
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for index, char in enumerate(normalized):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(normalized[index:])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            candidates.append(value)
    return candidates


def _first_json_object(text: str) -> Any:
    candidates = _json_object_candidates(text)
    if candidates:
        return candidates[0]
    raise BenchmarkOutputParseError("模型输出未找到合法JSON对象")


def parse_model_output(raw_model_output: str) -> BenchmarkLlmOutput:
    if not isinstance(raw_model_output, str) or not raw_model_output.strip():
        raise BenchmarkOutputParseError("模型输出为空")
    try:
        return BenchmarkLlmOutput.model_validate(_first_json_object(raw_model_output))
    except (ValidationError, BenchmarkOutputParseError) as exc:
        raise BenchmarkOutputParseError(
            f"模型输出不符合rehab_llm_benchmark_v1：{exc}"
        ) from exc


def _display_marker(marker: BenchmarkBiomarker) -> str:
    if not marker.available:
        return "未获得可用数值"
    if marker.value_text:
        return marker.value_text
    if marker.value is None:
        return "—"
    return f"{format(marker.value, '.6g')}{marker.unit or ''}"


def render_report(
    evaluation_input: BenchmarkEvaluationInput,
    model_output: BenchmarkLlmOutput,
    *,
    evidence_cards: Sequence[Any] = (),
) -> str:
    """Render facts in code and insert only the three validated LLM fields."""
    scores = evaluation_input.clinical_scores
    patient = evaluation_input.patient
    lines = [
        "# 康复大模型评测报告",
        "",
        "## 一、基本信息与临床评定",
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
            "## 二、生物标志物分析",
            "",
            model_output.biomarker_interpretation,
            "",
            "## 三、综合康复评估",
            "",
            model_output.integrated_assessment,
            "",
            "## 四、康复建议",
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
            if card.page_start is None:
                page = "页码不可用"
            elif card.page_start == card.page_end:
                page = f"第{card.page_start}页"
            else:
                page = f"第{card.page_start}-{card.page_end}页"
            source = getattr(card, "source_file", None) or getattr(card, "source_pdf", None)
            source_label = source or "原文文件未登记"
            source_type = getattr(card, "source_type", "")
            lines.append(
                f"- {card.uid}｜{card.title}｜{source_type}｜{page}｜{source_label}｜查看原文"
            )
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
        prompt_version=config.prompt_version or PROMPT_VERSION,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        generation_time_ms=generation_time,
        generation_time=generation_time,
        raw_model_output=raw_model_output,
        parsed_model_output=(
            parsed_model_output.model_dump(mode="json")
            if parsed_model_output is not None
            else None
        ),
        parse_success=parsed_model_output is not None,
        biomarker_version=str(
            evaluation_input.quality_metadata.get("biomarker_version") or ""
        ) or None,
        biomarker_metadata=dict(
            evaluation_input.quality_metadata.get("biomarker_metadata") or {}
        ),
        error=error,
    )


__all__ = [
    "BenchmarkOutputParseError",
    "build_evaluation_input",
    "build_experiment_log",
    "build_messages",
    "parse_model_output",
    "render_report",
]
