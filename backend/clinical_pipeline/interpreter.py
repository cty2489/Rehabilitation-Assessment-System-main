"""Deterministic observations for ``planner_rag`` v0.1.

The Interpreter converts canonical values into traceable findings. It does not
diagnose, explain mechanisms, recommend rehabilitation, prescribe dose, or plan
knowledge retrieval.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

from biomarker_refs import marker_ref
from inference_readings import brunnstrom_reading, hand_tone_reading

from .contracts import (
    CanonicalAssessmentContext,
    CanonicalBiomarker,
    Finding,
    FindingBasis,
    FindingBasisKind,
    FindingModality,
    FindingSeverity,
    FindingStatus,
    InterpretationResult,
)


ReferenceLookup = Callable[[str], Optional[Dict[str, Any]]]

_MULTIMODAL_MARKERS = {"corticomuscular_coherence_beta"}


def _has_finite_value(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _prediction_finding(
    *,
    metric_key: str,
    name: str,
    value: Any,
    unit: str,
    description: str,
    basis_kind: FindingBasisKind,
    source_field: str,
) -> Finding:
    missing = value is None
    finding_description = (
        "本次未获得该项模型预测结果；不得据此推断医生实测结果。"
        if missing
        else f"模型预测结果：{description}。该结果不是医生实测结论。"
    )
    return Finding(
        finding_id=f"prediction:{metric_key}",
        metric_key=metric_key,
        name=name,
        value=value,
        unit=unit,
        status=FindingStatus.MISSING if missing else FindingStatus.OBSERVED,
        severity=FindingSeverity.UNKNOWN,
        modality=FindingModality.CLINICAL_SCALE,
        description=finding_description,
        basis=FindingBasis(
            kind=FindingBasisKind.MISSING_INPUT if missing else basis_kind,
            description=(
                "CanonicalAssessmentContext中该模型预测字段缺失。"
                if missing
                else "该观察项来自深度模型预测字段，不是医生实测记录。"
            ),
        ),
        source_field=source_field,
    )


def _reference_basis(
    kind: FindingBasisKind,
    description: str,
    ref: Optional[Dict[str, Any]],
    *,
    include_expected_direction: bool = True,
) -> FindingBasis:
    ref = ref or {}
    reference_type = ref.get("reference_type")
    if reference_type not in {"healthy_norm", "directional_trend", "none"}:
        reference_type = None
    return FindingBasis(
        kind=kind,
        description=description,
        reference_type=reference_type,
        absolute_comparison_applicable=bool(
            ref.get("absolute_comparison_applicable", False)
        ),
        lower_bound=ref.get("lo"),
        upper_bound=ref.get("hi"),
        expected_direction=(
            ref.get("expected_direction") if include_expected_direction else None
        ),
        source_ids=[str(value) for value in ref.get("source", []) if str(value)],
    )


def _biomarker_finding(
    marker: CanonicalBiomarker,
    reference_lookup: ReferenceLookup,
) -> Finding:
    source_field = f"biomarkers.{marker.metric_key}.value"
    modality = (
        FindingModality.MULTIMODAL
        if marker.metric_key in _MULTIMODAL_MARKERS
        else FindingModality(marker.modality)
    )
    if not marker.available or marker.n_valid < 1 or not _has_finite_value(marker.value):
        description = "本次未获得可用数值。"
        return Finding(
            finding_id=f"biomarker:{marker.metric_key}",
            metric_key=marker.metric_key,
            name=marker.name,
            value=None,
            unit=marker.unit,
            status=FindingStatus.MISSING,
            severity=FindingSeverity.UNKNOWN,
            modality=modality,
            description=description,
            basis=_reference_basis(
                FindingBasisKind.MISSING_INPUT,
                "CanonicalAssessmentContext中该指标不可用或数值缺失。",
                None,
            ),
            source_field=source_field,
        )

    ref = reference_lookup(marker.metric_key)
    unit = marker.unit or (str(ref.get("units")) if ref and ref.get("units") else None)
    if (
        ref
        and ref.get("absolute_comparison_applicable")
        and (ref.get("lo") is not None or ref.get("hi") is not None)
    ):
        lo = ref.get("lo")
        hi = ref.get("hi")
        if lo is not None and marker.value < lo:
            status = FindingStatus.BELOW_REFERENCE
            description = "本次观测值低于现有参考范围。"
        elif hi is not None and marker.value > hi:
            status = FindingStatus.ABOVE_REFERENCE
            description = "本次观测值高于现有参考范围。"
        else:
            status = FindingStatus.WITHIN_REFERENCE
            description = "本次观测值处于现有参考范围内。"
        basis = _reference_basis(
            FindingBasisKind.ABSOLUTE_REFERENCE_RANGE,
            "现有参考元数据允许进行绝对范围比较。",
            ref,
        )
    elif ref and ref.get("reference_type") == "directional_trend":
        status = FindingStatus.DIRECTION_ONLY
        description = (
            "该指标仅适用于同设备、同流程、同条件的纵向复测比较；"
            "当前输入未提供历史测量，本次不作变化方向判断。"
        )
        basis = _reference_basis(
            FindingBasisKind.DIRECTIONAL_TREND,
            "现有参考元数据没有可靠绝对范围，且当前没有历史测量可供比较。",
            ref,
            include_expected_direction=False,
        )
    else:
        status = FindingStatus.NOT_CLASSIFIABLE
        description = "该指标没有可用于单次分类的可靠参考范围。"
        basis = _reference_basis(
            FindingBasisKind.NO_RELIABLE_REFERENCE,
            "现有参考元数据不支持单次正常或异常分类。",
            ref,
        )

    return Finding(
        finding_id=f"biomarker:{marker.metric_key}",
        metric_key=marker.metric_key,
        name=marker.name,
        value=marker.value,
        unit=unit,
        status=status,
        severity=FindingSeverity.UNKNOWN,
        modality=modality,
        description=description,
        basis=basis,
        source_field=source_field,
    )


class Interpreter:
    """Convert one canonical assessment into observation-only findings."""

    def __init__(self, reference_lookup: ReferenceLookup = marker_ref):
        self._reference_lookup = reference_lookup

    def interpret(self, context: CanonicalAssessmentContext) -> InterpretationResult:
        if not isinstance(context, CanonicalAssessmentContext):
            raise TypeError("Interpreter input must be CanonicalAssessmentContext")

        predictions = context.predictions
        findings = [
            _prediction_finding(
                metric_key="FMA_UE",
                name="FMA手部子量表，范围0–20",
                value=predictions.FMA_UE,
                unit="分",
                description="FMA手部子量表预测值，量表范围为0–20分",
                basis_kind=FindingBasisKind.SCALE_DEFINITION,
                source_field="predictions.FMA_UE",
            ),
            _prediction_finding(
                metric_key="hand_tone",
                name="手部肌张力（Hand MAS，模型预测）",
                value=predictions.hand_tone,
                unit="级",
                description=hand_tone_reading(predictions.hand_tone),
                basis_kind=FindingBasisKind.SCALE_READING,
                source_field="predictions.hand_tone",
            ),
            _prediction_finding(
                metric_key="hand_function",
                name="Brunnstrom手功能分期（模型预测）",
                value=predictions.hand_function,
                unit="期",
                description=brunnstrom_reading(predictions.hand_function),
                basis_kind=FindingBasisKind.SCALE_READING,
                source_field="predictions.hand_function",
            ),
        ]
        findings.extend(
            _biomarker_finding(marker, self._reference_lookup)
            for marker in context.biomarkers
        )

        # No clinically validated combination rules are configured in v0.1.
        return InterpretationResult(findings=findings, known_combinations=[])


__all__ = ["Interpreter", "ReferenceLookup"]
