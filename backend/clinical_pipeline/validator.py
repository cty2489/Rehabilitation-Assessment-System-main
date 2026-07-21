"""Deterministic safety validator for ``planner_rag`` v0.1 reports."""
from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal
from uuid import uuid4

from pydantic import Field

from .contracts import (
    ContractModel,
    FindingModality,
    ReportGenerationInput,
    RetrievalStatus,
    utc_now,
)
from .report_generator import ReportResult


class ValidationStatus(str, Enum):
    PASSED = "passed"
    WARNING = "warning"
    MANUAL_REVIEW = "manual_review"


class ValidationIssue(ContractModel):
    code: Literal[
        "evidence_limit_disclosed",
        "missing_evidence_limitation",
        "forged_source_id",
        "scale_prediction_not_disclosed",
        "deterministic_diagnosis",
        "unsupported_mechanism_conclusion",
        "drug_recommendation",
        "exact_training_dose",
    ]
    level: Literal["warning", "manual_review"]
    message: str = Field(min_length=1)
    details: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(ContractModel):
    schema_version: Literal["rehab.report-validation.v1"] = (
        "rehab.report-validation.v1"
    )
    validation_id: str = Field(default_factory=lambda: f"validation-{uuid4().hex}")
    report_id: str = Field(min_length=1, max_length=128)
    report_input_id: str = Field(min_length=1, max_length=128)
    status: ValidationStatus
    issues: List[ValidationIssue] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=utc_now)


_LIMITATION_TEXT = {
    RetrievalStatus.PARTIAL: "证据覆盖不完整",
    RetrievalStatus.INSUFFICIENT: "证据不足",
    RetrievalStatus.UNAVAILABLE: "检索证据不可用",
}
_INLINE_SOURCE_ID = re.compile(r"(?<![A-Za-z0-9._:-])SRC-[A-Za-z0-9._:-]+")
_DETERMINISTIC_DIAGNOSIS = re.compile(
    r"(?:诊断|确诊)\s*(?:为|是|：|:)|(?:可确诊|已经确诊)",
    re.IGNORECASE,
)
_MECHANISM_CONCLUSION = re.compile(
    r"(?:病理机制|发病机制)\s*(?:为|是|：|:)"
    r"|(?:证明|表明).{0,20}(?:病理机制|发病机制)",
    re.IGNORECASE,
)
_DRUG_RECOMMENDATION = re.compile(
    r"(?:药物|用药|服药|服用|口服|注射|开具)",
    re.IGNORECASE,
)
_DRUG_ADVICE_IN_NARRATIVE = re.compile(
    r"(?<!不)(?:建议|应当|应该|可考虑|推荐|需要).{0,20}"
    r"(?:药物|用药|服药|服用|口服|注射|开具)",
    re.IGNORECASE,
)
_EXACT_TRAINING_DOSE = re.compile(
    r"(?:每日|每天|每周|每次|每组)\s*\d+(?:\.\d+)?\s*"
    r"(?:分钟|小时|次|组|周|月|%)?"
    r"|\d+(?:\.\d+)?\s*(?:分钟|小时|次|组|周|个月|%|％)",
    re.IGNORECASE,
)
_EXACT_TRAINING_ADVICE_IN_NARRATIVE = re.compile(
    r"(?:建议|应当|应该|可考虑|推荐).{0,60}"
    r"(?:(?:每日|每天|每周|每次|每组)\s*\d+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s*(?:分钟|小时|次|组|周|个月|%|％))",
    re.IGNORECASE,
)


def _narrative_text(report: ReportResult) -> str:
    return "\n".join(
        [report.summary, report.evidence_summary]
        + [finding.statement for finding in report.findings]
        + report.limitations
        + report.recommendations
    )


def _allowed_source_ids(report_input: ReportGenerationInput) -> set[str]:
    core_sources = {
        source_id.strip()
        for entry in report_input.core_knowledge.entries
        for source_id in entry.source_ids
        if source_id.strip()
    }
    retrieval_sources = {
        source_id.strip()
        for evidence in report_input.retrieval.evidence
        for source_id in evidence.source_ids
        if source_id.strip()
    }
    return core_sources | retrieval_sources


def _cited_source_ids(report: ReportResult) -> set[str]:
    structured = {
        source_id.strip()
        for source_id in report.citations
        + [
            source_id
            for finding in report.findings
            for source_id in finding.citations
        ]
        if source_id.strip()
    }
    return structured | set(_INLINE_SOURCE_ID.findall(_narrative_text(report)))


class Validator:
    """Apply only the frozen v0.1 structural and language-boundary checks."""

    def validate(
        self,
        report: ReportResult,
        report_input: ReportGenerationInput,
    ) -> ValidationResult:
        if not isinstance(report, ReportResult):
            raise TypeError("report必须是ReportResult")
        if not isinstance(report_input, ReportGenerationInput):
            raise TypeError("report_input必须是ReportGenerationInput")

        issues: list[ValidationIssue] = []
        self._check_citations(report, report_input, issues)
        self._check_evidence_limit(report, report_input, issues)
        self._check_scale_provenance(report, report_input, issues)
        self._check_prohibited_content(report, issues)

        if any(issue.level == "manual_review" for issue in issues):
            status = ValidationStatus.MANUAL_REVIEW
        elif issues:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASSED
        return ValidationResult(
            report_id=report.report_id,
            report_input_id=report_input.input_id,
            status=status,
            issues=issues,
        )

    @staticmethod
    def _check_citations(
        report: ReportResult,
        report_input: ReportGenerationInput,
        issues: list[ValidationIssue],
    ) -> None:
        allowed = _allowed_source_ids(report_input)
        cited = _cited_source_ids(report)
        unknown = sorted(cited - allowed)
        if unknown:
            issues.append(
                ValidationIssue(
                    code="forged_source_id",
                    level="manual_review",
                    message="报告引用了本次固定核心知识和Retriever均未提供的source_id。",
                    details={"source_ids": unknown},
                )
            )

    @staticmethod
    def _check_evidence_limit(
        report: ReportResult,
        report_input: ReportGenerationInput,
        issues: list[ValidationIssue],
    ) -> None:
        retrieval_status = report_input.retrieval.status
        required = _LIMITATION_TEXT.get(retrieval_status)
        if required is None:
            return
        limitation_text = "\n".join([report.evidence_summary, *report.limitations])
        if required in limitation_text:
            issues.append(
                ValidationIssue(
                    code="evidence_limit_disclosed",
                    level="warning",
                    message=f"报告已披露检索状态限制：{required}。",
                    details={"retrieval_status": retrieval_status.value},
                )
            )
            return
        issues.append(
            ValidationIssue(
                code="missing_evidence_limitation",
                level="manual_review",
                message=f"报告未明确披露检索状态限制：{required}。",
                details={"retrieval_status": retrieval_status.value},
            )
        )

    @staticmethod
    def _check_scale_provenance(
        report: ReportResult,
        report_input: ReportGenerationInput,
        issues: list[ValidationIssue],
    ) -> None:
        scale_ids = {
            finding.finding_id
            for finding in report_input.findings.findings
            if finding.modality == FindingModality.CLINICAL_SCALE
        }
        missing: list[str] = []
        if scale_ids and "模型预测" not in report.summary:
            missing.append("summary")
        missing.extend(
            finding.finding_id
            for finding in report.findings
            if finding.finding_id in scale_ids and "模型预测" not in finding.statement
        )
        if missing:
            issues.append(
                ValidationIssue(
                    code="scale_prediction_not_disclosed",
                    level="manual_review",
                    message="报告中的量表内容未明确标记为模型预测结果。",
                    details={"locations": missing},
                )
            )

    @staticmethod
    def _check_prohibited_content(
        report: ReportResult,
        issues: list[ValidationIssue],
    ) -> None:
        narrative = _narrative_text(report)
        if _DETERMINISTIC_DIAGNOSIS.search(narrative):
            issues.append(
                ValidationIssue(
                    code="deterministic_diagnosis",
                    level="manual_review",
                    message="报告出现确定性诊断表达。",
                )
            )
        if _MECHANISM_CONCLUSION.search(narrative):
            issues.append(
                ValidationIssue(
                    code="unsupported_mechanism_conclusion",
                    level="manual_review",
                    message=(
                        "报告出现机制结论；v0.1不进行逐句证据蕴含判断，"
                        "该内容必须人工复核。"
                    ),
                )
            )

        recommendations = "\n".join(report.recommendations)
        if (
            _DRUG_RECOMMENDATION.search(recommendations)
            or _DRUG_ADVICE_IN_NARRATIVE.search(narrative)
        ):
            issues.append(
                ValidationIssue(
                    code="drug_recommendation",
                    level="manual_review",
                    message="报告出现药物相关建议。",
                )
            )
        if (
            _EXACT_TRAINING_DOSE.search(recommendations)
            or _EXACT_TRAINING_ADVICE_IN_NARRATIVE.search(narrative)
        ):
            issues.append(
                ValidationIssue(
                    code="exact_training_dose",
                    level="manual_review",
                    message="报告出现精确训练频率、强度、时长或疗程。",
                )
            )


__all__ = [
    "ValidationIssue",
    "ValidationResult",
    "ValidationStatus",
    "Validator",
]
