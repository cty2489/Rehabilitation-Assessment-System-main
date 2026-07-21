from __future__ import annotations

import unittest
from copy import deepcopy

from clinical_pipeline.contracts import (
    CoreKnowledgeBundle,
    CoreKnowledgeEntry,
    Finding,
    FindingBasis,
    FindingBasisKind,
    FindingModality,
    FindingStatus,
    InterpretationResult,
    KnowledgePlan,
    KnowledgeTopic,
    QualityDecision,
    ReportGenerationInput,
    RetrievalEvidence,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStatus,
)
from clinical_pipeline.report_generator import ReportResult
from clinical_pipeline.validator import ValidationStatus, Validator


def _finding() -> Finding:
    return Finding(
        finding_id="prediction:FMA_UE",
        metric_key="FMA_UE",
        name="FMA手部子量表，范围0–20",
        value=8,
        unit="分",
        status=FindingStatus.OBSERVED,
        modality=FindingModality.CLINICAL_SCALE,
        description="模型预测结果为8分，不是医生实测结论。",
        basis=FindingBasis(
            kind=FindingBasisKind.SCALE_READING,
            description="来源于深度模型预测字段。",
        ),
        source_field="predictions.FMA_UE",
    )


def _plan(topic_count: int) -> KnowledgePlan:
    return KnowledgePlan(
        planner_model_id="planner-test-model",
        topics=[
            KnowledgeTopic(
                topic_id=f"topic-{index}",
                label=f"测试主题{index}",
                finding_ids=["prediction:FMA_UE"],
            )
            for index in range(1, topic_count + 1)
        ],
        queries=[
            RetrievalQuery(
                query_id=f"query-{index}",
                topic_id=f"topic-{index}",
                text=f"测试查询{index}",
            )
            for index in range(1, topic_count + 1)
        ],
        reason="补充量表解释边界证据。",
    )


def _report_input(status: RetrievalStatus) -> ReportGenerationInput:
    topic_count = 2 if status == RetrievalStatus.PARTIAL else 1
    plan = _plan(topic_count)
    evidence = []
    if status in {RetrievalStatus.COMPLETE, RetrievalStatus.PARTIAL}:
        evidence = [
            RetrievalEvidence(
                evidence_id="evidence-1",
                query_id="query-1",
                chunk_id="KB-ASSESSMENT-001@1#001",
                text="量表结果需结合标准化临床评估解释。",
                rank=1,
                raw_score=0.82,
                source_ids=["SRC-001"],
                metadata={"knowledge_id": "KB-ASSESSMENT-001"},
            )
        ]
    covered = ["topic-1"] if evidence else []
    uncovered = [
        topic.topic_id for topic in plan.topics if topic.topic_id not in covered
    ]
    retrieval = RetrievalResult(
        attempt_id=f"attempt-{status.value}",
        status=status,
        queries=plan.queries,
        evidence=evidence,
        covered_topic_ids=covered,
        uncovered_topic_ids=uncovered,
    )
    return ReportGenerationInput(
        run_id=f"run-{status.value}",
        quality_decision=QualityDecision.PASS,
        findings=InterpretationResult(findings=[_finding()]),
        core_knowledge=CoreKnowledgeBundle(
            bundle_id="core-1",
            version="core-v1",
            entries=[
                CoreKnowledgeEntry(
                    knowledge_id="CORE-FMA-HAND",
                    system_key="FMA_UE",
                    allowed_interpretation="仅描述模型预测分数及量表范围。",
                    prohibited_interpretation="不得作为确定性诊断。",
                    source_ids=["SRC-CORE-001"],
                )
            ],
        ),
        knowledge_plan=plan,
        retrieval=retrieval,
        retrieval_barrier_call_id=retrieval.attempt_id,
    )


def _report(status: RetrievalStatus) -> ReportResult:
    citations = ["SRC-001"] if status in {
        RetrievalStatus.COMPLETE,
        RetrievalStatus.PARTIAL,
    } else []
    evidence_summary = {
        RetrievalStatus.COMPLETE: "现有证据覆盖本次知识主题。",
        RetrievalStatus.PARTIAL: "证据覆盖不完整，仍有主题未覆盖。",
        RetrievalStatus.INSUFFICIENT: "证据不足，未检索到合格来源。",
        RetrievalStatus.UNAVAILABLE: "检索证据不可用，本次未使用外部证据。",
    }[status]
    return ReportResult(
        report_id=f"report-{status.value}",
        report_model_id="report-test-model",
        summary="量表结果来自模型预测，当前仅作结构化观察描述。",
        findings=[
            {
                "finding_id": "prediction:FMA_UE",
                "statement": "模型预测的FMA手部子量表结果为8分。",
                "citations": citations,
            }
        ],
        evidence_summary=evidence_summary,
        limitations=["模型预测结果仍需结合标准化临床实测复核。"],
        recommendations=["建议由康复专业人员结合临床实测进行人工复核。"],
        citations=citations,
    )


def _replace_report(report: ReportResult, **changes) -> ReportResult:
    payload = deepcopy(report.model_dump())
    payload.update(changes)
    return ReportResult.model_validate(payload)


class ValidatorTests(unittest.TestCase):
    def test_complete_valid_report_is_passed(self) -> None:
        result = Validator().validate(
            _report(RetrievalStatus.COMPLETE),
            _report_input(RetrievalStatus.COMPLETE),
        )

        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.issues, [])

    def test_partial_with_disclosure_is_warning(self) -> None:
        result = Validator().validate(
            _report(RetrievalStatus.PARTIAL),
            _report_input(RetrievalStatus.PARTIAL),
        )

        self.assertEqual(result.status, ValidationStatus.WARNING)
        self.assertEqual([issue.code for issue in result.issues], [
            "evidence_limit_disclosed"
        ])

    def test_insufficient_with_disclosure_is_warning(self) -> None:
        result = Validator().validate(
            _report(RetrievalStatus.INSUFFICIENT),
            _report_input(RetrievalStatus.INSUFFICIENT),
        )

        self.assertEqual(result.status, ValidationStatus.WARNING)
        self.assertEqual(result.issues[0].details["retrieval_status"], "insufficient")

    def test_unavailable_without_fabricated_citation_is_warning(self) -> None:
        result = Validator().validate(
            _report(RetrievalStatus.UNAVAILABLE),
            _report_input(RetrievalStatus.UNAVAILABLE),
        )

        self.assertEqual(result.status, ValidationStatus.WARNING)
        self.assertEqual(result.issues[0].details["retrieval_status"], "unavailable")

    def test_core_knowledge_citation_is_valid_without_retrieval_evidence(self) -> None:
        report = _replace_report(
            _report(RetrievalStatus.UNAVAILABLE),
            citations=["SRC-CORE-001"],
            findings=[
                {
                    "finding_id": "prediction:FMA_UE",
                    "statement": "模型预测的FMA手部子量表结果为8分。",
                    "citations": ["SRC-CORE-001"],
                }
            ],
        )

        result = Validator().validate(
            report,
            _report_input(RetrievalStatus.UNAVAILABLE),
        )

        self.assertEqual(result.status, ValidationStatus.WARNING)
        self.assertNotIn("forged_source_id", [issue.code for issue in result.issues])

    def test_missing_evidence_limit_disclosure_is_manual_review(self) -> None:
        report = _replace_report(
            _report(RetrievalStatus.PARTIAL),
            evidence_summary="部分检索已经完成。",
        )

        result = Validator().validate(
            report,
            _report_input(RetrievalStatus.PARTIAL),
        )

        self.assertEqual(result.status, ValidationStatus.MANUAL_REVIEW)
        self.assertIn(
            "missing_evidence_limitation",
            [issue.code for issue in result.issues],
        )

    def test_scale_without_model_prediction_label_is_manual_review(self) -> None:
        report = _replace_report(
            _report(RetrievalStatus.COMPLETE),
            summary="当前量表结果仅作结构化观察描述。",
            findings=[
                {
                    "finding_id": "prediction:FMA_UE",
                    "statement": "FMA手部子量表结果为8分。",
                    "citations": ["SRC-001"],
                }
            ],
        )

        result = Validator().validate(
            report,
            _report_input(RetrievalStatus.COMPLETE),
        )

        self.assertEqual(result.status, ValidationStatus.MANUAL_REVIEW)
        self.assertIn(
            "scale_prediction_not_disclosed",
            [issue.code for issue in result.issues],
        )

    def test_unknown_citation_is_manual_review(self) -> None:
        report = _replace_report(
            _report(RetrievalStatus.COMPLETE),
            citations=["SRC-NOT-RETRIEVED"],
        )

        result = Validator().validate(
            report,
            _report_input(RetrievalStatus.COMPLETE),
        )

        self.assertEqual(result.status, ValidationStatus.MANUAL_REVIEW)
        self.assertIn("forged_source_id", [issue.code for issue in result.issues])

    def test_deterministic_diagnosis_is_manual_review(self) -> None:
        report = _replace_report(
            _report(RetrievalStatus.COMPLETE),
            summary="量表结果来自模型预测；诊断为脑卒中。",
        )

        result = Validator().validate(
            report,
            _report_input(RetrievalStatus.COMPLETE),
        )

        self.assertEqual(result.status, ValidationStatus.MANUAL_REVIEW)
        self.assertIn(
            "deterministic_diagnosis",
            [issue.code for issue in result.issues],
        )

    def test_negated_diagnosis_limitation_is_not_manual_review(self) -> None:
        report = _replace_report(
            _report(RetrievalStatus.INSUFFICIENT),
            summary=(
                "量表结果来自模型预测；当前证据不足以支持明确诊断。"
            ),
        )

        result = Validator().validate(
            report,
            _report_input(RetrievalStatus.INSUFFICIENT),
        )

        self.assertEqual(result.status, ValidationStatus.WARNING)
        self.assertNotIn(
            "deterministic_diagnosis",
            [issue.code for issue in result.issues],
        )

    def test_exact_training_dose_is_manual_review(self) -> None:
        report = _replace_report(
            _report(RetrievalStatus.COMPLETE),
            recommendations=["建议每天训练3次，每次30分钟。"],
        )

        result = Validator().validate(
            report,
            _report_input(RetrievalStatus.COMPLETE),
        )

        self.assertEqual(result.status, ValidationStatus.MANUAL_REVIEW)
        self.assertIn("exact_training_dose", [issue.code for issue in result.issues])

    def test_drug_or_dose_in_summary_cannot_bypass_validation(self) -> None:
        cases = (
            "量表结果来自模型预测；建议服用某种药物。",
            "量表结果来自模型预测；建议每天训练3次。",
        )
        for summary in cases:
            with self.subTest(summary=summary):
                report = _replace_report(
                    _report(RetrievalStatus.COMPLETE),
                    summary=summary,
                )

                result = Validator().validate(
                    report,
                    _report_input(RetrievalStatus.COMPLETE),
                )

                self.assertEqual(result.status, ValidationStatus.MANUAL_REVIEW)

    def test_mechanism_and_drug_recommendation_are_manual_review(self) -> None:
        cases = (
            ("summary", "量表结果来自模型预测；病理机制为皮质损伤。"),
            ("recommendations", ["建议服用某种药物。"]),
        )
        for field, value in cases:
            with self.subTest(field=field):
                report = _replace_report(
                    _report(RetrievalStatus.COMPLETE),
                    **{field: value},
                )

                result = Validator().validate(
                    report,
                    _report_input(RetrievalStatus.COMPLETE),
                )

                self.assertEqual(result.status, ValidationStatus.MANUAL_REVIEW)


if __name__ == "__main__":
    unittest.main()
