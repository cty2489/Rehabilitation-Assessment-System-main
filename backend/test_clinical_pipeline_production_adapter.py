from __future__ import annotations

import ast
import unittest
from pathlib import Path

from clinical_pipeline.contracts import (
    CanonicalPredictions,
    CoreKnowledgeBundle,
    CoreKnowledgeEntry,
    Finding,
    FindingBasis,
    FindingBasisKind,
    FindingModality,
    FindingStatus,
    InterpretationResult,
    PipelineRunTrace,
    QualityDecision,
    RetrievalEvidence,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStatus,
)
from clinical_pipeline.orchestrator import (
    OrchestrationResult,
    PipelineRunStatus,
    QualityGateResult,
)
from clinical_pipeline.production_adapter import (
    ProductionAdapterError,
    adapt_production_input,
    build_production_orchestrator,
    orchestration_metadata,
    render_compatible_markdown,
)
from clinical_pipeline.report_generator import ReportFinding, ReportResult
from clinical_pipeline.validator import (
    ValidationIssue,
    ValidationResult,
    ValidationStatus,
)
from schemas import PatientInfo


def _patient(patient_id: str = "P001") -> PatientInfo:
    return PatientInfo(
        patient_id=patient_id,
        name="张三",
        sex="男",
        age=62,
        diagnosis="脑梗死",
        disease_days=120,
        paralysis_side="左",
    )


def _predictions() -> dict:
    return {"FMA_UE": 8.0, "hand_tone": "2", "hand_function": 3}


def _biomarkers() -> dict:
    return {
        "groups": [
            {
                "key": "imu",
                "markers": [
                    {
                        "key": "movement_smoothness_sparc",
                        "name": "运动平滑度SPARC",
                        "value": -1.4,
                        "unit": "",
                        "available": True,
                        "n_valid": 3,
                    },
                    {
                        "key": "range_of_motion_proxy",
                        "name": "活动范围代理值",
                        "value": "—",
                        "unit": "°",
                        "available": False,
                        "n_valid": 0,
                    },
                ],
            }
        ]
    }


def _interpretation() -> InterpretationResult:
    return InterpretationResult(
        interpretation_id="interpretation-production",
        findings=[
            Finding(
                finding_id="prediction:FMA_UE",
                metric_key="FMA_UE",
                name="FMA手部子量表，范围0–20",
                value=8,
                unit="分",
                status=FindingStatus.OBSERVED,
                modality=FindingModality.CLINICAL_SCALE,
                description="模型预测结果：8分。",
                basis=FindingBasis(
                    kind=FindingBasisKind.SCALE_DEFINITION,
                    description="模型预测结果。",
                ),
                source_field="predictions.FMA_UE",
            )
        ],
    )


def _completed_result(
    *,
    retrieval_status: RetrievalStatus = RetrievalStatus.COMPLETE,
    validation_status: ValidationStatus = ValidationStatus.PASSED,
    core_only_citation: bool = False,
) -> OrchestrationResult:
    query = RetrievalQuery(
        query_id="query-1",
        topic_id="topic-1",
        text="FMA手部子量表解释边界",
    )
    evidence = []
    citations = []
    if retrieval_status == RetrievalStatus.COMPLETE:
        evidence = [
            RetrievalEvidence(
                evidence_id="evidence-1",
                query_id="query-1",
                chunk_id="chunk-1",
                text="量表结果需要结合标准化检查解释。",
                rank=1,
                raw_score=0.9,
                source_ids=["SRC-001"],
                metadata={"knowledge_id": "KB-001", "title": "FMA解释边界"},
            )
        ]
        citations = ["SRC-001"]
    elif core_only_citation:
        citations = ["SRC-CORE-001"]
    retrieval = RetrievalResult(
        attempt_id="retrieval-attempt-1",
        status=retrieval_status,
        queries=[query],
        evidence=evidence,
        covered_topic_ids=["topic-1"] if evidence else [],
        uncovered_topic_ids=[] if evidence else ["topic-1"],
    )
    report = ReportResult(
        report_id="report-production",
        report_model_id="qwen3_8b_hf",
        summary="量表结果来自模型预测，本次仅描述结构化观察。",
        findings=[
            ReportFinding(
                finding_id="prediction:FMA_UE",
                statement="模型预测的FMA手部子量表结果为8分。",
                citations=citations,
            )
        ],
        evidence_summary=(
            "检索证据不可用，本次仅使用结构化观察和固定核心知识。"
            if retrieval_status == RetrievalStatus.UNAVAILABLE
            else "检索证据覆盖本次知识主题。"
        ),
        limitations=[
            "检索证据不可用。"
            if retrieval_status == RetrievalStatus.UNAVAILABLE
            else "模型预测结果仍需结合临床实测复核。"
        ],
        recommendations=["建议由康复专业人员进行人工复核。"],
        citations=citations,
    )
    issues = []
    if validation_status == ValidationStatus.WARNING:
        issues = [
            ValidationIssue(
                code="evidence_limit_disclosed",
                level="warning",
                message="报告已披露检索证据不可用。",
            )
        ]
    elif validation_status == ValidationStatus.MANUAL_REVIEW:
        issues = [
            ValidationIssue(
                code="deterministic_diagnosis",
                level="manual_review",
                message="报告出现需要人工复核的表达。",
            )
        ]
    validation = ValidationResult(
        validation_id="validation-production",
        report_id=report.report_id,
        report_input_id="report-input-production",
        status=validation_status,
        issues=issues,
    )
    return OrchestrationResult(
        status=PipelineRunStatus.COMPLETED,
        trace=PipelineRunTrace(run_id="pipeline-production"),
        quality_gate=QualityGateResult(decision=QualityDecision.PASS),
        interpretation=_interpretation(),
        core_knowledge=CoreKnowledgeBundle(
            bundle_id="core-production",
            version="core-v1",
            entries=[
                CoreKnowledgeEntry(
                    knowledge_id="CORE-FMA-HAND",
                    system_key="FMA_UE",
                    allowed_interpretation="仅描述模型预测分数及量表范围。",
                    source_ids=["SRC-CORE-001"],
                )
            ],
        ),
        retrieval=retrieval,
        report=report,
        validation=validation,
    )


class ProductionAdapterTests(unittest.TestCase):
    def test_browser_and_device_inputs_use_the_same_canonical_adapter(self) -> None:
        for institution, patient in (
            ("hospital", _patient("HOSP001")),
            ("device", _patient("DEV001_0001")),
        ):
            with self.subTest(institution=institution):
                request = adapt_production_input(
                    patient=patient,
                    predictions_raw=_predictions(),
                    biomarkers=_biomarkers(),
                    quality={"status": "pass", "institution": institution},
                    assessment_id=f"assessment-{institution}",
                    patient_id=patient.patient_id,
                    report_model_id="qwen3_8b_hf",
                    context_id=f"session-{institution}",
                )

                value = request.assessment_input
                self.assertEqual(value.patient.patient_id, patient.patient_id)
                self.assertEqual(
                    value.predictions,
                    CanonicalPredictions(FMA_UE=8, hand_tone="2", hand_function=3),
                )
                self.assertEqual(len(value.biomarkers), 2)
                self.assertIsNone(value.biomarkers[1].value)
                self.assertFalse(value.biomarkers[1].available)
                self.assertEqual(request.report_model_id, "qwen3_8b_hf")

    def test_missing_critical_fields_raise_clear_errors(self) -> None:
        with self.assertRaisesRegex(ProductionAdapterError, "hand_function"):
            adapt_production_input(
                patient=_patient(),
                predictions_raw={"FMA_UE": 8, "hand_tone": "2"},
                biomarkers=None,
                quality={},
                assessment_id=None,
                patient_id="P001",
                report_model_id="qwen3_8b_hf",
            )
        with self.assertRaisesRegex(ProductionAdapterError, "report_model_id"):
            adapt_production_input(
                patient=_patient(),
                predictions_raw=_predictions(),
                biomarkers=None,
                quality={},
                assessment_id=None,
                patient_id="P001",
                report_model_id="",
            )
        with self.assertRaisesRegex(ProductionAdapterError, "不一致"):
            adapt_production_input(
                patient=_patient(),
                predictions_raw=_predictions(),
                biomarkers=None,
                quality={},
                assessment_id=None,
                patient_id="OTHER",
                report_model_id="qwen3_8b_hf",
            )

    def test_production_orchestrator_keeps_llm_roles_independent(self) -> None:
        orchestrator = build_production_orchestrator("same-model")

        self.assertIsNot(
            orchestrator._knowledge_planner._llm,
            orchestrator._report_generator._llm,
        )
        self.assertEqual(orchestrator._knowledge_planner._model_id, "same-model")
        self.assertEqual(orchestrator._report_generator._model_id, "same-model")

    def test_unavailable_retrieval_keeps_visible_report_clean(self) -> None:
        result = _completed_result(
            retrieval_status=RetrievalStatus.UNAVAILABLE,
            validation_status=ValidationStatus.WARNING,
        )

        markdown = render_compatible_markdown(
            patient=_patient(),
            result=result,
            assessment_validation_status="research_assessment",
            quality={"status": "pass"},
        )

        self.assertNotIn("报告校验提示", markdown)
        self.assertNotIn("报告校验状态", markdown)
        self.assertNotIn("Validator：", markdown)
        self.assertIn("检索证据不可用", markdown)
        self.assertIn("本次报告未引用外部检索来源", markdown)
        self.assertEqual(
            orchestration_metadata(result)["retrieval_status"], "unavailable"
        )

    def test_core_knowledge_citation_has_reference_details(self) -> None:
        result = _completed_result(
            retrieval_status=RetrievalStatus.UNAVAILABLE,
            validation_status=ValidationStatus.WARNING,
            core_only_citation=True,
        )

        markdown = render_compatible_markdown(
            patient=_patient(),
            result=result,
            assessment_validation_status="research_assessment",
            quality={"status": "pass"},
        )

        self.assertIn("【1】SRC-CORE-001；FMA_UE；CORE-FMA-HAND", markdown)

    def test_manual_review_stays_in_audit_metadata_not_visible_report(self) -> None:
        result = _completed_result(validation_status=ValidationStatus.MANUAL_REVIEW)

        markdown = render_compatible_markdown(
            patient=_patient(),
            result=result,
            assessment_validation_status="engineering_validation_only",
            quality={"status": "needs_review"},
        )

        self.assertNotIn("人工复核要求", markdown)
        self.assertNotIn("Validator：", markdown)
        self.assertNotIn("设备端工程验证提示", markdown)
        self.assertNotIn("信号质量复核", markdown)
        self.assertIn("【1】SRC-001；FMA解释边界；KB-001", markdown)
        self.assertEqual(
            orchestration_metadata(result)["validation_status"], "manual_review"
        )

    def test_main_has_no_legacy_stream_report_import_or_call(self) -> None:
        main_path = Path(__file__).with_name("main.py")
        tree = ast.parse(main_path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "report"
            for alias in node.names
        }
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("stream_report", imported)
        self.assertNotIn("stream_report", calls)


if __name__ == "__main__":
    unittest.main()
