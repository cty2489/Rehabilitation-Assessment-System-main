from __future__ import annotations

import json
import unittest
from copy import deepcopy
from typing import Sequence
from unittest.mock import Mock, patch

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
from clinical_pipeline.report_generator import (
    ExistingReportLlmClient,
    ReportGenerationError,
    ReportGenerator,
    ReportMessage,
    ReportResult,
    _json_payload,
    build_conservative_report,
    build_llm_strategy_recovery_report,
)


MODEL_ID = "report-generator-test-model"


class FakeReportLlmClient:
    model_id = MODEL_ID

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[ReportMessage], int]] = []

    def generate(
        self,
        messages: Sequence[ReportMessage],
        *,
        attempt: int,
    ) -> str:
        self.calls.append((list(messages), attempt))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


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


def _biomarker_finding() -> Finding:
    return Finding(
        finding_id="biomarker:movement_smoothness_sparc",
        metric_key="movement_smoothness_sparc",
        name="运动平滑度SPARC",
        value=-1.4,
        unit="",
        status=FindingStatus.NOT_CLASSIFIABLE,
        modality=FindingModality.IMU,
        description="本次设备测量值为-1.4，仅用于同条件观察。",
        basis=FindingBasis(
            kind=FindingBasisKind.NO_RELIABLE_REFERENCE,
            description="该指标没有可靠通用参考范围。",
        ),
        source_field="biomarkers.movement_smoothness_sparc",
    )


def _plan(topic_count: int = 1) -> KnowledgePlan:
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


def _evidence() -> RetrievalEvidence:
    return RetrievalEvidence(
        evidence_id="evidence-1",
        query_id="query-1",
        chunk_id="KB-ASSESSMENT-001@1#001",
        text="该量表结果需结合标准化临床评估解释。",
        rank=1,
        raw_score=0.82,
        source_ids=["SRC-001"],
        metadata={"knowledge_id": "KB-ASSESSMENT-001"},
    )


def _report_input(
    status: RetrievalStatus,
    *,
    findings: list[Finding] | None = None,
) -> ReportGenerationInput:
    topic_count = 2 if status == RetrievalStatus.PARTIAL else 1
    plan = _plan(topic_count)
    evidence = [_evidence()] if status in {
        RetrievalStatus.COMPLETE,
        RetrievalStatus.PARTIAL,
    } else []
    covered = ["topic-1"] if evidence else []
    uncovered = [
        topic.topic_id for topic in plan.topics if topic.topic_id not in covered
    ]
    retrieval = RetrievalResult(
        retrieval_id=f"retrieval-{status.value}",
        attempt_id=f"attempt-{status.value}",
        status=status,
        queries=plan.queries,
        collection="rehab-test" if status != RetrievalStatus.UNAVAILABLE else None,
        evidence=evidence,
        covered_topic_ids=covered,
        uncovered_topic_ids=uncovered,
    )
    return ReportGenerationInput(
        run_id=f"run-{status.value}",
        quality_decision=QualityDecision.PASS,
        findings=InterpretationResult(findings=findings or [_finding()]),
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


def _valid_payload(status: RetrievalStatus) -> dict:
    evidence_summary = {
        RetrievalStatus.COMPLETE: "现有检索证据覆盖本次知识主题。",
        RetrievalStatus.PARTIAL: "证据覆盖不完整，第二个主题没有合格证据。",
        RetrievalStatus.INSUFFICIENT: "证据不足，未检索到合格来源。",
        RetrievalStatus.UNAVAILABLE: "检索证据不可用，未使用外部证据。",
    }[status]
    return {
        "summary": "量表结果来自模型预测，当前仅作结构化观察描述。",
        "evidence_summary": evidence_summary,
        "limitations": ["模型预测结果仍需结合标准化临床实测复核。"],
        "recommendations": ["建议由康复专业人员结合临床实测进行人工复核。"],
    }


class ReportGeneratorTests(unittest.TestCase):
    def test_existing_client_reuses_selected_local_llm_once(self) -> None:
        model = Mock()
        messages = [{"role": "user", "content": "test"}]
        with (
            patch("report.llm_provider", return_value="local"),
            patch("report.REPORT_MODEL", model),
            patch("report._generate_local_text", return_value="{}") as generate,
        ):
            text = ExistingReportLlmClient(model_id=MODEL_ID).generate(
                messages,
                attempt=1,
            )

        self.assertEqual(text, "{}")
        model.ensure_loaded.assert_called_once_with()
        generate.assert_called_once_with(
            model,
            messages,
            sample=False,
            generation_prefill="</think>\n{",
            max_new_tokens=768,
            stop_on_json=True,
            required_top_keys=[
                "summary",
                "evidence_summary",
                "limitations",
                "recommendations",
            ],
        )

    def test_json_payload_accepts_fenced_or_prefixed_object(self) -> None:
        payload = _valid_payload(RetrievalStatus.COMPLETE)
        raw = "已生成如下内容：\n```json\n" + json.dumps(
            payload, ensure_ascii=False
        ) + "\n```"

        self.assertEqual(_json_payload(raw), payload)

    def test_conservative_report_keeps_input_findings_and_marks_fallback(self) -> None:
        report_input = _report_input(RetrievalStatus.COMPLETE)

        report = build_conservative_report(report_input, model_id=MODEL_ID)

        self.assertEqual(report.generation_mode, "fallback")
        self.assertEqual(
            [finding.finding_id for finding in report.findings],
            ["prediction:FMA_UE"],
        )
        self.assertIn("模型预测", report.summary)
        self.assertEqual(len(report.recommendations), 5)
        self.assertTrue(any("动作质量" in item for item in report.recommendations))
        self.assertTrue(any("既往相同设备" in item for item in report.recommendations))

    def test_llm_strategy_recovery_uses_rag_grounded_recommendations(self) -> None:
        report_input = _report_input(RetrievalStatus.COMPLETE)
        response = json.dumps(
            {
                "summary": "模型预测提示本次手部动作完成情况需要结合任务表现进一步观察。",
                "recommendations": [
                    "基于本次FMA_UE=8分及“结构化观察结果相关知识”检索主题：锚点：策略名称：抓握—放开衔接；对应本次观察：FMA手部模型预测；动作：完成抓握后主动放开；反馈：观察手指是否能分开；调整：代偿增加时降低物品难度。",
                    "策略名称：手指分离控制；对应本次观察：手部动作完成情况；动作：逐指打开与合拢；反馈：观察腕部是否代偿；调整：动作变形时暂停并回到较简单动作。",
                    "策略名称：任务转化；对应本次观察：本次手功能模型预测；动作：拿取、放置与松开轻量物品；反馈：观察抓握后能否主动放开；调整：需要较多辅助时减少任务步骤。",
                    "策略名称：动作质量反馈；对应本次观察：本次记录；动作：握拳、张指与捏取；反馈：记录完成质量和不适；调整：疼痛或张力增加时暂停当前动作。",
                    "策略名称：连续任务练习；对应本次观察：FMA手部模型预测；动作：将抓握和松开串联；反馈：观察动作连续性；调整：质量持续下降时降低任务复杂度。",
                ],
            },
            ensure_ascii=False,
        )

        with patch(
            "clinical_pipeline.report_generator._generate_strategy_recovery_text",
            return_value=response,
        ) as generate:
            report = build_llm_strategy_recovery_report(
                report_input,
                model_id=MODEL_ID,
            )

        self.assertEqual(report.generation_mode, "llm")
        self.assertEqual(len(report.recommendations), 5)
        self.assertIn("模型预测", report.summary)
        self.assertNotIn("基于本次", report.recommendations[0])
        self.assertNotIn("FMA_UE", report.recommendations[0])
        self.assertNotIn("检索主题", report.recommendations[0])
        self.assertIn("抓握—放开衔接", report.recommendations[0])
        generate.assert_called_once()
        self.assertEqual(generate.call_args.kwargs["attempt"], 1)

    def test_llm_strategy_recovery_accepts_numbered_strategy_string(self) -> None:
        report_input = _report_input(RetrievalStatus.COMPLETE)
        response = json.dumps(
            {
                "summary": "本次模型预测用于观察手部任务表现。",
                "recommendations": "\n".join(
                    f"{index}. 策略{index}：依据本次模型预测完成相应手部任务；观察动作质量，质量下降时降低难度。"
                    for index in range(1, 6)
                ),
            },
            ensure_ascii=False,
        )
        with patch(
            "clinical_pipeline.report_generator._generate_strategy_recovery_text",
            return_value=response,
        ):
            report = build_llm_strategy_recovery_report(
                report_input,
                model_id=MODEL_ID,
        )

        self.assertEqual(len(report.recommendations), 5)
        self.assertIn("策略1", report.recommendations[0])

    def test_compact_contract_is_the_selected_production_report_path(self) -> None:
        report_input = _report_input(RetrievalStatus.COMPLETE)
        compact_result = build_conservative_report(report_input, model_id=MODEL_ID)
        compact_result = compact_result.model_copy(
            update={"generation_mode": "llm"}
        )
        with patch(
            "clinical_pipeline.report_generator.build_llm_strategy_recovery_report",
            return_value=compact_result,
        ) as compact:
            result = ReportGenerator(
                ExistingReportLlmClient(model_id=MODEL_ID),
                prefer_compact_contract=True,
            ).generate(report_input)

        self.assertIs(result, compact_result)
        compact.assert_called_once_with(report_input, model_id=MODEL_ID)

    def test_complete_generates_report_with_one_llm_call(self) -> None:
        llm = FakeReportLlmClient(
            [json.dumps(_valid_payload(RetrievalStatus.COMPLETE), ensure_ascii=False)]
        )

        result = ReportGenerator(llm).generate(
            _report_input(RetrievalStatus.COMPLETE)
        )

        self.assertIsInstance(result, ReportResult)
        self.assertEqual(result.report_model_id, MODEL_ID)
        self.assertEqual(result.citations, ["SRC-CORE-001", "SRC-001"])
        self.assertEqual(
            result.findings[0].citations,
            ["SRC-CORE-001", "SRC-001"],
        )
        self.assertIn("仅描述模型预测分数及量表范围", result.findings[0].statement)
        self.assertIn("解释边界：不得作为确定性诊断", result.findings[0].statement)
        self.assertIn("模型预测", result.summary)
        self.assertEqual(len(llm.calls), 1)

    def test_partial_states_incomplete_evidence_coverage(self) -> None:
        llm = FakeReportLlmClient(
            [json.dumps(_valid_payload(RetrievalStatus.PARTIAL), ensure_ascii=False)]
        )

        result = ReportGenerator(llm).generate(
            _report_input(RetrievalStatus.PARTIAL)
        )

        self.assertIn("证据覆盖不完整", result.evidence_summary)
        self.assertEqual(result.citations, ["SRC-CORE-001", "SRC-001"])

    def test_insufficient_states_evidence_is_insufficient(self) -> None:
        llm = FakeReportLlmClient(
            [
                json.dumps(
                    _valid_payload(RetrievalStatus.INSUFFICIENT),
                    ensure_ascii=False,
                )
            ]
        )

        result = ReportGenerator(llm).generate(
            _report_input(RetrievalStatus.INSUFFICIENT)
        )

        self.assertIn("证据不足", result.evidence_summary)
        self.assertEqual(result.citations, ["SRC-CORE-001"])

    def test_unavailable_uses_core_sources_without_fabricating_rag_sources(self) -> None:
        llm = FakeReportLlmClient(
            [json.dumps(_valid_payload(RetrievalStatus.UNAVAILABLE), ensure_ascii=False)]
        )

        result = ReportGenerator(llm).generate(
            _report_input(RetrievalStatus.UNAVAILABLE)
        )

        self.assertIn("检索证据不可用", result.evidence_summary)
        self.assertEqual(result.citations, ["SRC-CORE-001"])
        self.assertEqual(result.findings[0].citations, ["SRC-CORE-001"])
        self.assertNotIn("SRC-001", result.citations)

    def test_core_knowledge_source_is_allowed_when_retrieval_unavailable(self) -> None:
        payload = _valid_payload(RetrievalStatus.UNAVAILABLE)
        llm = FakeReportLlmClient(
            [json.dumps(payload, ensure_ascii=False)]
        )

        result = ReportGenerator(llm).generate(
            _report_input(RetrievalStatus.UNAVAILABLE)
        )

        self.assertEqual(result.citations, ["SRC-CORE-001"])

    def test_all_findings_are_assembled_without_llm_copying_rows(self) -> None:
        payload = _valid_payload(RetrievalStatus.COMPLETE)
        llm = FakeReportLlmClient([json.dumps(payload, ensure_ascii=False)])

        result = ReportGenerator(llm).generate(
            _report_input(
                RetrievalStatus.COMPLETE,
                findings=[_finding(), _biomarker_finding()],
            )
        )

        self.assertEqual(len(result.findings), 2)
        self.assertEqual(
            [finding.finding_id for finding in result.findings],
            ["prediction:FMA_UE", "biomarker:movement_smoothness_sparc"],
        )
        self.assertIn("-1.4", result.findings[1].statement)
        self.assertIn("确定性代码", llm.calls[0][0][0]["content"])

    def test_llm_cannot_inject_unknown_source_id(self) -> None:
        invalid = _valid_payload(RetrievalStatus.COMPLETE)
        invalid["citations"] = ["SRC-NOT-RETRIEVED"]
        llm = FakeReportLlmClient(
            [
                json.dumps(invalid, ensure_ascii=False),
                json.dumps(invalid, ensure_ascii=False),
            ]
        )

        with self.assertRaisesRegex(ReportGenerationError, "连续两次"):
            ReportGenerator(llm).generate(_report_input(RetrievalStatus.COMPLETE))

        self.assertEqual(len(llm.calls), 2)

    def test_diagnosis_mechanism_drug_and_exact_dose_are_rejected(self) -> None:
        cases = (
            ("summary", "诊断为脑卒中。"),
            ("summary", "病理机制为皮质损伤。"),
            ("recommendation", "建议服用某种药物。"),
            ("recommendation", "建议每天训练3次。"),
        )
        for field, text in cases:
            with self.subTest(field=field, text=text):
                invalid = deepcopy(_valid_payload(RetrievalStatus.COMPLETE))
                if field == "summary":
                    invalid["summary"] = text
                else:
                    invalid["recommendations"] = [text]
                encoded = json.dumps(invalid, ensure_ascii=False)
                llm = FakeReportLlmClient([encoded, encoded])

                with self.assertRaises(ReportGenerationError):
                    ReportGenerator(llm).generate(
                        _report_input(RetrievalStatus.COMPLETE)
                    )

                self.assertEqual(len(llm.calls), 2)

    def test_negated_diagnosis_limitation_is_not_rejected(self) -> None:
        payload = _valid_payload(RetrievalStatus.INSUFFICIENT)
        payload["summary"] = (
            "量表结果来自模型预测；当前证据不足以支持明确诊断。"
        )
        llm = FakeReportLlmClient(
            [json.dumps(payload, ensure_ascii=False)]
        )

        result = ReportGenerator(llm).generate(
            _report_input(RetrievalStatus.INSUFFICIENT)
        )

        self.assertIn("不足以支持明确诊断", result.summary)
        self.assertEqual(len(llm.calls), 1)

    def test_invalid_json_is_retried_once(self) -> None:
        llm = FakeReportLlmClient(
            [
                "not-json",
                json.dumps(_valid_payload(RetrievalStatus.COMPLETE), ensure_ascii=False),
            ]
        )

        result = ReportGenerator(llm).generate(
            _report_input(RetrievalStatus.COMPLETE)
        )

        self.assertEqual(result.generation_mode, "llm")
        self.assertEqual([attempt for _, attempt in llm.calls], [1, 2])
        self.assertIn("上一次输出未通过", llm.calls[1][0][0]["content"])

    def test_two_invalid_json_responses_raise_without_fallback(self) -> None:
        llm = FakeReportLlmClient(["not-json", "still-not-json"])

        with self.assertRaisesRegex(ReportGenerationError, "连续两次"):
            ReportGenerator(llm).generate(_report_input(RetrievalStatus.COMPLETE))

        self.assertEqual([attempt for _, attempt in llm.calls], [1, 2])


if __name__ == "__main__":
    unittest.main()
