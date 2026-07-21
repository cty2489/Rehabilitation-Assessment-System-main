from __future__ import annotations

import json
import unittest
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
)
from clinical_pipeline.knowledge_planner import (
    ExistingLlmClient,
    KnowledgePlanner,
    PlannerMessage,
)


MODEL_ID = "planner-test-model"


def _interpretation() -> InterpretationResult:
    return InterpretationResult(
        interpretation_id="interpretation-planner-test",
        findings=[
            Finding(
                finding_id="biomarker:movement_smoothness_sparc",
                metric_key="movement_smoothness_sparc",
                name="运动平滑度SPARC",
                value=-1.4,
                unit=None,
                status=FindingStatus.NOT_CLASSIFIABLE,
                modality=FindingModality.IMU,
                description="当前仅记录观察值，不作正常或异常分类。",
                basis=FindingBasis(
                    kind=FindingBasisKind.NO_RELIABLE_REFERENCE,
                    description="没有可靠的单次绝对参考范围。",
                    reference_type="none",
                ),
                source_field="biomarkers.movement_smoothness_sparc.value",
            )
        ],
    )


def _core_knowledge() -> CoreKnowledgeBundle:
    return CoreKnowledgeBundle(
        bundle_id="core-planner-test",
        version="core-v1",
        entries=[
            CoreKnowledgeEntry(
                knowledge_id="KB-IMU-001",
                system_key="movement_smoothness_sparc",
                allowed_interpretation="仅允许用于同条件观察和补充证据检索。",
                prohibited_interpretation="不得根据单次值判定正常或异常。",
                source_ids=["SRC-001"],
            )
        ],
    )


def _valid_payload() -> dict:
    return {
        "topics": [
            {
                "topic_id": "topic-1",
                "label": "运动平滑度指标的解释边界",
                "finding_ids": ["biomarker:movement_smoothness_sparc"],
                "priority": "medium",
            }
        ],
        "queries": [
            {
                "query_id": "query-1",
                "topic_id": "topic-1",
                "text": "SPARC运动平滑度指标在同条件康复评估中的研究证据",
            }
        ],
        "reason": "补充该观察指标的适用边界与证据来源。",
        "generation_mode": "llm",
    }


class FakeLlmClient:
    model_id = MODEL_ID

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[PlannerMessage], int]] = []

    def generate(
        self,
        messages: Sequence[PlannerMessage],
        *,
        attempt: int,
    ) -> str:
        self.calls.append((list(messages), attempt))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class KnowledgePlannerTests(unittest.TestCase):
    def test_existing_client_reuses_selected_local_llm_once(self) -> None:
        model = Mock()
        messages = [{"role": "user", "content": "test"}]
        with (
            patch("report.llm_provider", return_value="local"),
            patch("report.REPORT_MODEL", model),
            patch(
                "report._generate_local_text",
                return_value=json.dumps(_valid_payload(), ensure_ascii=False),
            ) as generate,
        ):
            text = ExistingLlmClient(model_id=MODEL_ID).generate(
                messages,
                attempt=1,
            )

        model.ensure_loaded.assert_called_once_with()
        generate.assert_called_once_with(
            model,
            messages,
            sample=False,
            max_new_tokens=768,
        )
        self.assertIn('"generation_mode": "llm"', text)

    def test_normal_output_has_topic_and_query_with_one_llm_call(self) -> None:
        llm = FakeLlmClient([json.dumps(_valid_payload(), ensure_ascii=False)])

        plan = KnowledgePlanner(llm).plan(_interpretation(), _core_knowledge())

        self.assertEqual(plan.planner_model_id, MODEL_ID)
        self.assertEqual(plan.generation_mode, "llm")
        self.assertGreaterEqual(len(plan.topics), 1)
        self.assertGreaterEqual(len(plan.queries), 1)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0][1], 1)
        self.assertIn("core_knowledge", llm.calls[0][0][1]["content"])

    def test_needs_retrieval_is_forbidden_and_triggers_retry(self) -> None:
        forbidden = {**_valid_payload(), "needs_retrieval": True}
        llm = FakeLlmClient(
            [
                json.dumps(forbidden, ensure_ascii=False),
                json.dumps(_valid_payload(), ensure_ascii=False),
            ]
        )

        plan = KnowledgePlanner(llm).plan(_interpretation(), _core_knowledge())

        self.assertEqual(plan.generation_mode, "llm")
        self.assertEqual(len(llm.calls), 2)
        self.assertNotIn("needs_retrieval", plan.model_dump())

    def test_diagnosis_and_treatment_fields_are_forbidden(self) -> None:
        for forbidden_field in ("diagnosis", "treatment_advice"):
            with self.subTest(field=forbidden_field):
                forbidden = {**_valid_payload(), forbidden_field: "不允许的内容"}
                llm = FakeLlmClient(
                    [
                        json.dumps(forbidden, ensure_ascii=False),
                        json.dumps(_valid_payload(), ensure_ascii=False),
                    ]
                )

                plan = KnowledgePlanner(llm).plan(
                    _interpretation(),
                    _core_knowledge(),
                )

                self.assertEqual(plan.generation_mode, "llm")
                self.assertEqual(len(llm.calls), 2)
                self.assertNotIn(forbidden_field, plan.model_dump())

    def test_invalid_json_is_retried_once(self) -> None:
        llm = FakeLlmClient(
            [
                "not-json",
                json.dumps(_valid_payload(), ensure_ascii=False),
            ]
        )

        plan = KnowledgePlanner(llm).plan(_interpretation(), _core_knowledge())

        self.assertEqual(plan.generation_mode, "llm")
        self.assertEqual([attempt for _, attempt in llm.calls], [1, 2])
        self.assertIn("上一次输出未通过", llm.calls[1][0][0]["content"])

    def test_two_failures_use_finding_name_fallback(self) -> None:
        llm = FakeLlmClient(["not-json", RuntimeError("LLM unavailable")])

        plan = KnowledgePlanner(llm).plan(_interpretation(), _core_knowledge())

        self.assertEqual(plan.generation_mode, "fallback")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(len(plan.topics), 1)
        self.assertEqual(len(plan.queries), 1)
        self.assertIn("运动平滑度SPARC", plan.queries[0].text)
        self.assertNotIn("诊断", plan.model_dump_json())
        self.assertNotIn("治疗建议", plan.model_dump_json())
        self.assertNotIn("训练剂量", plan.model_dump_json())

    def test_clinical_conclusions_inside_allowed_fields_are_rejected(self) -> None:
        invalid = _valid_payload()
        invalid["reason"] = "诊断为某疾病，建议每天训练30分钟。"
        llm = FakeLlmClient(
            [
                json.dumps(invalid, ensure_ascii=False),
                json.dumps(_valid_payload(), ensure_ascii=False),
            ]
        )

        plan = KnowledgePlanner(llm).plan(_interpretation(), _core_knowledge())

        self.assertEqual(plan.generation_mode, "llm")
        self.assertEqual(len(llm.calls), 2)
        self.assertNotIn("诊断为", plan.reason)


if __name__ == "__main__":
    unittest.main()
