from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import Mock

import rag_client

from clinical_pipeline.config import (
    CoreKnowledgeConfig,
    KnowledgeGraphConfig,
    LlmRoleConfig,
    PipelineConfig,
)
from clinical_pipeline.contracts import (
    CallComponent,
    CanonicalBiomarker,
    CanonicalPredictions,
    CoreKnowledgeBundle,
    CoreKnowledgeEntry,
    PipelineStage,
    RetrievalStatus,
    ValidationDecision,
)
from clinical_pipeline.knowledge_planner import KnowledgePlanner, PlannerMessage
from clinical_pipeline.orchestrator import (
    ClinicalPipelineOrchestrator,
    ModuleExecutionStatus,
    PipelineAssessmentInput,
    PipelineModule,
    PipelinePatientInput,
    PipelineRunStatus,
)
from clinical_pipeline.report_generator import ReportGenerator, ReportMessage
from clinical_pipeline.retriever import Retriever
from clinical_pipeline.validator import ValidationStatus


PLANNER_MODEL_ID = "fake-planner-model"
REPORT_MODEL_ID = "fake-report-model"


class FakeCoreKnowledgeProvider:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[list[str]] = []

    def provide(self, system_keys) -> CoreKnowledgeBundle:
        keys = list(system_keys)
        self.calls.append(keys)
        if self.failure is not None:
            raise self.failure
        return CoreKnowledgeBundle(
            bundle_id="fake-core-bundle",
            version="fake-core-v1",
            entries=[
                CoreKnowledgeEntry(
                    knowledge_id=f"CORE-{index}",
                    system_key=key,
                    allowed_interpretation="仅允许描述输入中的观察结果。",
                    prohibited_interpretation="不得作确定性诊断或训练处方。",
                    source_ids=[f"SRC-CORE-{index}"],
                )
                for index, key in enumerate(keys, start=1)
            ],
        )


class FakePlannerLlm:
    model_id = PLANNER_MODEL_ID

    def __init__(self) -> None:
        self.calls: list[tuple[list[PlannerMessage], int]] = []

    def generate(self, messages, *, attempt: int) -> str:
        self.calls.append((list(messages), attempt))
        return json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "topic-1",
                        "label": "FMA手部子量表解释边界",
                        "finding_ids": ["prediction:FMA_UE"],
                        "priority": "medium",
                    }
                ],
                "queries": [
                    {
                        "query_id": "query-1",
                        "topic_id": "topic-1",
                        "text": "FMA手部子量表模型预测结果的解释边界",
                    }
                ],
                "reason": "补充量表解释边界与来源证据。",
                "generation_mode": "llm",
            },
            ensure_ascii=False,
        )


class FakeReportLlm:
    model_id = REPORT_MODEL_ID

    def __init__(self) -> None:
        self.calls: list[tuple[list[ReportMessage], int]] = []

    def generate(self, messages, *, attempt: int) -> str:
        values = list(messages)
        self.calls.append((values, attempt))
        unavailable = '"status":"unavailable"' in values[1]["content"]
        partial = '"status":"partial"' in values[1]["content"]
        evidence_summary = (
            "检索证据不可用，本次仅使用结构化观察和固定核心知识。"
            if unavailable
            else (
                "检索证据覆盖不完整，本次仍保留结构化观察和固定核心知识。"
                if partial
                else "检索证据覆盖本次知识主题。"
            )
        )
        return json.dumps(
            {
                "summary": "量表结果来自模型预测，当前仅作结构化观察描述。",
                "evidence_summary": evidence_summary,
                "limitations": ["模型预测结果仍需结合临床实测复核。"],
                "recommendations": ["建议由康复专业人员进行人工复核。"],
            },
            ensure_ascii=False,
        )


def _rag_response() -> dict:
    return {
        "schema_version": "rehab.rag.retrieve.v1",
        "collection": "fake-rag-collection",
        "results": [
            {
                "key": "q1",
                "query": "FMA手部子量表模型预测结果的解释边界",
                "hits": [
                    {
                        "rank": 1,
                        "score": 0.88,
                        "knowledge_id": "KB-ASSESSMENT-001",
                        "chunk_id": "KB-ASSESSMENT-001@1#001",
                        "title": "FMA解释边界",
                        "text": "量表结果需结合标准化临床评估解释。",
                        "metadata": {
                            "clinical_ready": True,
                            "source_ids": ["SRC-001"],
                        },
                    }
                ],
            }
        ],
    }


def _settings() -> rag_client.RagClientSettings:
    return rag_client.RagClientSettings(
        mode="off",
        service_url="http://127.0.0.1:8010",
        timeout_seconds=1.0,
        top_k_per_query=2,
        max_sources=6,
        max_context_chars=8000,
        assist_approved=False,
        shadow_include_demo=False,
        allow_demo_in_prompt=False,
        trace_enabled=False,
        trace_path=Path("/unused/orchestrator-rag-trace.jsonl"),
    )


def _config(*, graph_mode: str = "llm_only") -> PipelineConfig:
    return PipelineConfig(
        config_version="orchestrator-test-v0.1",
        core_knowledge=CoreKnowledgeConfig(bundle_version="fake-core-v1"),
        planner=LlmRoleConfig(model_id=PLANNER_MODEL_ID),
        knowledge_graph=KnowledgeGraphConfig(mode=graph_mode),
        report_generator=LlmRoleConfig(model_id=REPORT_MODEL_ID),
    )


def _marker() -> CanonicalBiomarker:
    return CanonicalBiomarker(
        metric_key="movement_smoothness_sparc",
        name="运动平滑度SPARC",
        value=-1.4,
        modality="imu",
        available=True,
        n_valid=1,
    )


def _input(
    *,
    patient_id: str | None = "P-DEMO-001",
    predictions: CanonicalPredictions | None = None,
    biomarkers: list[CanonicalBiomarker] | None = None,
) -> PipelineAssessmentInput:
    return PipelineAssessmentInput(
        patient=PipelinePatientInput(patient_id=patient_id),
        predictions=predictions
        if predictions is not None
        else CanonicalPredictions(FMA_UE=8, hand_tone="2", hand_function=3),
        biomarkers=[_marker()] if biomarkers is None else biomarkers,
    )


def _harness(
    *,
    rag_failure: bool = False,
    core_failure: Exception | None = None,
    graph_mode: str = "llm_only",
    graph_rag_adapter=None,
):
    planner_llm = FakePlannerLlm()
    report_llm = FakeReportLlm()
    transport = Mock(
        side_effect=TimeoutError("fake RAG unavailable")
        if rag_failure
        else None,
        return_value=None if rag_failure else _rag_response(),
    )
    core = FakeCoreKnowledgeProvider(core_failure)
    orchestrator = ClinicalPipelineOrchestrator(
        config=_config(graph_mode=graph_mode),
        core_knowledge_provider=core,
        knowledge_planner=KnowledgePlanner(planner_llm),
        retriever=Retriever(settings=_settings(), transport=transport),
        report_generator=ReportGenerator(report_llm),
        graph_rag_adapter=graph_rag_adapter,
    )
    return orchestrator, planner_llm, report_llm, transport, core


def _non_imu_markers() -> list[CanonicalBiomarker]:
    return [
        CanonicalBiomarker(
            metric_key="wrist_co_contraction_index",
            name="腕屈伸肌共收缩指数（CCI-腕）",
            value=0.42,
            modality="emg",
            available=True,
            n_valid=3,
        ),
        CanonicalBiomarker(
            metric_key="emg_activation_rms",
            name="肌肉激活幅度（RMS）",
            value=0.0011,
            modality="emg",
            available=True,
            n_valid=3,
        ),
        CanonicalBiomarker(
            metric_key="pathological_asymmetry_index",
            name="病理性半球不对称指数（PAI）",
            value=0.12,
            modality="eeg",
            available=True,
            n_valid=3,
        ),
        CanonicalBiomarker(
            metric_key="interhemispheric_motor_coherence",
            name="半球间运动皮层相干",
            value=0.28,
            modality="eeg",
            available=True,
            n_valid=3,
        ),
    ]


class ClinicalPipelineOrchestratorTests(unittest.TestCase):
    def test_normal_patient_runs_complete_pipeline(self) -> None:
        orchestrator, planner_llm, report_llm, transport, _ = _harness()

        result = orchestrator.run(_input())

        self.assertEqual(result.status, PipelineRunStatus.COMPLETED)
        self.assertEqual(result.validation.status, ValidationStatus.PASSED)
        self.assertEqual(result.trace.stage, PipelineStage.COMPLETED)
        self.assertEqual(result.trace.validation_decision, ValidationDecision.PASS)
        self.assertEqual(len(planner_llm.calls), 1)
        self.assertEqual(len(report_llm.calls), 1)
        transport.assert_called_once()
        self.assertEqual(
            result.planner_call_id,
            result.trace.calls_for(CallComponent.KNOWLEDGE_PLANNER_LLM)[0].call_id,
        )
        self.assertEqual(
            result.retriever_attempt_id,
            result.trace.calls_for(CallComponent.RETRIEVER)[0].call_id,
        )
        self.assertEqual(
            result.report_generator_call_id,
            result.trace.calls_for(CallComponent.REPORT_GENERATOR_LLM)[0].call_id,
        )
        self.assertEqual(
            result.trace.artifact_refs["validation_result"],
            result.validation.validation_id,
        )

    def test_missing_biomarkers_runs_fully_but_cannot_end_passed(self) -> None:
        orchestrator, planner_llm, report_llm, transport, _ = _harness()

        result = orchestrator.run(_input(biomarkers=[]))

        self.assertEqual(result.quality_gate.decision.value, "review")
        self.assertEqual(result.status, PipelineRunStatus.COMPLETED)
        self.assertEqual(result.validation.status, ValidationStatus.WARNING)
        self.assertEqual(result.trace.validation_decision, ValidationDecision.WARNING)
        self.assertEqual(result.trace.stage, PipelineStage.COMPLETED)
        self.assertEqual(len(planner_llm.calls), 1)
        self.assertEqual(len(report_llm.calls), 1)
        transport.assert_called_once()

    def test_missing_patient_id_blocks_with_zero_downstream_calls(self) -> None:
        orchestrator, planner_llm, report_llm, transport, _ = _harness()

        result = orchestrator.run(_input(patient_id=""))

        self.assertEqual(result.status, PipelineRunStatus.BLOCKED)
        self.assertEqual(result.trace.stage, PipelineStage.BLOCKED)
        self.assertTrue(any("患者标识" in reason for reason in result.block_reasons))
        self.assertEqual(
            result.trace.call_count(CallComponent.KNOWLEDGE_PLANNER_LLM), 0
        )
        self.assertEqual(result.trace.call_count(CallComponent.RETRIEVER), 0)
        self.assertEqual(
            result.trace.call_count(CallComponent.REPORT_GENERATOR_LLM), 0
        )
        self.assertEqual(planner_llm.calls, [])
        self.assertEqual(report_llm.calls, [])
        transport.assert_not_called()

    def test_missing_predictions_blocks(self) -> None:
        for predictions in (None, CanonicalPredictions()):
            with self.subTest(predictions=predictions):
                orchestrator, planner_llm, report_llm, transport, _ = _harness()
                value = _input()
                value.predictions = predictions

                result = orchestrator.run(value)

                self.assertEqual(result.status, PipelineRunStatus.BLOCKED)
                self.assertTrue(
                    any("模型预测结果完全缺失" in item for item in result.block_reasons)
                )
                self.assertEqual(planner_llm.calls, [])
                self.assertEqual(report_llm.calls, [])
                transport.assert_not_called()

    def test_retriever_unavailable_still_generates_limited_report(self) -> None:
        orchestrator, planner_llm, report_llm, transport, _ = _harness(
            rag_failure=True
        )

        result = orchestrator.run(_input())

        self.assertEqual(result.status, PipelineRunStatus.COMPLETED)
        self.assertEqual(result.retrieval.status, RetrievalStatus.UNAVAILABLE)
        self.assertIn("检索证据不可用", result.report.evidence_summary)
        self.assertEqual(
            result.report.citations,
            ["SRC-CORE-1", "SRC-CORE-2", "SRC-CORE-3", "SRC-CORE-4"],
        )
        self.assertNotIn("SRC-001", result.report.citations)
        self.assertEqual(result.validation.status, ValidationStatus.WARNING)
        self.assertEqual(len(planner_llm.calls), 1)
        self.assertEqual(len(report_llm.calls), 1)
        transport.assert_called_once()

    def test_graph_enhanced_mode_seeds_planner_and_retains_non_imu_topics(self) -> None:
        orchestrator, planner_llm, _, transport, _ = _harness(
            graph_mode="graph_enhanced"
        )

        result = orchestrator.run(_input(biomarkers=_non_imu_markers()))

        self.assertEqual(result.status, PipelineRunStatus.COMPLETED)
        self.assertIsNotNone(result.knowledge_graph_evidence)
        self.assertEqual(result.trace.artifact_refs["knowledge_graph_mode"], "graph_enhanced")
        self.assertEqual(result.trace.artifact_refs["knowledge_graph_status"], "matched")
        self.assertIn("knowledge_graph_context", planner_llm.calls[0][0][1]["content"])
        self.assertTrue(
            any(topic.topic_id.startswith("kg-topic-") for topic in result.knowledge_plan.topics)
        )
        payload = transport.call_args.args[1]
        self.assertGreater(len(payload["queries"]), 1)
        self.assertFalse(
            any(
                state["source_modality"] == "imu"
                for state in result.knowledge_graph_evidence["matched_indicator_states"]
            )
        )

    def test_graph_enhanced_filters_imu_before_planner_and_report_input(self) -> None:
        orchestrator, planner_llm, _, _, core = _harness(graph_mode="graph_enhanced")

        result = orchestrator.run(
            _input(biomarkers=[*_non_imu_markers(), _marker()])
        )

        self.assertEqual(result.status, PipelineRunStatus.COMPLETED)
        self.assertEqual(result.trace.artifact_refs["knowledge_graph_non_imu_scope"], "applied")
        self.assertNotIn("movement_smoothness_sparc", planner_llm.calls[0][0][1]["content"])
        self.assertNotIn("movement_smoothness_sparc", core.calls[0])
        self.assertFalse(
            any(
                finding.modality.value == "imu"
                for finding in result.report_input.findings.findings
            )
        )
        self.assertFalse(
            any(
                finding.modality.value == "imu"
                for finding in result.interpretation.findings
            )
        )
        self.assertFalse(result.knowledge_graph_evidence["data_quality_warnings"])
        self.assertTrue(result.knowledge_graph_evidence["measurement_context_topics"])

    def test_llm_only_keeps_legacy_imu_behavior(self) -> None:
        orchestrator, planner_llm, _, _, core = _harness(graph_mode="llm_only")

        result = orchestrator.run(_input(biomarkers=[*_non_imu_markers(), _marker()]))

        self.assertEqual(result.status, PipelineRunStatus.COMPLETED)
        self.assertEqual(result.trace.artifact_refs["knowledge_graph_non_imu_scope"], "not_applied")
        self.assertIn("movement_smoothness_sparc", planner_llm.calls[0][0][1]["content"])
        self.assertIn("movement_smoothness_sparc", core.calls[0])
        self.assertTrue(
            any(
                finding.modality.value == "imu"
                for finding in result.report_input.findings.findings
            )
        )

    def test_graph_failure_falls_back_to_legacy_complete_context(self) -> None:
        class FailingGraphAdapter:
            def build_evidence(self, *_args, **_kwargs):
                raise RuntimeError("intentional graph failure")

        orchestrator, planner_llm, _, _, core = _harness(
            graph_mode="graph_enhanced",
            graph_rag_adapter=FailingGraphAdapter(),
        )

        result = orchestrator.run(_input(biomarkers=[*_non_imu_markers(), _marker()]))

        self.assertEqual(result.status, PipelineRunStatus.COMPLETED)
        self.assertEqual(
            result.trace.artifact_refs["knowledge_graph_status"],
            "fallback_to_llm_only",
        )
        self.assertIn("movement_smoothness_sparc", planner_llm.calls[0][0][1]["content"])
        self.assertIn("movement_smoothness_sparc", core.calls[0])
        self.assertTrue(
            any(
                finding.modality.value == "imu"
                for finding in result.report_input.findings.findings
            )
        )

    def test_module_order_cannot_skip(self) -> None:
        orchestrator, *_ = _harness()

        result = orchestrator.run(_input())

        expected = list(PipelineModule)
        started = [
            event.module
            for event in result.module_events
            if event.status == ModuleExecutionStatus.STARTED
        ]
        completed = [
            event.module
            for event in result.module_events
            if event.status == ModuleExecutionStatus.COMPLETED
        ]
        self.assertEqual(started, expected)
        self.assertEqual(completed, expected)
        self.assertEqual(
            [event.sequence for event in result.module_events],
            list(range(1, len(result.module_events) + 1)),
        )
        trace_event_keys = [
            key
            for key in result.trace.artifact_refs
            if key.startswith("module_event:")
        ]
        self.assertEqual(
            trace_event_keys,
            [
                f"module_event:{event.sequence:03d}:"
                f"{event.module.value}:{event.status.value}"
                for event in result.module_events
            ],
        )

    def test_module_exception_records_failed_stage(self) -> None:
        orchestrator, planner_llm, report_llm, transport, _ = _harness(
            core_failure=RuntimeError("fake core failure")
        )

        result = orchestrator.run(_input())

        self.assertEqual(result.status, PipelineRunStatus.FAILED)
        self.assertEqual(result.failure.module, PipelineModule.CORE_KNOWLEDGE_PROVIDER)
        self.assertEqual(
            result.trace.artifact_refs["failure_stage"],
            PipelineModule.CORE_KNOWLEDGE_PROVIDER.value,
        )
        self.assertEqual(result.trace.stage, PipelineStage.INTERPRETED)
        self.assertEqual(result.module_events[-1].status, ModuleExecutionStatus.FAILED)
        self.assertEqual(planner_llm.calls, [])
        self.assertEqual(report_llm.calls, [])
        transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
