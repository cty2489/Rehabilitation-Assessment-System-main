from __future__ import annotations

import unittest

from pydantic import ValidationError

from clinical_pipeline.config import (
    CoreKnowledgeConfig,
    LlmRoleConfig,
    PipelineConfig,
)
from clinical_pipeline.contracts import (
    CallComponent,
    CoreKnowledgeBundle,
    CoreKnowledgeEntry,
    Finding,
    FindingModality,
    FindingStatus,
    InterpretationResult,
    KnowledgePlan,
    KnowledgeTopic,
    PipelineRunTrace,
    PipelineStage,
    QualityDecision,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStatus,
    ValidationDecision,
)
from clinical_pipeline.report_input import (
    ReportInputAssembler,
    ReportInputAssemblyError,
)
from clinical_pipeline.state_machine import PipelineStateError, PlannerRagStateMachine


MODEL_ID = "same-model-id"


def _findings() -> InterpretationResult:
    return InterpretationResult(
        interpretation_id="interpretation-1",
        findings=[
            Finding(
                finding_id="pred:FMA_UE",
                metric_key="FMA_UE",
                name="FMA手部分数",
                value=8,
                status=FindingStatus.OBSERVED,
                modality=FindingModality.CLINICAL_SCALE,
            )
        ],
    )


def _core() -> CoreKnowledgeBundle:
    return CoreKnowledgeBundle(
        bundle_id="core-1",
        version="core-v1",
        entries=[
            CoreKnowledgeEntry(
                knowledge_id="CORE-FMA-HAND",
                system_key="FMA_UE",
                allowed_interpretation="测试允许解释",
                prohibited_interpretation="测试禁止解释",
                source_ids=["SRC-CORE-1"],
            )
        ],
    )


def _plan() -> KnowledgePlan:
    return KnowledgePlan(
        plan_id="plan-1",
        planner_model_id=MODEL_ID,
        topics=[
            KnowledgeTopic(
                topic_id="topic-1",
                label="测试知识主题",
                finding_ids=["pred:FMA_UE"],
            )
        ],
        queries=[
            RetrievalQuery(
                query_id="query-1",
                topic_id="topic-1",
                text="测试检索查询",
            )
        ],
        reason="测试简短原因",
    )


def _advance_to_plan(machine: PlannerRagStateMachine, decision: QualityDecision):
    findings = _findings()
    core = _core()
    plan = _plan()
    machine.complete_quality_gate(decision)
    machine.complete_interpreter(findings)
    machine.complete_core_knowledge(core)
    planner_call = machine.start_planner(MODEL_ID)
    machine.complete_planner(planner_call, plan)
    return findings, core, plan


def _complete_retrieval(
    machine: PlannerRagStateMachine,
    plan: KnowledgePlan,
    status: RetrievalStatus = RetrievalStatus.COMPLETE,
) -> RetrievalResult:
    attempt_id = machine.start_retriever(plan.queries)
    retrieval = RetrievalResult(
        retrieval_id="retrieval-1",
        attempt_id=attempt_id,
        status=status,
        queries=plan.queries,
        evidence=[],
    )
    machine.complete_retriever(attempt_id, retrieval)
    return retrieval


class PlannerRagContractsTests(unittest.TestCase):
    def test_formal_mode_can_only_be_planner_rag(self) -> None:
        config = PipelineConfig(
            config_version="v0.1",
            core_knowledge=CoreKnowledgeConfig(bundle_version="core-v1"),
            planner=LlmRoleConfig(model_id=MODEL_ID),
            report_generator=LlmRoleConfig(model_id=MODEL_ID),
        )
        self.assertEqual(config.mode.value, "planner_rag")
        self.assertEqual(config.quality_gate.thresholds, [])
        self.assertEqual(config.quality_gate.conflict_rule_refs, [])

        with self.assertRaises(ValidationError):
            PipelineConfig(
                config_version="v0.1",
                mode="direct_report",
                core_knowledge=CoreKnowledgeConfig(bundle_version="core-v1"),
                planner=LlmRoleConfig(model_id=MODEL_ID),
                report_generator=LlmRoleConfig(model_id=MODEL_ID),
            )

        with self.assertRaises(ValidationError):
            KnowledgePlan.model_validate(
                {**_plan().model_dump(), "needs_retrieval": True}
            )


class PlannerRagStateMachineTests(unittest.TestCase):
    def test_block_records_zero_downstream_calls(self) -> None:
        machine = PlannerRagStateMachine()
        machine.complete_quality_gate(QualityDecision.BLOCK)

        with self.assertRaises(PipelineStateError):
            machine.start_planner(MODEL_ID)
        with self.assertRaises(PipelineStateError):
            machine.start_retriever(_plan().queries)
        with self.assertRaises(PipelineStateError):
            machine.start_report_generator(MODEL_ID)

        self.assertEqual(machine.trace.stage, PipelineStage.BLOCKED)
        self.assertEqual(
            machine.trace.call_count(CallComponent.KNOWLEDGE_PLANNER_LLM), 0
        )
        self.assertEqual(machine.trace.call_count(CallComponent.RETRIEVER), 0)
        self.assertEqual(
            machine.trace.call_count(CallComponent.REPORT_GENERATOR_LLM), 0
        )

    def test_pass_and_review_must_follow_every_stage(self) -> None:
        for decision in (QualityDecision.PASS, QualityDecision.REVIEW):
            with self.subTest(decision=decision):
                machine = PlannerRagStateMachine()
                findings = _findings()
                core = _core()
                plan = _plan()

                machine.complete_quality_gate(decision)
                with self.assertRaises(PipelineStateError):
                    machine.complete_core_knowledge(core)
                with self.assertRaises(PipelineStateError):
                    machine.start_planner(MODEL_ID)

                machine.complete_interpreter(findings)
                machine.complete_core_knowledge(core)
                planner_call = machine.start_planner(MODEL_ID)
                with self.assertRaises(PipelineStateError):
                    machine.start_retriever(plan.queries)
                machine.complete_planner(planner_call, plan)

                retrieval = _complete_retrieval(machine, plan)
                report_input = ReportInputAssembler().assemble(
                    machine=machine,
                    findings=findings,
                    core_knowledge=core,
                    knowledge_plan=plan,
                    retrieval=retrieval,
                )
                report_call = machine.start_report_generator(MODEL_ID)
                machine.complete_report_generator(report_call, "report-1")
                machine.start_validator()
                machine.complete_validator(ValidationDecision.WARNING)

                self.assertEqual(machine.trace.stage, PipelineStage.COMPLETED)
                self.assertEqual(report_input.quality_decision, decision)
                self.assertEqual(
                    [event.sequence for event in machine.trace.events],
                    list(range(1, len(machine.trace.events) + 1)),
                )

    def test_retrieval_completed_is_an_unskippable_report_barrier(self) -> None:
        machine = PlannerRagStateMachine()
        findings, core, plan = _advance_to_plan(machine, QualityDecision.PASS)
        assembler = ReportInputAssembler()

        with self.assertRaises(ReportInputAssemblyError):
            assembler.assemble(
                machine=machine,
                findings=findings,
                core_knowledge=core,
                knowledge_plan=plan,
                retrieval=None,
            )
        with self.assertRaises(PipelineStateError):
            machine.start_report_generator(MODEL_ID)

        attempt_id = machine.start_retriever(plan.queries)
        pending_result = RetrievalResult(
            attempt_id=attempt_id,
            status=RetrievalStatus.COMPLETE,
            queries=plan.queries,
        )
        with self.assertRaises(ReportInputAssemblyError):
            assembler.assemble(
                machine=machine,
                findings=findings,
                core_knowledge=core,
                knowledge_plan=plan,
                retrieval=pending_result,
            )

        machine.complete_retriever(attempt_id, pending_result)
        report_input = assembler.assemble(
            machine=machine,
            findings=findings,
            core_knowledge=core,
            knowledge_plan=plan,
            retrieval=pending_result,
        )
        self.assertEqual(machine.trace.stage, PipelineStage.REPORT_INPUT_READY)
        self.assertEqual(report_input.retrieval_barrier_call_id, attempt_id)

    def test_unavailable_requires_a_real_recorded_attempt(self) -> None:
        machine = PlannerRagStateMachine()
        findings, core, plan = _advance_to_plan(machine, QualityDecision.PASS)
        unproven = RetrievalResult(
            retrieval_id="retrieval-unavailable",
            attempt_id="not-recorded",
            status=RetrievalStatus.UNAVAILABLE,
            queries=plan.queries,
        )

        with self.assertRaises(PipelineStateError):
            machine.complete_retriever("not-recorded", unproven)

        attempt_id = machine.start_retriever(plan.queries)
        unavailable = unproven.model_copy(
            update={"attempt_id": attempt_id},
        )
        machine.complete_retriever(attempt_id, unavailable)
        report_input = ReportInputAssembler().assemble(
            machine=machine,
            findings=findings,
            core_knowledge=core,
            knowledge_plan=plan,
            retrieval=unavailable,
        )

        record = machine.trace.calls_for(CallComponent.RETRIEVER)[0]
        self.assertTrue(record.attempted)
        self.assertEqual(record.request_count, 1)
        self.assertEqual(record.retrieval_status, RetrievalStatus.UNAVAILABLE)
        self.assertEqual(report_input.retrieval.status, RetrievalStatus.UNAVAILABLE)

    def test_findings_or_core_knowledge_cannot_assemble_report_input_directly(self) -> None:
        machine = PlannerRagStateMachine()
        findings, core, plan = _advance_to_plan(machine, QualityDecision.PASS)
        retrieval = _complete_retrieval(machine, plan)
        cases = (
            (findings, None, None, None),
            (None, core, None, None),
            (findings, core, None, None),
        )
        for finding_value, core_value, plan_value, retrieval_value in cases:
            with self.subTest(
                findings=finding_value is not None,
                core=core_value is not None,
            ):
                with self.assertRaises(ReportInputAssemblyError):
                    ReportInputAssembler().assemble(
                        machine=machine,
                        findings=finding_value,
                        core_knowledge=core_value,
                        knowledge_plan=plan_value,
                        retrieval=retrieval_value,
                    )

        self.assertEqual(retrieval.status, RetrievalStatus.COMPLETE)
        self.assertEqual(machine.trace.stage, PipelineStage.RETRIEVAL_COMPLETED)

    def test_same_model_id_creates_independent_llm_call_records(self) -> None:
        machine = PlannerRagStateMachine()
        findings, core, plan = _advance_to_plan(machine, QualityDecision.PASS)
        retrieval = _complete_retrieval(machine, plan)
        ReportInputAssembler().assemble(
            machine=machine,
            findings=findings,
            core_knowledge=core,
            knowledge_plan=plan,
            retrieval=retrieval,
        )
        report_call_id = machine.start_report_generator(MODEL_ID)

        planner_record = machine.trace.calls_for(
            CallComponent.KNOWLEDGE_PLANNER_LLM
        )[0]
        report_record = machine.trace.calls_for(
            CallComponent.REPORT_GENERATOR_LLM
        )[0]
        self.assertEqual(planner_record.model_id, report_record.model_id)
        self.assertNotEqual(planner_record.call_id, report_record.call_id)
        self.assertEqual(report_record.call_id, report_call_id)
        self.assertEqual(len(machine.trace.calls), 3)

        restored = PipelineRunTrace.model_validate_json(machine.trace.model_dump_json())
        self.assertEqual(restored.run_id, machine.trace.run_id)
        self.assertEqual(len(restored.calls), 3)

    def test_multiple_queries_still_create_one_retriever_request(self) -> None:
        machine = PlannerRagStateMachine()
        findings = _findings()
        core = _core()
        base_plan = _plan()
        second_query = RetrievalQuery(
            query_id="query-2",
            topic_id="topic-1",
            text="第二条测试检索查询",
        )
        plan = KnowledgePlan.model_validate(
            {**base_plan.model_dump(), "queries": [*base_plan.queries, second_query]}
        )

        machine.complete_quality_gate(QualityDecision.PASS)
        machine.complete_interpreter(findings)
        machine.complete_core_knowledge(core)
        planner_call = machine.start_planner(MODEL_ID)
        machine.complete_planner(planner_call, plan)
        attempt_id = machine.start_retriever(plan.queries)

        record = machine.trace.calls_for(CallComponent.RETRIEVER)[0]
        self.assertEqual(record.call_id, attempt_id)
        self.assertEqual(record.request_count, 1)
        self.assertEqual(record.batch_query_count, 2)
        self.assertEqual(machine.trace.call_count(CallComponent.RETRIEVER), 1)


if __name__ == "__main__":
    unittest.main()
