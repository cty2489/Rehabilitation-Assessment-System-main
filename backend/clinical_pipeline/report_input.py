"""Report input assembly guarded by the completed Retriever audit record."""
from __future__ import annotations

from typing import Optional

from .contracts import (
    CallComponent,
    CallStatus,
    CoreKnowledgeBundle,
    InterpretationResult,
    KnowledgePlan,
    PipelineStage,
    ReportGenerationInput,
    RetrievalResult,
)
from .state_machine import PipelineStateError, PlannerRagStateMachine


class ReportInputAssemblyError(PipelineStateError):
    """Raised when report input is assembled before the retrieval barrier."""


class ReportInputAssembler:
    def assemble(
        self,
        *,
        machine: PlannerRagStateMachine,
        findings: Optional[InterpretationResult],
        core_knowledge: Optional[CoreKnowledgeBundle],
        knowledge_plan: Optional[KnowledgePlan],
        retrieval: Optional[RetrievalResult],
    ) -> ReportGenerationInput:
        trace = machine.trace
        if trace.stage != PipelineStage.RETRIEVAL_COMPLETED:
            raise ReportInputAssemblyError(
                "ReportGenerator input requires the retrieval_completed barrier"
            )
        if findings is None or core_knowledge is None:
            raise ReportInputAssemblyError(
                "findings and fixed core knowledge cannot bypass Retriever"
            )
        if knowledge_plan is None or retrieval is None:
            raise ReportInputAssemblyError(
                "knowledge plan and completed retrieval are required"
            )
        if trace.quality_decision is None or trace.quality_decision.value == "block":
            raise ReportInputAssemblyError("blocked runs cannot assemble report input")

        expected_refs = {
            "interpretation": findings.interpretation_id,
            "core_knowledge": core_knowledge.bundle_id,
            "knowledge_plan": knowledge_plan.plan_id,
            "retrieval": retrieval.retrieval_id,
        }
        for key, value in expected_refs.items():
            if trace.artifact_refs.get(key) != value:
                raise ReportInputAssemblyError(f"{key} does not belong to this pipeline run")

        planner_calls = trace.calls_for(CallComponent.KNOWLEDGE_PLANNER_LLM)
        if len(planner_calls) != 1 or planner_calls[0].status != CallStatus.COMPLETED:
            raise ReportInputAssemblyError("a completed Planner LLM call is required")

        retrieval_calls = trace.calls_for(CallComponent.RETRIEVER)
        if len(retrieval_calls) != 1:
            raise ReportInputAssemblyError("exactly one Retriever attempt is required")
        retrieval_call = retrieval_calls[0]
        if (
            retrieval_call.status != CallStatus.COMPLETED
            or not retrieval_call.attempted
            or retrieval_call.request_count != 1
            or retrieval_call.call_id != retrieval.attempt_id
            or retrieval_call.retrieval_status != retrieval.status
        ):
            raise ReportInputAssemblyError(
                "Retriever completion is not proven by the pipeline audit trace"
            )

        value = ReportGenerationInput(
            run_id=trace.run_id,
            quality_decision=trace.quality_decision,
            findings=findings,
            core_knowledge=core_knowledge,
            knowledge_plan=knowledge_plan,
            retrieval=retrieval,
            retrieval_barrier_call_id=retrieval_call.call_id,
        )
        machine.mark_report_input_ready(value)
        return value


__all__ = ["ReportInputAssembler", "ReportInputAssemblyError"]
