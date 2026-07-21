"""Strict execution order for the frozen ``planner_rag`` v0.1 pipeline."""
from __future__ import annotations

from typing import Optional, Sequence

from .contracts import (
    CallComponent,
    CallStatus,
    CoreKnowledgeBundle,
    InterpretationResult,
    KnowledgePlan,
    PipelineCallRecord,
    PipelineRunTrace,
    PipelineStage,
    PipelineStateEvent,
    QualityDecision,
    ReportGenerationInput,
    RetrievalQuery,
    RetrievalResult,
    ValidationDecision,
    utc_now,
)


class PipelineStateError(RuntimeError):
    """Raised when a caller attempts to bypass a planner_rag stage."""


class PlannerRagStateMachine:
    def __init__(self, trace: Optional[PipelineRunTrace] = None):
        self.trace = trace or PipelineRunTrace()

    def _require(self, expected: PipelineStage) -> None:
        if self.trace.stage != expected:
            raise PipelineStateError(
                f"expected stage {expected.value}, current stage is {self.trace.stage.value}"
            )

    def _transition(self, to_stage: PipelineStage, reason: str) -> None:
        event = PipelineStateEvent(
            sequence=len(self.trace.events) + 1,
            from_stage=self.trace.stage,
            to_stage=to_stage,
            reason=reason,
        )
        self.trace.events.append(event)
        self.trace.stage = to_stage
        self.trace.updated_at = utc_now()

    def _start_call(
        self,
        component: CallComponent,
        *,
        model_id: Optional[str] = None,
        batch_query_count: Optional[int] = None,
    ) -> PipelineCallRecord:
        record = PipelineCallRecord(
            component=component,
            model_id=model_id,
            batch_query_count=batch_query_count,
        )
        self.trace.calls.append(record)
        self.trace.updated_at = utc_now()
        return record

    def _running_call(
        self, call_id: str, component: CallComponent
    ) -> PipelineCallRecord:
        matches = [record for record in self.trace.calls if record.call_id == call_id]
        if len(matches) != 1:
            raise PipelineStateError(f"unknown call_id: {call_id}")
        record = matches[0]
        if record.component != component or record.status != CallStatus.STARTED:
            raise PipelineStateError(f"call {call_id} is not a running {component.value}")
        return record

    def complete_quality_gate(self, decision: QualityDecision) -> None:
        self._require(PipelineStage.CREATED)
        self.trace.quality_decision = decision
        if decision == QualityDecision.BLOCK:
            self._transition(PipelineStage.BLOCKED, "quality_gate_blocked")
        else:
            self._transition(
                PipelineStage.QUALITY_GATE_COMPLETED,
                f"quality_gate_{decision.value}",
            )

    def complete_interpreter(self, result: InterpretationResult) -> None:
        self._require(PipelineStage.QUALITY_GATE_COMPLETED)
        self.trace.artifact_refs["interpretation"] = result.interpretation_id
        self._transition(PipelineStage.INTERPRETED, "interpreter_completed")

    def complete_core_knowledge(self, bundle: CoreKnowledgeBundle) -> None:
        self._require(PipelineStage.INTERPRETED)
        self.trace.artifact_refs["core_knowledge"] = bundle.bundle_id
        self._transition(
            PipelineStage.CORE_KNOWLEDGE_READY,
            "core_knowledge_ready",
        )

    def start_planner(self, model_id: str) -> str:
        self._require(PipelineStage.CORE_KNOWLEDGE_READY)
        record = self._start_call(
            CallComponent.KNOWLEDGE_PLANNER_LLM,
            model_id=model_id,
        )
        self._transition(PipelineStage.PLANNER_RUNNING, "planner_llm_started")
        return record.call_id

    def complete_planner(self, call_id: str, plan: KnowledgePlan) -> None:
        self._require(PipelineStage.PLANNER_RUNNING)
        record = self._running_call(call_id, CallComponent.KNOWLEDGE_PLANNER_LLM)
        if record.model_id != plan.planner_model_id:
            raise PipelineStateError("planner model_id does not match its call record")
        record.completed_at = utc_now()
        record.status = CallStatus.COMPLETED
        self.trace.artifact_refs["knowledge_plan"] = plan.plan_id
        self._transition(PipelineStage.PLANNER_COMPLETED, "planner_llm_completed")

    def start_retriever(self, queries: Sequence[RetrievalQuery]) -> str:
        self._require(PipelineStage.PLANNER_COMPLETED)
        if not queries:
            raise PipelineStateError("retriever requires at least one planner query")
        if self.trace.call_count(CallComponent.RETRIEVER):
            raise PipelineStateError("planner_rag v0.1 allows one retrieval request")
        record = self._start_call(
            CallComponent.RETRIEVER,
            batch_query_count=len(queries),
        )
        self._transition(PipelineStage.RETRIEVAL_RUNNING, "retriever_attempt_started")
        return record.call_id

    def complete_retriever(self, call_id: str, result: RetrievalResult) -> None:
        self._require(PipelineStage.RETRIEVAL_RUNNING)
        record = self._running_call(call_id, CallComponent.RETRIEVER)
        if result.attempt_id != call_id:
            raise PipelineStateError("retrieval result does not match the recorded attempt")
        if record.batch_query_count != len(result.queries):
            raise PipelineStateError("retrieval result query count does not match the batch")
        record.retrieval_status = result.status
        record.completed_at = utc_now()
        record.status = CallStatus.COMPLETED
        self.trace.artifact_refs["retrieval"] = result.retrieval_id
        self._transition(PipelineStage.RETRIEVAL_COMPLETED, "retriever_completed")

    def mark_report_input_ready(self, value: ReportGenerationInput) -> None:
        self._require(PipelineStage.RETRIEVAL_COMPLETED)
        if value.run_id != self.trace.run_id:
            raise PipelineStateError("report input belongs to a different pipeline run")
        self.trace.artifact_refs["report_input"] = value.input_id
        self._transition(PipelineStage.REPORT_INPUT_READY, "report_input_assembled")

    def start_report_generator(self, model_id: str) -> str:
        self._require(PipelineStage.REPORT_INPUT_READY)
        record = self._start_call(
            CallComponent.REPORT_GENERATOR_LLM,
            model_id=model_id,
        )
        self._transition(
            PipelineStage.REPORT_GENERATOR_RUNNING,
            "report_generator_llm_started",
        )
        return record.call_id

    def complete_report_generator(self, call_id: str, report_ref: str) -> None:
        self._require(PipelineStage.REPORT_GENERATOR_RUNNING)
        record = self._running_call(call_id, CallComponent.REPORT_GENERATOR_LLM)
        if not report_ref.strip():
            raise PipelineStateError("report_ref must not be empty")
        record.completed_at = utc_now()
        record.status = CallStatus.COMPLETED
        self.trace.artifact_refs["report"] = report_ref
        self._transition(PipelineStage.REPORT_GENERATED, "report_generator_llm_completed")

    def start_validator(self) -> None:
        self._require(PipelineStage.REPORT_GENERATED)
        self._transition(PipelineStage.VALIDATOR_RUNNING, "validator_started")

    def complete_validator(self, decision: ValidationDecision) -> None:
        self._require(PipelineStage.VALIDATOR_RUNNING)
        self.trace.validation_decision = decision
        self._transition(PipelineStage.COMPLETED, f"validator_{decision.value}")


__all__ = ["PipelineStateError", "PlannerRagStateMachine"]
