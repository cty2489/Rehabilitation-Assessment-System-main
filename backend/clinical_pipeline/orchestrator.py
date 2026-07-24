"""Non-production orchestrator for the minimal ``planner_rag`` v0.1 flow."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import Field

from .config import PipelineConfig
from .contracts import (
    CallStatus,
    CanonicalAssessmentContext,
    CanonicalBiomarker,
    CanonicalPatientInfo,
    CanonicalPredictions,
    ContractModel,
    CoreKnowledgeBundle,
    InterpretationResult,
    KnowledgePlan,
    PipelineRunTrace,
    QualityDecision,
    ReportGenerationInput,
    RetrievalResult,
    ValidationDecision,
    utc_now,
)
from .core_knowledge import CoreKnowledgeProvider
from .interpreter import Interpreter
from .knowledge_planner import KnowledgePlanner
from .report_generator import ReportGenerator, ReportResult
from .report_input import ReportInputAssembler
from .retriever import Retriever
from .state_machine import PlannerRagStateMachine
from .validator import ValidationResult, ValidationStatus, Validator


class PipelinePatientInput(ContractModel):
    patient_id: Optional[str] = Field(default=None, max_length=64)
    age: Optional[int] = None
    sex: Optional[str] = Field(default=None, max_length=32)
    diagnosis: Optional[str] = Field(default=None, max_length=255)
    disease_days: Optional[int] = None
    paralysis_side: Optional[str] = Field(default=None, max_length=32)


class PipelineAssessmentInput(ContractModel):
    schema_version: Literal["rehab.pipeline-assessment-input.v1"] = (
        "rehab.pipeline-assessment-input.v1"
    )
    context_id: str = Field(default_factory=lambda: f"context-{uuid4().hex}")
    assessment_id: Optional[str] = Field(default=None, max_length=128)
    patient: Optional[PipelinePatientInput] = None
    predictions: Optional[CanonicalPredictions] = None
    biomarkers: Optional[List[CanonicalBiomarker]] = None
    quality_metadata: Dict[str, Any] = Field(default_factory=dict)


class QualityGateIssue(ContractModel):
    code: Literal[
        "missing_patient_id",
        "predictions_missing",
        "biomarkers_missing",
    ]
    level: Literal["block", "review"]
    message: str = Field(min_length=1)


class QualityGateResult(ContractModel):
    schema_version: Literal["rehab.quality-gate-result.v1"] = (
        "rehab.quality-gate-result.v1"
    )
    decision: QualityDecision
    issues: List[QualityGateIssue] = Field(default_factory=list)


class QualityGate:
    """Apply only presence checks; no clinical thresholds are used."""

    def evaluate(self, value: PipelineAssessmentInput) -> QualityGateResult:
        if not isinstance(value, PipelineAssessmentInput):
            raise TypeError("QualityGate输入必须是PipelineAssessmentInput")

        issues: list[QualityGateIssue] = []
        patient_id = str(value.patient.patient_id or "").strip() if value.patient else ""
        if not patient_id:
            issues.append(
                QualityGateIssue(
                    code="missing_patient_id",
                    level="block",
                    message="缺少患者标识，流程已阻断。",
                )
            )

        predictions = value.predictions
        predictions_missing = predictions is None or all(
            item is None
            for item in (
                predictions.FMA_UE if predictions else None,
                predictions.hand_tone if predictions else None,
                predictions.hand_function if predictions else None,
            )
        )
        if predictions_missing:
            issues.append(
                QualityGateIssue(
                    code="predictions_missing",
                    level="block",
                    message="模型预测结果完全缺失，流程已阻断。",
                )
            )

        if not value.biomarkers:
            issues.append(
                QualityGateIssue(
                    code="biomarkers_missing",
                    level="review",
                    message="生物标志物完全缺失，流程继续但必须复核。",
                )
            )

        if any(issue.level == "block" for issue in issues):
            decision = QualityDecision.BLOCK
        elif issues:
            decision = QualityDecision.REVIEW
        else:
            decision = QualityDecision.PASS
        return QualityGateResult(decision=decision, issues=issues)


class PipelineModule(str, Enum):
    QUALITY_GATE = "QualityGate"
    INTERPRETER = "Interpreter"
    CORE_KNOWLEDGE_PROVIDER = "CoreKnowledgeProvider"
    KNOWLEDGE_PLANNER = "KnowledgePlanner"
    RETRIEVER = "Retriever"
    REPORT_INPUT_ASSEMBLER = "ReportInputAssembler"
    REPORT_GENERATOR = "ReportGenerator"
    VALIDATOR = "Validator"


class ModuleExecutionStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class ModuleExecutionEvent(ContractModel):
    event_id: str = Field(default_factory=lambda: f"module-event-{uuid4().hex}")
    sequence: int = Field(ge=1)
    module: PipelineModule
    status: ModuleExecutionStatus
    detail: Optional[str] = Field(default=None, max_length=500)
    created_at: datetime = Field(default_factory=utc_now)


class PipelineRunStatus(str, Enum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class PipelineFailure(ContractModel):
    module: PipelineModule
    error_type: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=500)


class OrchestrationResult(ContractModel):
    schema_version: Literal["rehab.pipeline-orchestration-result.v1"] = (
        "rehab.pipeline-orchestration-result.v1"
    )
    status: PipelineRunStatus
    trace: PipelineRunTrace
    module_events: List[ModuleExecutionEvent] = Field(default_factory=list)
    quality_gate: Optional[QualityGateResult] = None
    canonical_context: Optional[CanonicalAssessmentContext] = None
    interpretation: Optional[InterpretationResult] = None
    core_knowledge: Optional[CoreKnowledgeBundle] = None
    knowledge_plan: Optional[KnowledgePlan] = None
    retrieval: Optional[RetrievalResult] = None
    report_input: Optional[ReportGenerationInput] = None
    report: Optional[ReportResult] = None
    validation: Optional[ValidationResult] = None
    planner_call_id: Optional[str] = Field(default=None, max_length=128)
    retriever_attempt_id: Optional[str] = Field(default=None, max_length=128)
    report_generator_call_id: Optional[str] = Field(default=None, max_length=128)
    block_reasons: List[str] = Field(default_factory=list)
    failure: Optional[PipelineFailure] = None


class ClinicalPipelineOrchestrator:
    """Run all frozen v0.1 modules without touching production integrations."""

    def __init__(
        self,
        *,
        config: PipelineConfig,
        quality_gate: Optional[QualityGate] = None,
        interpreter: Optional[Interpreter] = None,
        core_knowledge_provider: Optional[CoreKnowledgeProvider] = None,
        knowledge_planner: Optional[KnowledgePlanner] = None,
        retriever: Optional[Retriever] = None,
        report_input_assembler: Optional[ReportInputAssembler] = None,
        report_generator: Optional[ReportGenerator] = None,
        validator: Optional[Validator] = None,
    ) -> None:
        self._config = config
        self._quality_gate = quality_gate or QualityGate()
        self._interpreter = interpreter or Interpreter()
        self._core_knowledge_provider = (
            core_knowledge_provider or CoreKnowledgeProvider()
        )
        self._knowledge_planner = knowledge_planner or KnowledgePlanner()
        self._retriever = retriever or Retriever()
        self._report_input_assembler = (
            report_input_assembler or ReportInputAssembler()
        )
        self._report_generator = report_generator or ReportGenerator()
        self._validator = validator or Validator()

    def run(self, value: PipelineAssessmentInput) -> OrchestrationResult:
        if not isinstance(value, PipelineAssessmentInput):
            raise TypeError("Orchestrator输入必须是PipelineAssessmentInput")

        trace = PipelineRunTrace()
        machine = PlannerRagStateMachine(trace)
        module_events: list[ModuleExecutionEvent] = []
        quality_gate_result: Optional[QualityGateResult] = None
        canonical_context: Optional[CanonicalAssessmentContext] = None
        interpretation: Optional[InterpretationResult] = None
        core_knowledge: Optional[CoreKnowledgeBundle] = None
        knowledge_plan: Optional[KnowledgePlan] = None
        retrieval: Optional[RetrievalResult] = None
        report_input: Optional[ReportGenerationInput] = None
        report: Optional[ReportResult] = None
        validation: Optional[ValidationResult] = None
        planner_call_id: Optional[str] = None
        retriever_attempt_id: Optional[str] = None
        report_generator_call_id: Optional[str] = None
        current_module: Optional[PipelineModule] = None
        active_call_id: Optional[str] = None

        try:
            current_module = PipelineModule.QUALITY_GATE
            self._record_module_event(trace, module_events, current_module, "started")
            quality_gate_result = self._quality_gate.evaluate(value)
            machine.complete_quality_gate(quality_gate_result.decision)
            trace.artifact_refs["quality_gate_decision"] = (
                quality_gate_result.decision.value
            )
            self._record_module_event(
                trace,
                module_events,
                current_module,
                "completed",
                detail=quality_gate_result.decision.value,
            )
            current_module = None

            if quality_gate_result.decision == QualityDecision.BLOCK:
                reasons = [
                    issue.message
                    for issue in quality_gate_result.issues
                    if issue.level == "block"
                ]
                return self._result(
                    status=PipelineRunStatus.BLOCKED,
                    trace=trace,
                    module_events=module_events,
                    quality_gate=quality_gate_result,
                    block_reasons=reasons,
                )

            current_module = PipelineModule.INTERPRETER
            self._record_module_event(trace, module_events, current_module, "started")
            canonical_context = self._canonical_context(value, quality_gate_result)
            trace.artifact_refs["canonical_context"] = canonical_context.context_id
            interpretation = self._interpreter.interpret(canonical_context)
            machine.complete_interpreter(interpretation)
            self._record_module_event(
                trace, module_events, current_module, "completed"
            )
            current_module = None

            current_module = PipelineModule.CORE_KNOWLEDGE_PROVIDER
            self._record_module_event(trace, module_events, current_module, "started")
            system_keys = list(
                dict.fromkeys(finding.metric_key for finding in interpretation.findings)
            )
            core_knowledge = self._core_knowledge_provider.provide(system_keys)
            machine.complete_core_knowledge(core_knowledge)
            self._record_module_event(
                trace, module_events, current_module, "completed"
            )
            current_module = None

            current_module = PipelineModule.KNOWLEDGE_PLANNER
            self._record_module_event(trace, module_events, current_module, "started")
            planner_call_id = machine.start_planner(self._config.planner.model_id)
            active_call_id = planner_call_id
            trace.artifact_refs["planner_call_id"] = planner_call_id
            knowledge_plan = self._knowledge_planner.plan(
                interpretation,
                core_knowledge,
            )
            machine.complete_planner(planner_call_id, knowledge_plan)
            active_call_id = None
            self._record_module_event(
                trace, module_events, current_module, "completed"
            )
            current_module = None

            current_module = PipelineModule.RETRIEVER
            self._record_module_event(trace, module_events, current_module, "started")
            retriever_attempt_id = machine.start_retriever(knowledge_plan.queries)
            active_call_id = retriever_attempt_id
            trace.artifact_refs["retriever_attempt_id"] = retriever_attempt_id
            retrieval = self._retriever.retrieve(
                knowledge_plan,
                attempt_id=retriever_attempt_id,
            )
            machine.complete_retriever(retriever_attempt_id, retrieval)
            active_call_id = None
            self._record_module_event(
                trace,
                module_events,
                current_module,
                "completed",
                detail=retrieval.status.value,
            )
            current_module = None

            current_module = PipelineModule.REPORT_INPUT_ASSEMBLER
            self._record_module_event(trace, module_events, current_module, "started")
            report_input = self._report_input_assembler.assemble(
                machine=machine,
                findings=interpretation,
                core_knowledge=core_knowledge,
                knowledge_plan=knowledge_plan,
                retrieval=retrieval,
            )
            self._record_module_event(
                trace, module_events, current_module, "completed"
            )
            current_module = None

            current_module = PipelineModule.REPORT_GENERATOR
            self._record_module_event(trace, module_events, current_module, "started")
            report_generator_call_id = machine.start_report_generator(
                self._config.report_generator.model_id
            )
            active_call_id = report_generator_call_id
            trace.artifact_refs["report_generator_call_id"] = (
                report_generator_call_id
            )
            report = self._report_generator.generate(report_input)
            if report.report_model_id != self._config.report_generator.model_id:
                raise ValueError("ReportGenerator model_id与PipelineConfig不一致")
            machine.complete_report_generator(
                report_generator_call_id,
                report.report_id,
            )
            active_call_id = None
            self._record_module_event(
                trace, module_events, current_module, "completed"
            )
            current_module = None

            current_module = PipelineModule.VALIDATOR
            self._record_module_event(trace, module_events, current_module, "started")
            machine.start_validator()
            validation = self._validator.validate(report, report_input)
            if (
                quality_gate_result.decision == QualityDecision.REVIEW
                and validation.status == ValidationStatus.PASSED
            ):
                validation = validation.model_copy(
                    update={"status": ValidationStatus.WARNING}
                )
                trace.artifact_refs["quality_gate_review_forced_warning"] = "true"
            trace.artifact_refs["validation_result"] = validation.validation_id
            trace.artifact_refs["validation_status"] = validation.status.value
            machine.complete_validator(self._validation_decision(validation.status))
            self._record_module_event(
                trace,
                module_events,
                current_module,
                "completed",
                detail=validation.status.value,
            )
            current_module = None

            return self._result(
                status=PipelineRunStatus.COMPLETED,
                trace=trace,
                module_events=module_events,
                quality_gate=quality_gate_result,
                canonical_context=canonical_context,
                interpretation=interpretation,
                core_knowledge=core_knowledge,
                knowledge_plan=knowledge_plan,
                retrieval=retrieval,
                report_input=report_input,
                report=report,
                validation=validation,
                planner_call_id=planner_call_id,
                retriever_attempt_id=retriever_attempt_id,
                report_generator_call_id=report_generator_call_id,
            )
        except Exception as exc:  # noqa: BLE001 - return an auditable failed run
            failed_module = current_module or PipelineModule.QUALITY_GATE
            if active_call_id:
                self._mark_call_failed(trace, active_call_id)
            self._record_module_event(
                trace,
                module_events,
                failed_module,
                "failed",
                detail=f"{type(exc).__name__}: {str(exc)[:400]}",
            )
            trace.artifact_refs["failure_stage"] = failed_module.value
            trace.artifact_refs["failure_error_type"] = type(exc).__name__
            message = str(exc).strip() or "模块执行失败"
            return self._result(
                status=PipelineRunStatus.FAILED,
                trace=trace,
                module_events=module_events,
                quality_gate=quality_gate_result,
                canonical_context=canonical_context,
                interpretation=interpretation,
                core_knowledge=core_knowledge,
                knowledge_plan=knowledge_plan,
                retrieval=retrieval,
                report_input=report_input,
                report=report,
                validation=validation,
                planner_call_id=planner_call_id,
                retriever_attempt_id=retriever_attempt_id,
                report_generator_call_id=report_generator_call_id,
                failure=PipelineFailure(
                    module=failed_module,
                    error_type=type(exc).__name__,
                    message=message[:500],
                ),
            )

    @staticmethod
    def _canonical_context(
        value: PipelineAssessmentInput,
        quality_gate: QualityGateResult,
    ) -> CanonicalAssessmentContext:
        if value.patient is None or value.predictions is None:
            raise ValueError("通过QualityGate后患者和predictions必须存在")
        patient_id = str(value.patient.patient_id or "").strip()
        metadata = dict(value.quality_metadata)
        metadata["quality_gate_issues"] = [
            issue.model_dump(mode="json") for issue in quality_gate.issues
        ]
        return CanonicalAssessmentContext(
            context_id=value.context_id,
            assessment_id=value.assessment_id,
            quality_decision=quality_gate.decision,
            patient=CanonicalPatientInfo(
                patient_id=patient_id,
                age=value.patient.age,
                sex=value.patient.sex,
                diagnosis=value.patient.diagnosis,
                disease_days=value.patient.disease_days,
                paralysis_side=value.patient.paralysis_side,
            ),
            predictions=value.predictions,
            biomarkers=list(value.biomarkers or []),
            quality_metadata=metadata,
        )

    @staticmethod
    def _validation_decision(status: ValidationStatus) -> ValidationDecision:
        return {
            ValidationStatus.PASSED: ValidationDecision.PASS,
            ValidationStatus.WARNING: ValidationDecision.WARNING,
            ValidationStatus.MANUAL_REVIEW: ValidationDecision.MANUAL_REVIEW,
        }[status]

    @staticmethod
    def _mark_call_failed(trace: PipelineRunTrace, call_id: str) -> None:
        for record in trace.calls:
            if record.call_id == call_id and record.status == CallStatus.STARTED:
                record.status = CallStatus.FAILED
                record.completed_at = utc_now()
                trace.updated_at = utc_now()
                return

    @staticmethod
    def _record_module_event(
        trace: PipelineRunTrace,
        events: list[ModuleExecutionEvent],
        module: PipelineModule,
        status: Literal["started", "completed", "failed"],
        *,
        detail: Optional[str] = None,
    ) -> None:
        event = ModuleExecutionEvent(
            sequence=len(events) + 1,
            module=module,
            status=ModuleExecutionStatus(status),
            detail=detail,
        )
        events.append(event)
        event_key = (
            f"module_event:{event.sequence:03d}:"
            f"{event.module.value}:{event.status.value}"
        )
        trace.artifact_refs[event_key] = event.event_id
        trace.artifact_refs[f"module_status:{module.value}"] = event.status.value
        trace.updated_at = utc_now()

    @staticmethod
    def _result(
        *,
        status: PipelineRunStatus,
        trace: PipelineRunTrace,
        module_events: list[ModuleExecutionEvent],
        quality_gate: Optional[QualityGateResult] = None,
        canonical_context: Optional[CanonicalAssessmentContext] = None,
        interpretation: Optional[InterpretationResult] = None,
        core_knowledge: Optional[CoreKnowledgeBundle] = None,
        knowledge_plan: Optional[KnowledgePlan] = None,
        retrieval: Optional[RetrievalResult] = None,
        report_input: Optional[ReportGenerationInput] = None,
        report: Optional[ReportResult] = None,
        validation: Optional[ValidationResult] = None,
        planner_call_id: Optional[str] = None,
        retriever_attempt_id: Optional[str] = None,
        report_generator_call_id: Optional[str] = None,
        block_reasons: Optional[list[str]] = None,
        failure: Optional[PipelineFailure] = None,
    ) -> OrchestrationResult:
        return OrchestrationResult(
            status=status,
            trace=trace,
            module_events=module_events,
            quality_gate=quality_gate,
            canonical_context=canonical_context,
            interpretation=interpretation,
            core_knowledge=core_knowledge,
            knowledge_plan=knowledge_plan,
            retrieval=retrieval,
            report_input=report_input,
            report=report,
            validation=validation,
            planner_call_id=planner_call_id,
            retriever_attempt_id=retriever_attempt_id,
            report_generator_call_id=report_generator_call_id,
            block_reasons=block_reasons or [],
            failure=failure,
        )


__all__ = [
    "ClinicalPipelineOrchestrator",
    "ModuleExecutionEvent",
    "ModuleExecutionStatus",
    "OrchestrationResult",
    "PipelineAssessmentInput",
    "PipelineFailure",
    "PipelineModule",
    "PipelinePatientInput",
    "PipelineRunStatus",
    "QualityGate",
    "QualityGateIssue",
    "QualityGateResult",
]
