"""Versioned data contracts for the frozen ``planner_rag`` v0.1 pipeline.

This module intentionally contains no clinical thresholds and performs no model,
retrieval, persistence, or report-generation work.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PipelineMode(str, Enum):
    PLANNER_RAG = "planner_rag"


class QualityDecision(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    BLOCK = "block"


class PipelineStage(str, Enum):
    CREATED = "created"
    QUALITY_GATE_COMPLETED = "quality_gate_completed"
    BLOCKED = "blocked"
    INTERPRETED = "interpreted"
    CORE_KNOWLEDGE_READY = "core_knowledge_ready"
    PLANNER_RUNNING = "planner_running"
    PLANNER_COMPLETED = "planner_completed"
    RETRIEVAL_RUNNING = "retrieval_running"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    REPORT_INPUT_READY = "report_input_ready"
    REPORT_GENERATOR_RUNNING = "report_generator_running"
    REPORT_GENERATED = "report_generated"
    VALIDATOR_RUNNING = "validator_running"
    COMPLETED = "completed"


class CallComponent(str, Enum):
    KNOWLEDGE_PLANNER_LLM = "knowledge_planner_llm"
    RETRIEVER = "retriever"
    REPORT_GENERATOR_LLM = "report_generator_llm"


class CallStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class RetrievalStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    UNAVAILABLE = "unavailable"


class ValidationDecision(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    MANUAL_REVIEW = "manual_review"


class FindingStatus(str, Enum):
    OBSERVED = "observed"
    WITHIN_REFERENCE = "within_reference"
    BELOW_REFERENCE = "below_reference"
    ABOVE_REFERENCE = "above_reference"
    DIRECTION_ONLY = "direction_only"
    NOT_CLASSIFIABLE = "not_classifiable"
    MISSING = "missing"


class FindingSeverity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class FindingModality(str, Enum):
    CLINICAL_SCALE = "clinical_scale"
    EEG = "eeg"
    EMG = "emg"
    IMU = "imu"
    MULTIMODAL = "multimodal"


class Finding(ContractModel):
    finding_id: str = Field(min_length=1, max_length=128)
    metric_key: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    value: Any = None
    unit: Optional[str] = Field(default=None, max_length=64)
    status: FindingStatus
    severity: FindingSeverity = FindingSeverity.UNKNOWN
    modality: FindingModality


class KnownCombination(ContractModel):
    combination_id: str = Field(min_length=1, max_length=128)
    finding_ids: List[str] = Field(min_length=2)
    relation: str = Field(min_length=1, max_length=128)
    rule_id: str = Field(min_length=1, max_length=128)


class InterpretationResult(ContractModel):
    schema_version: Literal["rehab.interpretation.v1"] = "rehab.interpretation.v1"
    interpretation_id: str = Field(default_factory=lambda: f"interpretation-{uuid4().hex}")
    findings: List[Finding] = Field(min_length=1)
    known_combinations: List[KnownCombination] = Field(default_factory=list)


class CoreKnowledgeEntry(ContractModel):
    knowledge_id: str = Field(min_length=1, max_length=128)
    system_key: str = Field(min_length=1, max_length=128)
    allowed_interpretation: str = Field(min_length=1)
    prohibited_interpretation: Optional[str] = None
    source_ids: List[str] = Field(default_factory=list)


class CoreKnowledgeBundle(ContractModel):
    schema_version: Literal["rehab.core-knowledge.v1"] = "rehab.core-knowledge.v1"
    bundle_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    entries: List[CoreKnowledgeEntry] = Field(min_length=1)


class KnowledgeTopic(ContractModel):
    topic_id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=255)
    finding_ids: List[str] = Field(min_length=1)
    priority: Literal["high", "medium", "low"] = "medium"


class RetrievalQuery(ContractModel):
    query_id: str = Field(min_length=1, max_length=128)
    topic_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=4000)


class KnowledgePlan(ContractModel):
    schema_version: Literal["rehab.knowledge-plan.v1"] = "rehab.knowledge-plan.v1"
    plan_id: str = Field(default_factory=lambda: f"plan-{uuid4().hex}")
    planner_model_id: str = Field(min_length=1, max_length=255)
    topics: List[KnowledgeTopic] = Field(min_length=1)
    queries: List[RetrievalQuery] = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=1000)
    generation_mode: Literal["llm", "fallback"] = "llm"


class RetrievalEvidence(ContractModel):
    evidence_id: str = Field(min_length=1, max_length=128)
    query_id: str = Field(min_length=1, max_length=128)
    chunk_id: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1)
    rank: int = Field(ge=1)
    raw_score: float
    source_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(ContractModel):
    schema_version: Literal["rehab.pipeline.evidence.v1"] = "rehab.pipeline.evidence.v1"
    retrieval_id: str = Field(default_factory=lambda: f"retrieval-{uuid4().hex}")
    attempt_id: str = Field(min_length=1, max_length=128)
    attempted: Literal[True] = True
    request_count: Literal[1] = 1
    status: RetrievalStatus
    queries: List[RetrievalQuery] = Field(min_length=1)
    collection: Optional[str] = Field(default=None, max_length=255)
    evidence: List[RetrievalEvidence] = Field(default_factory=list)
    covered_topic_ids: List[str] = Field(default_factory=list)
    uncovered_topic_ids: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unavailable_has_no_evidence(self) -> "RetrievalResult":
        if self.status == RetrievalStatus.UNAVAILABLE and self.evidence:
            raise ValueError("unavailable retrieval cannot contain evidence")
        return self


class ReportGenerationInput(ContractModel):
    schema_version: Literal["rehab.report-generation-input.v1"] = (
        "rehab.report-generation-input.v1"
    )
    input_id: str = Field(default_factory=lambda: f"report-input-{uuid4().hex}")
    run_id: str = Field(min_length=1, max_length=128)
    mode: Literal[PipelineMode.PLANNER_RAG] = PipelineMode.PLANNER_RAG
    quality_decision: Literal[QualityDecision.PASS, QualityDecision.REVIEW]
    findings: InterpretationResult
    core_knowledge: CoreKnowledgeBundle
    knowledge_plan: KnowledgePlan
    retrieval: RetrievalResult
    retrieval_barrier_call_id: str = Field(min_length=1, max_length=128)
    assembled_at: datetime = Field(default_factory=utc_now)


class PipelineStateEvent(ContractModel):
    sequence: int = Field(ge=1)
    from_stage: PipelineStage
    to_stage: PipelineStage
    reason: str = Field(min_length=1, max_length=255)
    created_at: datetime = Field(default_factory=utc_now)


class PipelineCallRecord(ContractModel):
    call_id: str = Field(default_factory=lambda: f"call-{uuid4().hex}")
    component: CallComponent
    status: CallStatus = CallStatus.STARTED
    model_id: Optional[str] = Field(default=None, max_length=255)
    attempted: bool = True
    request_count: int = Field(default=1, ge=1)
    batch_query_count: Optional[int] = Field(default=None, ge=1)
    retrieval_status: Optional[RetrievalStatus] = None
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_component_fields(self) -> "PipelineCallRecord":
        if self.component in {
            CallComponent.KNOWLEDGE_PLANNER_LLM,
            CallComponent.REPORT_GENERATOR_LLM,
        } and not self.model_id:
            raise ValueError("LLM call records require model_id")
        if self.component == CallComponent.RETRIEVER and self.request_count != 1:
            raise ValueError("planner_rag v0.1 allows exactly one retrieval request")
        if self.status == CallStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed call records require completed_at")
        return self


class PipelineRunTrace(ContractModel):
    schema_version: Literal["rehab.pipeline-run-trace.v1"] = (
        "rehab.pipeline-run-trace.v1"
    )
    run_id: str = Field(default_factory=lambda: f"pipeline-{uuid4().hex}")
    mode: Literal[PipelineMode.PLANNER_RAG] = PipelineMode.PLANNER_RAG
    stage: PipelineStage = PipelineStage.CREATED
    quality_decision: Optional[QualityDecision] = None
    events: List[PipelineStateEvent] = Field(default_factory=list)
    calls: List[PipelineCallRecord] = Field(default_factory=list)
    artifact_refs: Dict[str, str] = Field(default_factory=dict)
    validation_decision: Optional[ValidationDecision] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def calls_for(self, component: CallComponent) -> List[PipelineCallRecord]:
        return [record for record in self.calls if record.component == component]

    def call_count(self, component: CallComponent) -> int:
        return len(self.calls_for(component))


__all__ = [
    "CallComponent",
    "CallStatus",
    "CoreKnowledgeBundle",
    "CoreKnowledgeEntry",
    "Finding",
    "FindingModality",
    "FindingSeverity",
    "FindingStatus",
    "InterpretationResult",
    "KnowledgePlan",
    "KnowledgeTopic",
    "KnownCombination",
    "PipelineCallRecord",
    "PipelineMode",
    "PipelineRunTrace",
    "PipelineStage",
    "PipelineStateEvent",
    "QualityDecision",
    "ReportGenerationInput",
    "RetrievalEvidence",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievalStatus",
    "ValidationDecision",
    "utc_now",
]
