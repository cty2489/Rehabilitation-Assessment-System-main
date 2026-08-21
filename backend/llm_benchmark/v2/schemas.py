"""Version-isolated contracts for the two-section Benchmark v2."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas import (
    AssessmentInputMode,
    BenchmarkBiomarker,
    BenchmarkClinicalScores,
    BenchmarkGraphContext,
    BenchmarkKnowledgeContext,
    BenchmarkPatientInfo,
    ClinicalScoreSource,
    RehabilitationAction,
)

PROMPT_VERSION = "rehab_llm_benchmark_v2"
INPUT_SCHEMA_VERSION = "rehab.llm-benchmark-input.v2"
LOG_SCHEMA_VERSION = "rehab.llm-benchmark-log.v2"
OUTPUT_SCHEMA_VERSION = "rehab.llm-benchmark-output.v2"


class _StrictModelV2(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BenchmarkRunConfig(_StrictModelV2):
    assessment_input_mode: AssessmentInputMode = AssessmentInputMode.MANUAL_CLINICAL_SCORES
    clinical_score_source: Optional[ClinicalScoreSource] = None
    rag_enabled: bool = False
    rag_version: Optional[str] = Field(default=None, max_length=128)
    knowledge_graph_enabled: bool = False
    kg_version: Optional[str] = Field(default=None, max_length=128)
    prompt_version: str = Field(default=PROMPT_VERSION, min_length=1, max_length=128)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    max_new_tokens: Optional[int] = Field(default=2048, ge=1)

    @model_validator(mode="after")
    def validate_v2_contract(self) -> "BenchmarkRunConfig":
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError(f"Benchmark v2只允许使用{PROMPT_VERSION}")
        expected_source = (
            ClinicalScoreSource.CLINICIAN_PROVIDED
            if self.assessment_input_mode == AssessmentInputMode.MANUAL_CLINICAL_SCORES
            else ClinicalScoreSource.DL_PREDICTION
        )
        if self.clinical_score_source is None:
            self.clinical_score_source = expected_source
        elif self.clinical_score_source != expected_source:
            raise ValueError("clinical_score_source必须与assessment_input_mode一致")
        if self.rag_enabled and not self.rag_version:
            raise ValueError("rag_enabled=true时必须提供rag_version")
        if not self.rag_enabled and self.rag_version is not None:
            raise ValueError("rag_enabled=false时rag_version必须为空")
        if self.knowledge_graph_enabled and not self.kg_version:
            raise ValueError("knowledge_graph_enabled=true时必须提供kg_version")
        if not self.knowledge_graph_enabled and self.kg_version is not None:
            raise ValueError("knowledge_graph_enabled=false时kg_version必须为空")
        return self


BENCHMARK_PRESETS_V2: Dict[str, Dict[str, Any]] = {
    "benchmark_stage1_v2": {
        "assessment_input_mode": AssessmentInputMode.MANUAL_CLINICAL_SCORES,
        "clinical_score_source": ClinicalScoreSource.CLINICIAN_PROVIDED,
        "rag_enabled": False,
        "knowledge_graph_enabled": False,
        "prompt_version": PROMPT_VERSION,
    }
}


def benchmark_preset(name: str = "benchmark_stage1_v2") -> BenchmarkRunConfig:
    try:
        values = dict(BENCHMARK_PRESETS_V2[str(name).strip()])
    except KeyError as exc:
        raise ValueError(f"未知Benchmark v2 preset：{name}") from exc
    return BenchmarkRunConfig(**values)


class BenchmarkEvaluationInput(_StrictModelV2):
    schema_version: Literal[INPUT_SCHEMA_VERSION] = INPUT_SCHEMA_VERSION
    evaluation_input_id: str = Field(default_factory=lambda: f"benchmark-input-v2-{uuid4().hex}", min_length=1)
    config: BenchmarkRunConfig = Field(default_factory=BenchmarkRunConfig)
    patient: BenchmarkPatientInfo
    clinical_scores: BenchmarkClinicalScores
    clinical_score_source: ClinicalScoreSource
    biomarkers: List[BenchmarkBiomarker] = Field(min_length=1)
    quality_status: Literal["pass", "review"] = "pass"
    quality_metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkLlmOutput(_StrictModelV2):
    """The only two generated sections allowed by Benchmark v2."""

    integrated_assessment: str = Field(min_length=1, max_length=12000)
    rehabilitation_plan: List[RehabilitationAction]


class BenchmarkExperimentLog(_StrictModelV2):
    schema_version: Literal[LOG_SCHEMA_VERSION] = LOG_SCHEMA_VERSION
    experiment_id: str = Field(default_factory=lambda: f"experiment-v2-{uuid4().hex}")
    batch_id: Optional[str] = Field(default=None, max_length=128)
    patient_id: str = Field(min_length=1, max_length=64)
    anonymous_report_id: str = Field(min_length=1, max_length=64)
    assessment_input_mode: AssessmentInputMode
    clinical_score_source: ClinicalScoreSource
    fma_wrist: Optional[float] = Field(default=None, ge=0.0)
    fma_hand: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    hand_mas: Optional[str] = None
    brunnstrom_hand: Optional[int] = Field(default=None, ge=1, le=6)
    llm_model_id: str = Field(min_length=1, max_length=255)
    llm_model_name: str = Field(min_length=1, max_length=255)
    rag_enabled: bool
    rag_version: Optional[str] = None
    knowledge_graph_enabled: bool
    kg_version: Optional[str] = None
    prompt_version: Literal[PROMPT_VERSION] = PROMPT_VERSION
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_new_tokens: Optional[int] = Field(default=None, ge=1)
    generation_time_ms: Optional[float] = Field(default=None, ge=0.0)
    generation_time: Optional[float] = Field(default=None, ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_model_output: str = ""
    parsed_model_output: Optional[Dict[str, Any]] = None
    parse_success: bool = False
    biomarker_version: Optional[str] = None
    biomarker_metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


__all__ = [
    "AssessmentInputMode",
    "BENCHMARK_PRESETS_V2",
    "BenchmarkBiomarker",
    "BenchmarkClinicalScores",
    "BenchmarkEvaluationInput",
    "BenchmarkExperimentLog",
    "BenchmarkGraphContext",
    "BenchmarkKnowledgeContext",
    "BenchmarkLlmOutput",
    "BenchmarkPatientInfo",
    "BenchmarkRunConfig",
    "ClinicalScoreSource",
    "INPUT_SCHEMA_VERSION",
    "LOG_SCHEMA_VERSION",
    "OUTPUT_SCHEMA_VERSION",
    "PROMPT_VERSION",
    "RehabilitationAction",
    "benchmark_preset",
]
