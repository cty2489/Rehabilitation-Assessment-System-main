"""Versioned contracts for the rehabilitation LLM benchmark preparation.

The field names for patient scores and biomarkers intentionally mirror the
existing production contracts: ``FMA_UE``, ``hand_tone``, ``hand_function``
and the computed marker rows.  This module does not calculate or overwrite any
clinical value.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AssessmentInputMode(str, Enum):
    DL_PREDICTION = "dl_prediction"
    MANUAL_CLINICAL_SCORES = "manual_clinical_scores"


class ClinicalScoreSource(str, Enum):
    DL_PREDICTION = "dl_prediction"
    CLINICIAN_PROVIDED = "clinician_provided"


class BenchmarkRunConfig(_StrictModel):
    """Independent benchmark switches; production defaults remain untouched."""

    assessment_input_mode: AssessmentInputMode = AssessmentInputMode.MANUAL_CLINICAL_SCORES
    clinical_score_source: Optional[ClinicalScoreSource] = None
    rag_enabled: bool = False
    rag_version: Optional[str] = Field(default=None, max_length=128)
    knowledge_graph_enabled: bool = False
    kg_version: Optional[str] = Field(default=None, max_length=128)
    prompt_version: str = Field(default="rehab_llm_benchmark_v1", min_length=1, max_length=128)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    max_new_tokens: Optional[int] = Field(default=2048, ge=1)

    @model_validator(mode="after")
    def validate_independent_switches(self) -> "BenchmarkRunConfig":
        if self.prompt_version != "rehab_llm_benchmark_v1":
            raise ValueError("Benchmark当前只允许使用rehab_llm_benchmark_v1")
        expected_source = (
            ClinicalScoreSource.CLINICIAN_PROVIDED
            if self.assessment_input_mode == AssessmentInputMode.MANUAL_CLINICAL_SCORES
            else ClinicalScoreSource.DL_PREDICTION
        )
        if self.clinical_score_source is None:
            self.clinical_score_source = expected_source
        elif self.clinical_score_source != expected_source:
            raise ValueError(
                "clinical_score_source必须与assessment_input_mode一致"
            )
        if self.rag_enabled and not self.rag_version:
            raise ValueError("rag_enabled=true时必须提供rag_version")
        if not self.rag_enabled and self.rag_version is not None:
            raise ValueError("rag_enabled=false时rag_version必须为空")
        if self.knowledge_graph_enabled and not self.kg_version:
            raise ValueError("knowledge_graph_enabled=true时必须提供kg_version")
        if not self.knowledge_graph_enabled and self.kg_version is not None:
            raise ValueError("knowledge_graph_enabled=false时kg_version必须为空")
        return self


BENCHMARK_PRESETS: Dict[str, Dict[str, Any]] = {
    "benchmark_stage1": {
        "assessment_input_mode": AssessmentInputMode.MANUAL_CLINICAL_SCORES,
        "clinical_score_source": ClinicalScoreSource.CLINICIAN_PROVIDED,
        "rag_enabled": False,
        "knowledge_graph_enabled": False,
    },
    "benchmark_stage2_baseline": {
        "assessment_input_mode": AssessmentInputMode.MANUAL_CLINICAL_SCORES,
        "clinical_score_source": ClinicalScoreSource.CLINICIAN_PROVIDED,
        "rag_enabled": False,
        "knowledge_graph_enabled": False,
    },
    "benchmark_stage2_rag": {
        "assessment_input_mode": AssessmentInputMode.MANUAL_CLINICAL_SCORES,
        "clinical_score_source": ClinicalScoreSource.CLINICIAN_PROVIDED,
        "rag_enabled": True,
        "rag_version": "rehab_knowledge_v1_candidate",
        "knowledge_graph_enabled": False,
    },
    "benchmark_stage2_kg": {
        "assessment_input_mode": AssessmentInputMode.MANUAL_CLINICAL_SCORES,
        "clinical_score_source": ClinicalScoreSource.CLINICIAN_PROVIDED,
        "rag_enabled": False,
        "knowledge_graph_enabled": True,
        "kg_version": "kg_current",
    },
    "benchmark_stage2_rag_kg": {
        "assessment_input_mode": AssessmentInputMode.MANUAL_CLINICAL_SCORES,
        "clinical_score_source": ClinicalScoreSource.CLINICIAN_PROVIDED,
        "rag_enabled": True,
        "rag_version": "rehab_knowledge_v1_candidate",
        "knowledge_graph_enabled": True,
        "kg_version": "kg_current",
    },
}


def benchmark_preset(name: str) -> BenchmarkRunConfig:
    """Return a validated copy of a named benchmark experiment preset."""
    try:
        values = dict(BENCHMARK_PRESETS[str(name).strip()])
    except KeyError as exc:
        raise ValueError(f"未知Benchmark preset：{name}") from exc
    return BenchmarkRunConfig(**values)


class BenchmarkPatientInfo(_StrictModel):
    patient_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    sex: Optional[str] = Field(default=None, max_length=32)
    age: Optional[int] = Field(default=None, ge=0, le=150)
    diagnosis: Optional[str] = Field(default=None, max_length=255)
    disease_days: Optional[int] = Field(default=None, ge=0)
    paralysis_side: Optional[str] = Field(default=None, max_length=32)


class BenchmarkClinicalScores(_StrictModel):
    """Benchmark-only clinical facts with explicit wrist/hand separation.

    ``FMA_UE``/``hand_tone``/``hand_function`` are accepted only as input
    aliases for older callers. They are not serialized into the benchmark
    prompt or output; the canonical experiment names are the four fields below.
    """

    fma_wrist: Optional[float] = Field(default=None, ge=0.0)
    fma_hand: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    hand_mas: Optional[Literal["0", "1", "1+", "2", "3", "4"]] = None
    brunnstrom_hand: Optional[int] = Field(default=None, ge=1, le=6)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_names(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        aliases = {
            "FMA_UE": "fma_hand",
            "hand_tone": "hand_mas",
            "hand_function": "brunnstrom_hand",
        }
        for old_name, new_name in aliases.items():
            if new_name not in data and old_name in data:
                data[new_name] = data[old_name]
            data.pop(old_name, None)
        return data

    @model_validator(mode="after")
    def require_core_scores(self) -> "BenchmarkClinicalScores":
        missing = [
            name
            for name, value in (
                ("fma_hand", self.fma_hand),
                ("hand_mas", self.hand_mas),
                ("brunnstrom_hand", self.brunnstrom_hand),
            )
            if value is None
        ]
        if missing:
            raise ValueError("缺少临床评分：" + "、".join(missing))
        return self

    # Read-only compatibility accessors for existing code/tests. The
    # benchmark layer serializes only the canonical names above.
    @property
    def FMA_UE(self) -> float:
        assert self.fma_hand is not None
        return self.fma_hand

    @property
    def hand_tone(self) -> str:
        assert self.hand_mas is not None
        return self.hand_mas

    @property
    def hand_function(self) -> int:
        assert self.brunnstrom_hand is not None
        return self.brunnstrom_hand


class BenchmarkBiomarker(_StrictModel):
    """One row copied from the existing signal-derived 26-marker result."""

    metric_key: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    value: Optional[float] = None
    value_text: Optional[str] = Field(default=None, max_length=255)
    unit: Optional[str] = Field(default=None, max_length=64)
    modality: Literal["eeg", "emg", "imu"]
    available: bool = True
    clinical_usable: bool = True
    n_valid: int = Field(default=0, ge=0)


class BenchmarkEvaluationInput(_StrictModel):
    """Fixed facts supplied identically to each future model."""

    schema_version: Literal["rehab.llm-benchmark-input.v1"] = (
        "rehab.llm-benchmark-input.v1"
    )
    evaluation_input_id: str = Field(
        default_factory=lambda: f"benchmark-input-{uuid4().hex}", min_length=1
    )
    config: BenchmarkRunConfig = Field(default_factory=BenchmarkRunConfig)
    patient: BenchmarkPatientInfo
    clinical_scores: BenchmarkClinicalScores
    clinical_score_source: ClinicalScoreSource
    biomarkers: List[BenchmarkBiomarker] = Field(min_length=1)
    quality_status: Literal["pass", "review"] = "pass"
    quality_metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkEvidenceCard(_StrictModel):
    uid: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=500)
    source_type: Literal["paper", "guideline", "textbook"]
    page_start: Optional[int] = Field(default=None, ge=1)
    page_end: Optional[int] = Field(default=None, ge=1)
    source_file: Optional[str] = Field(default=None, max_length=1024)
    source_pdf: Optional[str] = Field(default=None, max_length=1024)


class BenchmarkKnowledgeContext(_StrictModel):
    rag_version: str = Field(min_length=1, max_length=128)
    evidence_cards: List[BenchmarkEvidenceCard] = Field(default_factory=list)
    retrieved_text: List[str] = Field(default_factory=list)


class BenchmarkGraphContext(_StrictModel):
    kg_version: str = Field(min_length=1, max_length=128)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    relations: List[Dict[str, Any]] = Field(default_factory=list)


class RehabilitationAction(_StrictModel):
    action: str = Field(min_length=1, max_length=1000)
    goal: str = Field(min_length=1, max_length=1000)
    reason: str = Field(min_length=1, max_length=1500)
    precaution: str = Field(min_length=1, max_length=1500)


class BenchmarkLlmOutput(_StrictModel):
    """The only content a benchmark LLM is allowed to author."""

    biomarker_interpretation: str = Field(min_length=1, max_length=12000)
    integrated_assessment: str = Field(min_length=1, max_length=12000)
    rehabilitation_plan: List[RehabilitationAction] = Field(min_length=3, max_length=5)


class DoctorReviewScore(_StrictModel):
    clinical_correctness: int = Field(..., ge=1, le=5)
    completeness: int = Field(..., ge=1, le=5)
    clinical_usefulness: int = Field(..., ge=1, le=5)
    safety: int = Field(..., ge=1, le=5)
    hallucination_or_factual_error: bool
    serious_clinical_error: bool


class BenchmarkExperimentLog(_StrictModel):
    """One model generation record; raw output is never replaced by parsed data."""

    schema_version: Literal["rehab.llm-benchmark-log.v1"] = "rehab.llm-benchmark-log.v1"
    experiment_id: str = Field(default_factory=lambda: f"experiment-{uuid4().hex}")
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
    prompt_version: str = Field(min_length=1, max_length=128)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_new_tokens: Optional[int] = Field(default=None, ge=1)
    # Canonical benchmark field is explicitly expressed in milliseconds.
    generation_time_ms: Optional[float] = Field(default=None, ge=0.0)
    # Compatibility field for earlier preparation records; new writers set both.
    generation_time: Optional[float] = Field(default=None, ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_model_output: str = ""
    parsed_model_output: Optional[Dict[str, Any]] = None
    parse_success: bool = False
    biomarker_version: Optional[str] = None
    biomarker_metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class BlindReviewPacket(_StrictModel):
    """The doctor-facing projection; model and ablation metadata are excluded."""

    anonymous_report_id: str = Field(min_length=1, max_length=64)
    patient: BenchmarkPatientInfo
    report_markdown: str = Field(min_length=1)
    doctor_score: Optional[DoctorReviewScore] = None


class BenchmarkRegistryMapping(_StrictModel):
    """Restricted backend mapping retained separately from blind-review data."""

    anonymous_report_id: str = Field(min_length=1, max_length=64)
    patient_id: str = Field(min_length=1, max_length=64)
    llm_model_id: str = Field(min_length=1, max_length=255)
    assessment_input_mode: AssessmentInputMode
    clinical_score_source: ClinicalScoreSource
    rag_enabled: bool
    rag_version: Optional[str] = None
    knowledge_graph_enabled: bool
    kg_version: Optional[str] = None


__all__ = [
    "AssessmentInputMode",
    "ClinicalScoreSource",
    "BenchmarkBiomarker",
    "BenchmarkClinicalScores",
    "BenchmarkEvaluationInput",
    "BenchmarkEvidenceCard",
    "BenchmarkExperimentLog",
    "BenchmarkGraphContext",
    "BenchmarkKnowledgeContext",
    "BenchmarkLlmOutput",
    "BenchmarkPatientInfo",
    "BenchmarkRegistryMapping",
    "BenchmarkRunConfig",
    "BENCHMARK_PRESETS",
    "benchmark_preset",
    "BlindReviewPacket",
    "DoctorReviewScore",
    "RehabilitationAction",
]
