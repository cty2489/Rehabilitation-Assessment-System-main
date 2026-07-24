"""Configuration structure for ``planner_rag`` v0.1.

No medical or model-confidence threshold is supplied here. Future threshold
entries must carry explicit verification provenance before they can be loaded.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal

from pydantic import Field

from .contracts import ContractModel, PipelineMode


class VerifiedThreshold(ContractModel):
    key: str = Field(min_length=1, max_length=128)
    value: float
    unit: str = Field(min_length=1, max_length=64)
    evidence_ref: str = Field(min_length=1, max_length=255)
    verified_by: str = Field(min_length=1, max_length=255)
    verified_at: datetime


class QualityGateConfig(ContractModel):
    thresholds: List[VerifiedThreshold] = Field(default_factory=list)
    conflict_rule_refs: List[str] = Field(default_factory=list)


class LlmRoleConfig(ContractModel):
    model_id: str = Field(min_length=1, max_length=255)


class CoreKnowledgeConfig(ContractModel):
    required: Literal[True] = True
    bundle_version: str = Field(min_length=1, max_length=128)


class RetrieverConfig(ContractModel):
    batch_request_count: Literal[1] = 1


class PipelineConfig(ContractModel):
    schema_version: Literal["rehab.pipeline-config.v1"] = "rehab.pipeline-config.v1"
    config_version: str = Field(min_length=1, max_length=128)
    mode: Literal[PipelineMode.PLANNER_RAG] = PipelineMode.PLANNER_RAG
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    core_knowledge: CoreKnowledgeConfig
    planner: LlmRoleConfig
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    report_generator: LlmRoleConfig


__all__ = [
    "CoreKnowledgeConfig",
    "LlmRoleConfig",
    "PipelineConfig",
    "QualityGateConfig",
    "RetrieverConfig",
    "VerifiedThreshold",
]
