"""Prompt assembly for Benchmark v2; v1 prompt loading is untouched."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..schemas import BenchmarkGraphContext, BenchmarkKnowledgeContext
from .schemas import BenchmarkEvaluationInput, PROMPT_VERSION

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / f"{PROMPT_VERSION}.txt"


def load_prompt() -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    if f"prompt_version = {PROMPT_VERSION}" not in text:
        raise RuntimeError(f"固定Prompt缺少版本标记：{PROMPT_VERSION}")
    return text


def build_messages(
    evaluation_input: BenchmarkEvaluationInput,
    *,
    knowledge_context: Optional[BenchmarkKnowledgeContext] = None,
    graph_context: Optional[BenchmarkGraphContext] = None,
) -> list[dict[str, str]]:
    config = evaluation_input.config
    if not config.rag_enabled and knowledge_context is not None:
        raise ValueError("RAG关闭时不能向v2 Prompt注入knowledge_context")
    if not config.knowledge_graph_enabled and graph_context is not None:
        raise ValueError("KG关闭时不能向v2 Prompt注入graph_context")
    if config.rag_enabled and knowledge_context is None:
        raise ValueError("RAG开启时必须提供knowledge_context")
    if config.knowledge_graph_enabled and graph_context is None:
        raise ValueError("KG开启时必须提供graph_context")

    fixed_input = evaluation_input.model_dump(mode="json")
    fixed_input["patient"].pop("name", None)
    payload: dict[str, Any] = {
        "patient_context": fixed_input.pop("patient"),
        "clinical_scores": fixed_input.pop("clinical_scores"),
        "biomarkers": fixed_input.pop("biomarkers"),
        "clinical_score_source": fixed_input.pop("clinical_score_source"),
        "fixed_assessment_input": fixed_input,
    }
    if config.rag_enabled:
        payload["rag_context"] = knowledge_context.model_dump(mode="json")
    if config.knowledge_graph_enabled:
        payload["knowledge_graph_context"] = graph_context.model_dump(mode="json")
    return [
        {"role": "system", "content": load_prompt()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)},
    ]


__all__ = ["PROMPT_PATH", "PROMPT_VERSION", "build_messages", "load_prompt"]
