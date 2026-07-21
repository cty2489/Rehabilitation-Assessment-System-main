"""Batch retrieval adapter for the isolated ``planner_rag`` v0.1 pipeline."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

import rag_client

from .contracts import (
    KnowledgePlan,
    RetrievalEvidence,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStatus,
)


RagTransport = Callable[[str, Dict[str, Any], float], Dict[str, Any]]


def _source_ids(metadata: Dict[str, Any]) -> list[str]:
    raw = metadata.get("source_ids")
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    for item in raw:
        value = str(item).strip()
        if value and value not in values:
            values.append(value)
    return values


def _evidence_from_hit(
    hit: Any,
    *,
    query_id: str,
) -> Optional[RetrievalEvidence]:
    if not isinstance(hit, dict):
        return None
    chunk_id = str(hit.get("chunk_id") or "").strip()
    knowledge_id = str(hit.get("knowledge_id") or "").strip()
    text = str(hit.get("text") or "").strip()
    try:
        rank = int(hit.get("rank"))
        score = float(hit.get("score"))
    except (TypeError, ValueError):
        return None
    if (
        not chunk_id
        or not knowledge_id
        or not text
        or rank < 1
        or not math.isfinite(score)
    ):
        return None

    raw_metadata = hit.get("metadata") or {}
    if not isinstance(raw_metadata, dict):
        return None
    metadata = dict(raw_metadata)
    metadata["knowledge_id"] = knowledge_id
    metadata["title"] = str(hit.get("title") or "").strip()
    evidence_id = f"evidence-{uuid5(NAMESPACE_URL, f'{query_id}|{chunk_id}').hex}"
    return RetrievalEvidence(
        evidence_id=evidence_id,
        query_id=query_id,
        chunk_id=chunk_id,
        text=text,
        rank=rank,
        raw_score=score,
        source_ids=_source_ids(metadata),
        metadata=metadata,
    )


class Retriever:
    """Call the existing RAG ``/v1/retrieve`` endpoint exactly once per plan."""

    def __init__(
        self,
        *,
        settings: Optional[rag_client.RagClientSettings] = None,
        transport: Optional[RagTransport] = None,
    ) -> None:
        self._settings = settings or rag_client.RagClientSettings.from_env()
        self._transport = transport or rag_client._post_json

    def retrieve(
        self,
        plan: KnowledgePlan,
        *,
        attempt_id: Optional[str] = None,
    ) -> RetrievalResult:
        if not isinstance(plan, KnowledgePlan):
            raise TypeError("plan必须是KnowledgePlan")
        resolved_attempt_id = (attempt_id or f"retrieval-attempt-{uuid4().hex}").strip()
        if not resolved_attempt_id:
            raise ValueError("attempt_id不能为空")

        wire_to_query = {
            f"q{index}": query
            for index, query in enumerate(plan.queries, start=1)
        }
        payload = {
            "queries": [
                {"key": wire_key, "text": query.text}
                for wire_key, query in wire_to_query.items()
            ],
            "top_k": self._settings.top_k_per_query,
            "include_demo": False,
        }

        try:
            response = self._transport(
                f"{self._settings.service_url}/v1/retrieve",
                payload,
                self._settings.timeout_seconds,
            )
            evidence, collection = self._parse_response(response, wire_to_query)
        except Exception:  # noqa: BLE001 - adapter converts RAG failures to contract state
            return RetrievalResult(
                attempt_id=resolved_attempt_id,
                status=RetrievalStatus.UNAVAILABLE,
                queries=plan.queries,
                evidence=[],
                covered_topic_ids=[],
                uncovered_topic_ids=[topic.topic_id for topic in plan.topics],
            )

        covered_topic_ids = self._covered_topics(plan, evidence)
        uncovered_topic_ids = [
            topic.topic_id
            for topic in plan.topics
            if topic.topic_id not in covered_topic_ids
        ]
        if not evidence:
            status = RetrievalStatus.INSUFFICIENT
        elif uncovered_topic_ids:
            status = RetrievalStatus.PARTIAL
        else:
            status = RetrievalStatus.COMPLETE
        return RetrievalResult(
            attempt_id=resolved_attempt_id,
            status=status,
            queries=plan.queries,
            collection=collection or None,
            evidence=evidence,
            covered_topic_ids=covered_topic_ids,
            uncovered_topic_ids=uncovered_topic_ids,
        )

    @staticmethod
    def _parse_response(
        response: Dict[str, Any],
        wire_to_query: Dict[str, RetrievalQuery],
    ) -> tuple[list[RetrievalEvidence], str]:
        if not isinstance(response, dict):
            raise RuntimeError("RAG服务返回值不是JSON对象")
        if response.get("schema_version") != "rehab.rag.retrieve.v1":
            raise RuntimeError("RAG服务返回了不支持的schema_version")
        results = response.get("results")
        if not isinstance(results, list):
            raise RuntimeError("RAG服务results不是列表")

        evidence: list[RetrievalEvidence] = []
        seen_result_keys: set[str] = set()
        seen_query_chunks: set[tuple[str, str]] = set()
        for result in results:
            if not isinstance(result, dict):
                raise RuntimeError("RAG服务result不是对象")
            wire_key = str(result.get("key") or "").strip()
            if wire_key not in wire_to_query or wire_key in seen_result_keys:
                raise RuntimeError("RAG服务返回了未知或重复的query key")
            seen_result_keys.add(wire_key)
            hits = result.get("hits")
            if not isinstance(hits, list):
                raise RuntimeError("RAG服务hits不是列表")
            query_id = wire_to_query[wire_key].query_id
            for hit in hits:
                item = _evidence_from_hit(hit, query_id=query_id)
                if item is None:
                    continue
                identity = (item.query_id, item.chunk_id)
                if identity in seen_query_chunks:
                    continue
                seen_query_chunks.add(identity)
                evidence.append(item)

        collection = str(response.get("collection") or "").strip()
        return evidence, collection

    @staticmethod
    def _covered_topics(
        plan: KnowledgePlan,
        evidence: list[RetrievalEvidence],
    ) -> list[str]:
        evidence_query_ids = {item.query_id for item in evidence}
        queries_by_topic: Dict[str, set[str]] = {}
        for query in plan.queries:
            queries_by_topic.setdefault(query.topic_id, set()).add(query.query_id)
        return [
            topic.topic_id
            for topic in plan.topics
            if queries_by_topic.get(topic.topic_id, set()) & evidence_query_ids
        ]


__all__ = ["RagTransport", "Retriever"]
