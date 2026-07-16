"""Governed indexing and dense semantic retrieval orchestration."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol, Sequence, Tuple

from .vector_store import VectorHit, VectorPoint


class Embedder(Protocol):
    dimension: int

    def encode(self, texts: Iterable[str]) -> List[List[float]]: ...


class VectorStore(Protocol):
    def replace_collection(
        self,
        collection: str,
        *,
        dimension: int,
        points: Iterable[VectorPoint],
    ) -> int: ...

    def search(self, collection: str, vector: List[float], top_k: int) -> List[VectorHit]: ...


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    score: float
    knowledge_id: str
    chunk_id: str
    title: str
    text: str
    metadata: Dict[str, Any]


def load_chunks(path: str | Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("schema_version") != "rehab.knowledge.chunk.v1":
            raise ValueError(f"line {line_number}: unsupported chunk schema")
        for field in ("chunk_id", "knowledge_id", "text", "metadata"):
            if field not in item:
                raise ValueError(f"line {line_number}: missing {field}")
        if item["chunk_id"] in seen:
            raise ValueError(f"duplicate chunk_id: {item['chunk_id']}")
        seen.add(item["chunk_id"])
        chunks.append(item)
    if not chunks:
        raise ValueError("chunk file is empty")
    return chunks


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rehab-rag:{chunk_id}"))


def validate_index_governance(
    chunks: List[Dict[str, Any]],
    *,
    allow_demo: bool = False,
) -> List[str]:
    unreviewed = [
        item["chunk_id"]
        for item in chunks
        if not bool(item.get("metadata", {}).get("clinical_ready"))
    ]
    if unreviewed and not allow_demo:
        raise ValueError(
            f"refusing to index {len(unreviewed)} non-clinical-ready chunks; "
            "pass --allow-demo only for an isolated retrieval experiment"
        )
    return unreviewed


def build_index(
    chunks: List[Dict[str, Any]],
    *,
    embedder: Embedder,
    store: VectorStore,
    collection: str,
    allow_demo: bool = False,
) -> Dict[str, Any]:
    unreviewed = validate_index_governance(chunks, allow_demo=allow_demo)
    vectors = embedder.encode(item["text"] for item in chunks)
    if len(vectors) != len(chunks):
        raise RuntimeError("embedding count does not match chunk count")
    points = []
    for item, vector in zip(chunks, vectors):
        if len(vector) != embedder.dimension:
            raise RuntimeError(f"unexpected vector dimension for {item['chunk_id']}")
        metadata = dict(item.get("metadata", {}))
        points.append(
            VectorPoint(
                point_id=_point_id(item["chunk_id"]),
                vector=vector,
                payload={
                    "chunk_id": item["chunk_id"],
                    "knowledge_id": item["knowledge_id"],
                    "entry_version": item.get("entry_version", ""),
                    "title": metadata.get("title", ""),
                    "text": item["text"],
                    "metadata": metadata,
                },
            )
        )
    indexed = store.replace_collection(
        collection,
        dimension=embedder.dimension,
        points=points,
    )
    return {
        "collection": collection,
        "indexed_chunks": indexed,
        "vector_dimension": embedder.dimension,
        "clinical_ready_chunks": len(chunks) - len(unreviewed),
        "demo_chunks": len(unreviewed),
    }


def retrieve(
    query: str,
    *,
    embedder: Embedder,
    store: VectorStore,
    collection: str,
    top_k: int,
) -> List[RetrievalResult]:
    return retrieve_many(
        [("query", query)],
        embedder=embedder,
        store=store,
        collection=collection,
        top_k=top_k,
    )[0][1]


def retrieve_many(
    queries: Sequence[Tuple[str, str]],
    *,
    embedder: Embedder,
    store: VectorStore,
    collection: str,
    top_k: int,
) -> List[Tuple[str, List[RetrievalResult]]]:
    """Embed a query batch once, then retrieve ranked hits for each query."""
    if not queries:
        raise ValueError("queries must not be empty")
    clean_queries = [(str(key).strip(), str(query).strip()) for key, query in queries]
    if any(not key or not query for key, query in clean_queries):
        raise ValueError("query keys and text must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    vectors = embedder.encode(query for _, query in clean_queries)
    if len(vectors) != len(clean_queries):
        raise RuntimeError("query embedding count does not match query count")

    batches: List[Tuple[str, List[RetrievalResult]]] = []
    for (key, _), vector in zip(clean_queries, vectors):
        hits = store.search(collection, vector, top_k)
        results = [
            RetrievalResult(
                rank=index,
                score=hit.score,
                knowledge_id=str(hit.payload.get("knowledge_id", "")),
                chunk_id=str(hit.payload.get("chunk_id", "")),
                title=str(hit.payload.get("title", "")),
                text=str(hit.payload.get("text", "")),
                metadata=dict(hit.payload.get("metadata", {})),
            )
            for index, hit in enumerate(hits, start=1)
        ]
        batches.append((key, results))
    return batches


def filter_governed_results(
    results: Iterable[RetrievalResult],
    *,
    include_demo: bool,
) -> List[RetrievalResult]:
    """Keep demo evidence only when an isolated experiment requested it."""
    values = list(results)
    if include_demo:
        return values
    return [item for item in values if bool(item.metadata.get("clinical_ready"))]


__all__ = [
    "RetrievalResult",
    "build_index",
    "filter_governed_results",
    "load_chunks",
    "retrieve",
    "retrieve_many",
    "validate_index_governance",
]
