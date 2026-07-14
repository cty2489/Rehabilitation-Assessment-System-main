"""Qdrant adapter supporting local persistent mode and a later server mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class VectorPoint:
    point_id: str
    vector: List[float]
    payload: Dict[str, Any]


@dataclass(frozen=True)
class VectorHit:
    score: float
    payload: Dict[str, Any]


class QdrantVectorStore:
    def __init__(
        self,
        *,
        path: str | Path | None = None,
        url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed; use the isolated requirements-rag.txt environment"
            ) from exc
        if bool(path) == bool(url):
            raise ValueError("provide exactly one of path or url")
        self._models = models
        if path:
            local_path = Path(path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(local_path))
        else:
            self._client = QdrantClient(url=str(url), timeout=timeout)

    def replace_collection(
        self,
        collection: str,
        *,
        dimension: int,
        points: Iterable[VectorPoint],
    ) -> int:
        if self._client.collection_exists(collection):
            self._client.delete_collection(collection)
        self._client.create_collection(
            collection_name=collection,
            vectors_config=self._models.VectorParams(
                size=dimension,
                distance=self._models.Distance.COSINE,
            ),
        )
        values = list(points)
        if values:
            self._client.upsert(
                collection_name=collection,
                points=[
                    self._models.PointStruct(
                        id=point.point_id,
                        vector=point.vector,
                        payload=point.payload,
                    )
                    for point in values
                ],
                wait=True,
            )
        return len(values)

    def search(self, collection: str, vector: List[float], top_k: int) -> List[VectorHit]:
        response = self._client.query_points(
            collection_name=collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            VectorHit(score=float(point.score), payload=dict(point.payload or {}))
            for point in response.points
        ]

    def close(self) -> None:
        self._client.close()


__all__ = ["QdrantVectorStore", "VectorHit", "VectorPoint"]
