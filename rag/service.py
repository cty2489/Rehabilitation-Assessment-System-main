"""Local-only HTTP service for governed rehabilitation knowledge retrieval."""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import RagSettings
from .embedding import SentenceTransformerEmbedder
from .retrieval import filter_governed_results, retrieve_many
from .vector_store import QdrantVectorStore


class QueryItem(BaseModel):
    key: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=4000)


class RetrieveRequest(BaseModel):
    queries: List[QueryItem] = Field(min_length=1, max_length=8)
    top_k: int = Field(default=3, ge=1, le=20)
    include_demo: bool = False


def create_app(settings: RagSettings | None = None) -> FastAPI:
    cfg = settings or RagSettings.from_env()
    runtime: Dict[str, Any] = {
        "embedder": None,
        "store": None,
        "lock": threading.Lock(),
    }

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if cfg.enabled:
            runtime["embedder"] = SentenceTransformerEmbedder(
                cfg.embedding_model,
                device=cfg.device,
                max_sequence_length=cfg.max_sequence_length,
                batch_size=cfg.batch_size,
            )
            runtime["store"] = QdrantVectorStore(
                url=cfg.qdrant_url if cfg.backend == "server" else None,
                path=cfg.qdrant_path if cfg.backend == "local" else None,
            )
        try:
            yield
        finally:
            store = runtime.get("store")
            if store is not None:
                store.close()

    app = FastAPI(
        title="Rehabilitation RAG Retrieval Service",
        version="0.3.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        loaded = runtime.get("embedder") is not None and runtime.get("store") is not None
        return {
            "status": "ok" if loaded else "disabled",
            "enabled": cfg.enabled,
            "loaded": loaded,
            "collection": cfg.collection,
            "backend": cfg.backend,
            "allow_demo": cfg.allow_demo,
        }

    @app.post("/v1/retrieve")
    def retrieve_endpoint(request: RetrieveRequest) -> Dict[str, Any]:
        embedder = runtime.get("embedder")
        store = runtime.get("store")
        if not cfg.enabled or embedder is None or store is None:
            raise HTTPException(status_code=503, detail="RAG service is disabled or not ready")
        if request.include_demo and not cfg.allow_demo:
            raise HTTPException(status_code=403, detail="demo evidence is disabled by governance policy")

        fetch_k = request.top_k if request.include_demo else min(50, request.top_k * 3)
        query_pairs = [(item.key, item.text) for item in request.queries]
        started = time.perf_counter()
        with runtime["lock"]:
            batches = retrieve_many(
                query_pairs,
                embedder=embedder,
                store=store,
                collection=cfg.collection,
                top_k=fetch_k,
            )

        query_text = {item.key: item.text for item in request.queries}
        results = []
        for key, hits in batches:
            governed = filter_governed_results(hits, include_demo=request.include_demo)
            results.append(
                {
                    "key": key,
                    "query": query_text[key],
                    "hits": [
                        {
                            "rank": index,
                            "score": round(hit.score, 6),
                            "knowledge_id": hit.knowledge_id,
                            "chunk_id": hit.chunk_id,
                            "title": hit.title,
                            "text": hit.text,
                            "metadata": hit.metadata,
                        }
                        for index, hit in enumerate(governed[: request.top_k], start=1)
                    ],
                }
            )
        return {
            "schema_version": "rehab.rag.retrieve.v1",
            "collection": cfg.collection,
            "demo_evidence_included": request.include_demo,
            "retrieval_ms": round((time.perf_counter() - started) * 1000, 1),
            "results": results,
        }

    return app


app = create_app()


__all__ = ["app", "create_app"]
