#!/usr/bin/env python3
"""Run one dense semantic query against the standalone RAG index."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.config import RagSettings
from rag.embedding import SentenceTransformerEmbedder
from rag.retrieval import retrieve
from rag.vector_store import QdrantVectorStore


def main() -> int:
    settings = RagSettings.from_env(ROOT)
    parser = argparse.ArgumentParser(description="Search the dense RAG demo index")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--qdrant-path", default=str(settings.qdrant_path))
    parser.add_argument(
        "--qdrant-url",
        default=settings.qdrant_url if settings.backend == "server" else "",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    embedder = SentenceTransformerEmbedder(
        args.model,
        device=args.device,
        max_sequence_length=settings.max_sequence_length,
        batch_size=settings.batch_size,
    )
    store = QdrantVectorStore(
        url=args.qdrant_url or None,
        path=None if args.qdrant_url else args.qdrant_path,
    )
    try:
        results = retrieve(
            args.query,
            embedder=embedder,
            store=store,
            collection=args.collection,
            top_k=args.top_k,
        )
    finally:
        store.close()
    payload = {
        "schema_version": "rehab.rag.search.v1",
        "query": args.query,
        "collection": args.collection,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
