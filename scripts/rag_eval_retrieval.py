#!/usr/bin/env python3
"""Evaluate Hit@K and MRR for the dense retrieval demo."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.config import RagSettings
from rag.embedding import SentenceTransformerEmbedder
from rag.retrieval import retrieve
from rag.vector_store import QdrantVectorStore


def _read_jsonl(path: str | Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    settings = RagSettings.from_env(ROOT)
    parser = argparse.ArgumentParser(description="Evaluate the dense RAG demo index")
    parser.add_argument(
        "--queries",
        default=str(ROOT / "knowledge_base/eval/demo_queries.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=max(3, settings.top_k))
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--qdrant-path", default=str(settings.qdrant_path))
    parser.add_argument(
        "--qdrant-url",
        default=settings.qdrant_url if settings.backend == "server" else "",
    )
    parser.add_argument("--min-hit-at-3", type=float, default=1.0)
    args = parser.parse_args()

    cases = _read_jsonl(args.queries)
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
    started = time.perf_counter()
    details = []
    reciprocal_ranks = []
    try:
        for case in cases:
            results = retrieve(
                case["query"],
                embedder=embedder,
                store=store,
                collection=args.collection,
                top_k=args.top_k,
            )
            ids = [result.knowledge_id for result in results]
            expected = case["expected_knowledge_id"]
            rank = ids.index(expected) + 1 if expected in ids else None
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            details.append(
                {
                    "query": case["query"],
                    "expected_knowledge_id": expected,
                    "rank": rank,
                    "top_ids": ids,
                }
            )
    finally:
        store.close()
    total = len(details)
    hit_at_1 = sum(item["rank"] == 1 for item in details) / total
    hit_at_3 = sum(bool(item["rank"] and item["rank"] <= 3) for item in details) / total
    report = {
        "schema_version": "rehab.rag.retrieval-eval.v1",
        "collection": args.collection,
        "cases": total,
        "hit_at_1": round(hit_at_1, 4),
        "hit_at_3": round(hit_at_3, 4),
        "mrr": round(sum(reciprocal_ranks) / total, 4),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "details": details,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if hit_at_3 >= args.min_hit_at_3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
