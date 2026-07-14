#!/usr/bin/env python3
"""Embed prepared chunks and replace the configured Qdrant collection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.config import RagSettings
from rag.embedding import SentenceTransformerEmbedder
from rag.retrieval import build_index, load_chunks, validate_index_governance
from rag.vector_store import QdrantVectorStore


def main() -> int:
    settings = RagSettings.from_env(ROOT)
    parser = argparse.ArgumentParser(description="Build the dense RAG demo index")
    parser.add_argument(
        "--chunks",
        default=str(ROOT / "knowledge_base/runtime/rehab_knowledge_demo_v0_1/chunks.jsonl"),
    )
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--qdrant-path", default=str(settings.qdrant_path))
    parser.add_argument(
        "--qdrant-url",
        default=settings.qdrant_url if settings.backend == "server" else "",
    )
    parser.add_argument("--allow-demo", action="store_true")
    parser.add_argument(
        "--manifest-out",
        default=str(ROOT / "knowledge_base/runtime/rehab_knowledge_demo_v0_1/index_manifest.json"),
    )
    args = parser.parse_args()

    started = time.perf_counter()
    chunks = load_chunks(args.chunks)
    validate_index_governance(chunks, allow_demo=args.allow_demo)
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
        summary = build_index(
            chunks,
            embedder=embedder,
            store=store,
            collection=args.collection,
            allow_demo=args.allow_demo,
        )
    finally:
        store.close()
    summary.update(
        {
            "schema_version": "rehab.rag.index-manifest.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "embedding_model": args.model,
            "embedding_device": args.device,
            "backend": "server" if args.qdrant_url else "local",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    )
    manifest = Path(args.manifest_out)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
