#!/usr/bin/env python3
"""Build the isolated bge-m3/Qdrant candidate index.

This uses the project's existing dense-retrieval contract: bge-m3, 1024
dimensions, Cosine, no independent reranker.  It only targets the new
``rehab_knowledge_v1_candidate`` collection and never opens the old collection.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE = ROOT / "knowledge_base" / "v1_candidate"


def load_chunks(release: Path) -> list[dict]:
    return [json.loads(line) for line in (release / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--qdrant-path", default=None)
    parser.add_argument("--collection", default="rehab_knowledge_v1_candidate")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE))
    args = parser.parse_args()
    release = Path(args.release_dir).resolve()
    qdrant_path = args.qdrant_path or str(release / "qdrant")
    chunks = load_chunks(release)
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise SystemExit(
            "Embedding runner unavailable: install the existing RAG environment's "
            "sentence-transformers and qdrant-client, then rerun this script. "
            f"Missing dependency: {exc}"
        ) from exc

    started = time.perf_counter()
    encoder = SentenceTransformer(args.model, device=args.device)
    client = QdrantClient(path=qdrant_path)
    vectors = encoder.encode([c["content"] for c in chunks], batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)
    client.recreate_collection(
        collection_name=args.collection,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(
            id=index,
            vector=vector.tolist(),
            payload={
                "chunk_id": c["chunk_id"],
                "knowledge_id": c["metadata"].get("source_id", ""),
                "entry_version": c["metadata"].get("entry_version", ""),
                "title": (
                    c["metadata"].get("title")
                    or c["metadata"].get("book")
                    or c["metadata"].get("source_pdf")
                    or ""
                ),
                "text": c["content"],
                "metadata": dict(c["metadata"]),
            },
        )
        for index, (c, vector) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=args.collection, points=points)
    summary = {
        "schema_version": "rehab.embedding.manifest.v1",
        "model": args.model,
        "dimensions": 1024,
        "distance": "Cosine",
        "collection": args.collection,
        "qdrant_path": qdrant_path,
        "count": len(points),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "reranker": "none",
    }
    (release / "embedding_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
