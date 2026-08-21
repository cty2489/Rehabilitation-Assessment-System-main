#!/usr/bin/env python3
"""Run the fixed candidate retrieval smoke set against the isolated index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RELEASE = ROOT / "knowledge_base" / "v1_candidate"


def load_chunks(release: Path) -> list[dict]:
    return [json.loads(line) for line in (release / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

QUERIES = [
    ("卒中后静息肌电与上肢运动障碍", {"PAPER-U106", "PAPER-U091"}),
    ("中位频率 MDF 与痉挛 MAS", {"PAPER-U123", "PAPER-U090"}),
    ("半球间相干与运动恢复", {"PAPER-U295"}),
    ("FMA 手部评分与标准量表边界", {"PAPER-U212", "TEXTBOOK-U178"}),
    ("改良 Ashworth 肌张力分级定义", {"TEXTBOOK-U178", "SRC-024"}),
    ("镜像疗法脑卒中手功能康复", {"SRC-020", "SRC-024"}),
    ("作业疗法上肢功能评定", {"TEXTBOOK-U181"}),
    ("康复评定 ROM 肌力平衡步态", {"TEXTBOOK-U178", "TEXTBOOK-U179"}),
]
PENDING_TEXTBOOK_UIDS = {"U179", "U180", "U181"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="rehab_knowledge_v1_candidate")
    parser.add_argument("--qdrant-path", default=None)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--release-dir", default=str(RELEASE))
    args = parser.parse_args()
    release = Path(args.release_dir).resolve()
    qdrant_path = args.qdrant_path or str(release / "qdrant")
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise SystemExit(f"Retrieval runner unavailable; build embeddings first in the existing RAG environment: {exc}") from exc
    encoder = SentenceTransformer(args.model, device="cpu")
    client = QdrantClient(path=qdrant_path)
    indexed_uids = {c["metadata"].get("uid") for c in load_chunks(release) if c["metadata"].get("source_type") == "textbook"}
    result_rows = []
    for query, expected in QUERIES:
        vector = encoder.encode(query, normalize_embeddings=True).tolist()
        if hasattr(client, "query_points"):
            hits = client.query_points(
                collection_name=args.collection,
                query=vector,
                limit=args.top_k,
                with_payload=True,
                with_vectors=False,
            ).points
        else:
            hits = client.search(collection_name=args.collection, query_vector=vector, limit=args.top_k)
        def metadata_for(hit):
            return dict(hit.payload.get("metadata") or {})

        hit_sources = [
            hit.payload.get("source_id")
            or metadata_for(hit).get("source_id")
            or hit.payload.get("knowledge_id")
            for hit in hits
        ]
        pending_expected = sorted(uid for uid in expected if uid.startswith("TEXTBOOK-") and uid.removeprefix("TEXTBOOK-") not in indexed_uids)
        evaluable = not pending_expected
        result_rows.append({"query": query, "expected_source_ids": sorted(expected), "pending_expected_source_ids": pending_expected, "status": "skipped_expected_source_pending" if pending_expected else "evaluated", "hit_source_ids": hit_sources, "top_k_hit": bool(expected.intersection(hit_sources)) if evaluable else None, "pending_textbook_leak": sorted(PENDING_TEXTBOOK_UIDS.intersection({s.removeprefix("TEXTBOOK-") for s in hit_sources if s and s.startswith("TEXTBOOK-")})), "hits": [{"score": hit.score, "chunk_id": hit.payload.get("chunk_id"), "source_id": metadata_for(hit).get("source_id") or hit.payload.get("source_id"), "uid": metadata_for(hit).get("uid") or hit.payload.get("uid"), "page_start": metadata_for(hit).get("page_start") or hit.payload.get("page_start")} for hit in hits]})
    evaluated = [r for r in result_rows if r["top_k_hit"] is not None]
    summary = {"schema_version": "rehab.knowledge.retrieval-test.v1", "collection": args.collection, "model": args.model, "top_k": args.top_k, "queries": result_rows, "evaluated_query_count": len(evaluated), "skipped_query_count": len(result_rows) - len(evaluated), "hit_rate": sum(r["top_k_hit"] for r in evaluated) / len(evaluated) if evaluated else None, "pending_textbook_leak_count": sum(bool(r["pending_textbook_leak"]) for r in result_rows)}
    (release / "retrieval_tests.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
