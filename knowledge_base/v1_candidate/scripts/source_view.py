#!/usr/bin/env python3
"""Resolve a chunk click to its source PDF and 1-based page.

The function returns data for an API/frontend adapter.  It does not expose a
filesystem path outside the registered source manifest and never substitutes a
URL for the local source/page relationship.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RELEASE = ROOT / "knowledge_base" / "v1_candidate"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_detail(chunk_id: str) -> dict:
    chunk = next((c for c in _jsonl(RELEASE / "chunks.jsonl") if c["chunk_id"] == chunk_id), None)
    if chunk is None:
        raise KeyError(f"unknown chunk_id: {chunk_id}")
    meta = chunk["metadata"]
    source = next((s for s in _jsonl(RELEASE / "sources.jsonl") if s["source_id"] == meta["source_id"]), None)
    if source is None:
        raise KeyError(f"unknown source_id: {meta['source_id']}")
    page = meta.get("page_start")
    source_file_rel = meta.get("source_file") or source.get("source_file")
    original_pdf = meta.get("original_pdf") or meta.get("source_pdf") or source.get("original_pdf") or source.get("source_pdf")
    pdf_path = (ROOT / source_file_rel).resolve() if source_file_rel else None
    if pdf_path is not None and not pdf_path.is_file():
        raise FileNotFoundError(str(pdf_path))
    return {
        "chunk_id": chunk_id,
        "source_id": source["source_id"],
        "uid": meta.get("uid") or source.get("uid") or source["source_id"],
        "source_type": meta.get("source_type", source.get("source_type")),
        "evidence_level": meta.get("evidence_level", source.get("evidence_level")),
        "source_file_id": meta.get("source_file_id") or source.get("source_file_id"),
        "source_file": source_file_rel,
        "original_pdf": original_pdf,
        "source_pdf": source_file_rel,
        "page": page,
        "page_numbers": meta.get("page_numbers", []),
        "page_image": meta.get("page_image"),
        "bbox": meta.get("bbox"),
        "source_page_locator": meta.get("source_page_locator"),
        "source_view_route": meta.get("source_page_url") or meta.get("source_view_route"),
        "source_pdf_url": meta.get("source_pdf_url"),
        "source_page_url": meta.get("source_page_url"),
        "ocr_scan_page_url": meta.get("ocr_scan_page_url"),
        "book": meta.get("book", source.get("book")),
        "chapter": meta.get("chapter"),
        "section": meta.get("section"),
        "title": meta.get("title", source.get("title", source.get("book"))),
        "raw_ocr_text": meta.get("raw_ocr_text"),
        "corrected_text": meta.get("corrected_text"),
    }


if __name__ == "__main__":
    chunks = _jsonl(RELEASE / "chunks.jsonl")
    print(json.dumps(source_detail(chunks[0]["chunk_id"]), ensure_ascii=False, indent=2))
