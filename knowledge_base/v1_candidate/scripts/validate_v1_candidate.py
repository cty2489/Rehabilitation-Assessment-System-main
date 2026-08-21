#!/usr/bin/env python3
"""Validate source/page governance for the candidate release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[3]
RELEASE = ROOT / "knowledge_base" / "v1_candidate"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))
    sources = {s["source_id"]: s for s in jsonl(RELEASE / "sources.jsonl")}
    chunks = jsonl(RELEASE / "chunks.jsonl")
    source_files = {s["source_file_id"]: s for s in jsonl(RELEASE / "source_file_manifest.jsonl")} if (RELEASE / "source_file_manifest.jsonl").exists() else {}
    errors: list[str] = []
    seen: set[str] = set()
    page_checks = 0
    file_hash_checks = 0
    pdf_page_counts: dict[str, int] = {}
    for file_id, record in source_files.items():
        staged = ROOT / record["source_file"]
        if not staged.is_file():
            errors.append(f"source file missing: {file_id}")
            continue
        if digest(staged) != record["sha256"]:
            errors.append(f"source file hash mismatch: {file_id}")
        file_hash_checks += 1
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid in seen:
            errors.append(f"duplicate chunk_id: {cid}")
        seen.add(cid)
        meta = chunk["metadata"]
        source_id = meta.get("source_id")
        if source_id not in sources:
            errors.append(f"chunk {cid} has unknown source {source_id}")
            continue
        if meta.get("evidence_level") not in {"research", "textbook_reference"}:
            errors.append(f"chunk {cid} has invalid evidence_level")
        for key in ("uid", "source_file_id", "source_file", "page_start", "page_end", "source_type", "evidence_level", "knowledge_role", "original_pdf", "page_number", "source_page_url"):
            if key not in meta:
                errors.append(f"chunk {cid} missing source-file field {key}")
        if not meta.get("uid"):
            errors.append(f"chunk {cid} has empty uid")
        if not meta.get("source_file"):
            errors.append(f"chunk {cid} has empty source_file")
        source_file_id = meta.get("source_file_id")
        if source_file_id not in source_files:
            errors.append(f"chunk {cid} has unknown source_file_id {source_file_id}")
        elif not (ROOT / source_files[source_file_id]["source_file"]).exists():
            errors.append(f"chunk {cid} staged source file missing")
        if meta.get("source_type") == "textbook":
            for key in ("uid", "book", "chapter", "section", "page_start", "page_end", "topic", "source_pdf"):
                if key not in meta:
                    errors.append(f"textbook chunk {cid} missing {key}")
        source_path = meta.get("source_file")
        if source_path:
            pdf = ROOT / source_path
            if not pdf.is_file():
                errors.append(f"chunk {cid} source_pdf missing: {source_path}")
            elif meta.get("page_start"):
                key = str(pdf)
                if key not in pdf_page_counts:
                    with fitz.open(pdf) as doc:
                        pdf_page_counts[key] = len(doc)
                if not 1 <= int(meta["page_start"]) <= pdf_page_counts[key]:
                    errors.append(f"chunk {cid} page outside PDF: {meta['page_start']}")
                page_checks += 1
    snapshot_paths = [ROOT / ".codex_tmp/rag_v03/runtime_v03/manifest.json", ROOT / "knowledge_base/rag_runtime_v0.2/manifest.json"]
    snapshot_fingerprints = {str(p.relative_to(ROOT)): digest(p) for p in snapshot_paths if p.exists()}
    result = {
        "status": "pass" if not errors else "fail",
        "release": manifest.get("release"),
        "sources": len(sources),
        "chunks": len(chunks),
        "unique_chunk_ids": len(seen),
        "page_checks": page_checks,
        "file_hash_checks": file_hash_checks,
        "source_files": len(source_files),
        "source_file_library_check": "pass" if source_files and not any("source-file field" in e or "staged source file" in e for e in errors) else "fail",
        "errors": errors,
        "old_snapshot_manifest_sha256": snapshot_fingerprints,
        "old_snapshot_mutation_check": "read_only_fingerprint_recorded",
    }
    (RELEASE / "validation_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
