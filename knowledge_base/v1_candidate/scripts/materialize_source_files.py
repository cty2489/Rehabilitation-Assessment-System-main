#!/usr/bin/env python3
"""Materialize the immutable source-file layer for v1_candidate.

The default storage mode is a hardlink to the retained original file, with a
relative symlink fallback. This creates a self-describing source library while
avoiding a second full copy on the same filesystem. ``--copy`` makes a
portable physical copy when a release is being packaged for another host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[3]
RELEASE = ROOT / "knowledge_base" / "v1_candidate"
LIBRARY = RELEASE / "source_files"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def page_count(path: Path) -> int | None:
    if path.suffix.lower() != ".pdf":
        return None
    with fitz.open(path) as doc:
        return len(doc)


def project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = ROOT / value
    return path if path.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action="store_true", help="copy bytes instead of hardlink/symlink")
    args = parser.parse_args()
    sources = jsonl(RELEASE / "sources.jsonl")
    chunks = jsonl(RELEASE / "chunks.jsonl")
    if LIBRARY.exists():
        shutil.rmtree(LIBRARY)
    LIBRARY.mkdir(parents=True)

    original_to_file: dict[str, dict] = {}
    source_file_ids: dict[str, list[str]] = defaultdict(list)
    for source in sources:
        paths = []
        for key in ("source_pdf", "source_document", "original_source_pdf"):
            value = source.get(key)
            if value and value not in paths:
                paths.append(value)
        for original_rel in paths:
            original = project_path(original_rel)
            if original is None:
                continue
            digest = sha256(original)
            file_id = f"FILE-{digest[:16]}"
            if original_rel in original_to_file:
                source_file_ids[source["source_id"]].append(file_id)
                continue
            suffix = original.suffix.lower() or ".bin"
            library_name = f"{file_id}{suffix}"
            target = LIBRARY / library_name
            storage_mode = "copy" if args.copy else "hardlink"
            if args.copy:
                shutil.copy2(original, target)
            else:
                try:
                    os.link(original, target)
                except OSError:
                    storage_mode = "relative_symlink"
                    target.symlink_to(os.path.relpath(original, target.parent))
            record = {
                "schema_version": "rehab.knowledge.source-file.v1",
                "source_file_id": file_id,
                "source_file": f"knowledge_base/v1_candidate/source_files/{library_name}",
                "original_pdf": original_rel if original.suffix.lower() == ".pdf" else None,
                "original_document": original_rel if original.suffix.lower() != ".pdf" else None,
                "sha256": digest,
                "size_bytes": original.stat().st_size,
                "media_type": "application/pdf" if original.suffix.lower() == ".pdf" else "application/octet-stream",
                "page_count": page_count(original),
                "storage_mode": storage_mode,
                "immutable": True,
            }
            original_to_file[original_rel] = record
            source_file_ids[source["source_id"]].append(file_id)

    source_by_id = {s["source_id"]: s for s in sources}
    for source in sources:
        ids = list(dict.fromkeys(source_file_ids.get(source["source_id"], [])))
        primary_rel = source.get("source_pdf") or source.get("source_document")
        primary = original_to_file.get(primary_rel) if primary_rel else None
        source["source_file_id"] = primary["source_file_id"] if primary else None
        source["source_file"] = primary["source_file"] if primary else None
        source["original_pdf"] = primary["original_pdf"] if primary else None
        source["source_file_ids"] = ids
        source["source_file_library"] = "knowledge_base/v1_candidate/source_files"
        source["source_file_immutable"] = True

    for chunk in chunks:
        meta = chunk["metadata"]
        source = source_by_id[meta["source_id"]]
        # Preserve an explicit stable UID on every chunk, including legacy
        # guide records that only carried source_id before materialization.
        meta["uid"] = meta.get("uid") or source.get("uid") or source["source_id"]
        primary = original_to_file.get(meta.get("source_pdf") or meta.get("source_document"))
        if primary:
            page = meta.get("page_start")
            meta.update({
                "source_file_id": primary["source_file_id"],
                "source_file": primary["source_file"],
                "original_pdf": primary["original_pdf"] or primary["original_document"],
                "page_number": page,
                "original_pdf_page": page,
                "source_pdf_url": f"/api/rag/source-files/{primary['source_file_id']}/pdf",
                "source_page_url": f"/api/rag/source-files/{primary['source_file_id']}/pages/{page}" if page else f"/api/rag/source-files/{primary['source_file_id']}",
                "source_view_route": f"/api/rag/source-files/{primary['source_file_id']}/pages/{page}" if page else f"/api/rag/source-files/{primary['source_file_id']}",
            })
        if meta.get("page_image"):
            meta["ocr_scan_page_url"] = f"/api/rag/ocr-scan-pages/{meta['source_id']}/{meta.get('page_start')}"
            meta["source_page_mode"] = "pdf_page_plus_ocr_scan_page"
        else:
            meta["source_page_mode"] = "pdf_page"

    file_rows = sorted(original_to_file.values(), key=lambda row: row["source_file_id"])
    write_jsonl(RELEASE / "source_file_manifest.jsonl", file_rows)
    write_jsonl(RELEASE / "sources.jsonl", sources)
    write_jsonl(RELEASE / "chunks.jsonl", chunks)
    manifest_path = RELEASE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_file_library"] = {
        "directory": "source_files",
        "manifest": "source_file_manifest.jsonl",
        "files": len(file_rows),
        "immutable": True,
        "storage_mode": "copy" if args.copy else "hardlink_or_relative_symlink",
        "supports": ["paper_pdf", "guideline_pdf", "textbook_pdf", "ocr_textbook_scan_page"],
    }
    manifest["counts"]["source_files"] = len(file_rows)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    contract = {
        "schema_version": "rehab.knowledge.source-view.v2",
        "chain": "chunk_id -> uid -> source_file_id -> source_file -> page_number -> original_pdf",
        "routes": {
            "pdf": "/api/rag/source-files/{source_file_id}/pdf",
            "page": "/api/rag/source-files/{source_file_id}/pages/{page_number}",
            "ocr_scan_page": "/api/rag/ocr-scan-pages/{source_id}/{page_number}",
        },
        "page_number_base": 1,
        "click_behavior": "page route redirects/opens PDF with #page={page_number}; OCR textbook page also exposes the exact scanned page image",
        "supported_source_types": ["paper", "guideline", "textbook"],
        "immutable_file_check": "source_file_manifest.sha256 must match the staged bytes before serving",
    }
    (RELEASE / "source_view_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_files": len(file_rows), "chunks_updated": len(chunks), "library": str(LIBRARY), "storage_mode": manifest["source_file_library"]["storage_mode"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
