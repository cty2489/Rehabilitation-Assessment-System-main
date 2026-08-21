#!/usr/bin/env python3
"""Build the additive ``rehab_knowledge_v1`` candidate release.

This builder deliberately lives beside the existing runtime snapshots.  It does
not mutate v0.2/v0.3 data, reclassify papers, change evidence grades, or touch
the clinical pipeline.  The output is a portable JSONL release whose chunk
metadata always points back to a source file and, where available, a PDF page.

Textbook OCR is resumable and may be incomplete while the OCR workers run.  A
source is still registered in that case, with failed/missing pages recorded in
the manifest; rerunning this builder after OCR completion fills the chunks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import fitz


BASE = Path(__file__).resolve().parents[3]
OUT = BASE / "knowledge_base" / "v1_candidate"
LITERATURE_TABLE = BASE / "文献二次筛选" / "06_文献二次筛选总表.tsv"
CORE_DIRECT_TABLE = BASE / "文献二次筛选" / "20_core_direct_candidate清单.tsv"
CORE_BACKGROUND_TABLE = BASE / "文献二次筛选" / "21_core_background_candidate清单.tsv"
GUIDE_PACKAGE = BASE / "knowledge_base" / "指南知识库增量_v0.2.json"
OCR_ROOT = BASE / "output" / "ocr_validation_20260807" / "full_B"
_NATIVE_PAGE_CACHE: dict[str, dict[int, str]] = {}

TEXTBOOKS = {
    "U177": {
        "book": "康复功能评定学-第三版（扫描副本，复用U178可读版本）",
        "original_pdf": "模型文献/教材/康复功能评定学-第三版.pdf",
        "source_pdf": "模型文献/教材/康复功能评定学.pdf",
        "source_uid": "U178",
        "mode": "reuse_text_layer",
        "original_page_count": 656,
    },
    "U178": {
        "book": "康复功能评定学（第3版）",
        "source_pdf": "模型文献/教材/康复功能评定学.pdf",
        "mode": "native_text_layer",
    },
    "U179": {
        "book": "康复评定学",
        "source_pdf": "模型文献/教材/康复评定学.pdf",
        "mode": "ocr_b",
    },
    "U180": {
        "book": "康复疗法评定学（第2版）",
        "source_pdf": "模型文献/教材/康复疗法评定学+第2版_恽晓平主编2014.pdf",
        "mode": "ocr_b",
    },
    "U181": {
        "book": "作业疗法学",
        "source_pdf": "模型文献/教材/作业疗法学.pdf",
        "mode": "ocr_b",
    },
}
TEXTBOOK_ALIAS_TO_SOURCE = {"U177": "U178", "U178": "U178", "U179": "U179", "U180": "U180", "U181": "U181"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def resolve_literature_pdf(relative_path: str) -> Path:
    """Resolve the retained post-dedup path without changing the catalog."""
    rel = Path(relative_path)
    candidates = [BASE / "模型文献" / rel, BASE / "模型文献" / "专题原始文献" / rel, BASE / "模型文献" / "教材" / rel]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    name_hits = list((BASE / "模型文献").rglob(rel.name))
    if len(name_hits) == 1:
        return name_hits[0]
    raise FileNotFoundError(f"cannot resolve literature path: {relative_path}")


def relative_project_path(path: Path) -> str:
    return path.relative_to(BASE).as_posix()


def load_core_subcategories() -> dict[str, str]:
    result: dict[str, str] = {}
    for path, label in ((CORE_DIRECT_TABLE, "core_direct_candidate"), (CORE_BACKGROUND_TABLE, "core_background_candidate")):
        for row in read_tsv(path):
            result[row["uid"]] = label
    return result


def selected_literature_rows() -> list[dict[str, str]]:
    subcategories = load_core_subcategories()
    rows = []
    for row in read_tsv(LITERATURE_TABLE):
        category = row.get("分类", "")
        if category not in {"core_rag_candidate", "training_recommendation_candidate", "method_or_reference_only"}:
            continue
        row = dict(row)
        role = subcategories.get(row["uid"], category)
        row["ingest_role"] = "method_reference_only" if role == "method_or_reference_only" else role
        rows.append(row)
    return rows


def knowledge_role(source_type: str, classification: str | None = None) -> str:
    """Return the stable retrieval role without changing source classification."""
    role_map = {
        "core_direct_candidate": "core_direct",
        "core_background_candidate": "core_background",
        "training_recommendation_candidate": "training",
        "method_reference_only": "method_reference",
    }
    if source_type == "textbook":
        return "textbook_reference"
    if source_type == "guideline":
        return "guideline"
    return role_map.get(classification or "", "research_reference")


def page_count(path: Path) -> int | None:
    try:
        with fitz.open(path) as doc:
            return len(doc)
    except Exception:
        return None


def text_layer_pages(path: Path) -> dict[int, str]:
    with fitz.open(path) as doc:
        return {i + 1: page.get_text("text") for i, page in enumerate(doc)}


def normalize_native_text(text: str) -> str:
    # Layout-only normalization. No spelling or semantic correction is applied.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def split_text(text: str, max_chars: int = 1400) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks: list[str] = []
    buf = ""
    for paragraph in paragraphs:
        candidate = f"{buf}\n{paragraph}" if buf else paragraph
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf.strip())
            buf = ""
        while len(paragraph) > max_chars:
            cut = max(paragraph.rfind("。", 0, max_chars), paragraph.rfind(" ", 0, max_chars))
            cut = cut if cut >= max_chars // 2 else max_chars
            chunks.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        buf = paragraph
    if buf:
        chunks.append(buf.strip())
    return [c for c in chunks if len(c) >= 40]


def heading_update(chapter: str, section: str, lines: list[str]) -> tuple[str, str]:
    for line in lines:
        clean = re.sub(r"\s+", " ", line).strip()
        if not clean or len(clean) > 90:
            continue
        if re.search(r"第\s*[一二三四五六七八九十百0-9]+\s*章", clean):
            chapter, section = clean, ""
        elif re.search(r"第\s*[一二三四五六七八九十百0-9]+\s*节", clean):
            section = clean
        elif re.match(r"^(?:[一二三四五六七八九十百]+|[0-9]+)[、.．]", clean) and not clean.endswith(("。", "；", ";")):
            section = clean
    return chapter, section


def textbook_page_text(uid: str, page: int) -> tuple[str, dict]:
    spec = TEXTBOOKS[uid]
    if uid == "U177":
        uid = "U178"
        spec = TEXTBOOKS[uid]
    if spec["mode"] == "ocr_b":
        page_dir = OCR_ROOT / uid / f"page_{page:04d}"
        raw_path = page_dir / "raw_ocr_text.txt"
        corrected_path = page_dir / "corrected_text.txt"
        if not corrected_path.exists():
            return "", {"status": "missing", "raw_path": str(raw_path), "corrected_path": str(corrected_path)}
        return corrected_path.read_text(encoding="utf-8"), {
            "status": "success",
            "raw_path": relative_project_path(raw_path),
            "corrected_path": relative_project_path(corrected_path),
            "page_image": relative_project_path(page_dir / "page_image.png"),
            "confidence_path": relative_project_path(page_dir / "ocr_confidence.json"),
            "bbox_path": relative_project_path(page_dir / "bbox.json"),
            "text_derivation": "corrected_text_conservative_normalization",
        }
    source = BASE / spec["source_pdf"]
    cache_key = str(source)
    if cache_key not in _NATIVE_PAGE_CACHE:
        _NATIVE_PAGE_CACHE[cache_key] = text_layer_pages(source)
    pages = _NATIVE_PAGE_CACHE[cache_key]
    text = pages.get(page, "")
    return normalize_native_text(text), {
        "status": "success" if text.strip() else "missing",
        "raw_path": None,
        "corrected_path": None,
        "page_image": None,
        "confidence_path": None,
        "bbox_path": None,
        "text_derivation": "native_pdf_text_layer",
    }


def build_source_record(uid: str, *, ocr_excluded: bool = False) -> dict:
    spec = TEXTBOOKS[uid]
    source_pdf = BASE / spec["source_pdf"]
    record = {
        "schema_version": "rehab.knowledge.source.v2",
        "source_id": f"TEXTBOOK-{uid}",
        "uid": uid,
        "source_type": "textbook",
        "evidence_level": "textbook_reference",
        "knowledge_role": knowledge_role("textbook"),
        "book": spec["book"],
        "source_pdf": spec["source_pdf"],
        "source_pdf_sha256": sha256_file(source_pdf),
        "page_count": page_count(source_pdf),
        "ocr_mode": spec["mode"],
        "retrieval_role": "reference",
        "expert_review_status": "pending",
        "clinical_ready": False,
        "rag_indexed": uid == "U178" or (spec["mode"] == "ocr_b" and not ocr_excluded),
        "ocr_pending": spec["mode"] == "ocr_b" and ocr_excluded,
    }
    if uid == "U177":
        record.update({
            "page_count": spec["original_page_count"],
            "reused_page_count": TEXTBOOKS["U178"].get("page_count", 658),
            "source_pdf": TEXTBOOKS["U178"]["source_pdf"],
            "original_source_pdf": spec["original_pdf"],
            "original_page_count": spec["original_page_count"],
            "reused_from_uid": "U178",
            "note": "扫描副本不重新OCR；内容和页码复用U178可读文字层，避免重复入库。",
        })
    if uid == "U178":
        record["source_aliases"] = ["U177"]
    return record


def build_textbook_chunks(uid: str, source_record: dict) -> tuple[list[dict], dict]:
    count = source_record.get("page_count") or 0
    chunks: list[dict] = []
    chapter, section = "", ""
    success, failed = 0, []
    ordinal = 0
    for page in range(1, count + 1):
        text, refs = textbook_page_text(uid, page)
        if not text.strip():
            failed.append(page)
            continue
        success += 1
        lines = text.splitlines()
        chapter, section = heading_update(chapter, section, lines)
        topic = section or chapter or "教材正文"
        for piece in split_text(text):
            ordinal += 1
            chunk_id = f"TEXTBOOK-{uid}-p{page:04d}-c{ordinal:04d}"
            content_hash = hashlib.sha256(piece.encode("utf-8")).hexdigest()
            chunks.append({
                "schema_version": "rehab.knowledge.chunk.v2",
                "chunk_id": chunk_id,
                "content": piece,
                "metadata": {
                    "source_id": source_record["source_id"],
                    "uid": uid,
                    "source_type": "textbook",
                    "evidence_level": "textbook_reference",
                    "knowledge_role": "textbook_reference",
                    "book": source_record["book"],
                    "chapter": chapter,
                    "section": section,
                    "page_start": page,
                    "page_end": page,
                    "page_numbers": [page],
                    "topic": topic,
                    "source_pdf": source_record["source_pdf"],
                    "source_file": source_record["source_pdf"],
                    "source_page_locator": f"{source_record['source_pdf']}#page={page}",
                    "source_view_route": f"/api/rag/sources/{source_record['source_id']}/pages/{page}",
                    "raw_ocr_text": refs.get("raw_path"),
                    "corrected_text": refs.get("corrected_path"),
                    "page_image": refs.get("page_image"),
                    "ocr_confidence": refs.get("confidence_path"),
                    "bbox": refs.get("bbox_path"),
                    "text_derivation": refs.get("text_derivation"),
                    "content_hash": content_hash,
                    "expert_review_status": "pending",
                    "clinical_ready": False,
                },
            })
    return chunks, {"pages": count, "success_pages": success, "failed_pages": failed, "chunk_count": len(chunks)}


def build_paper_source(row: dict[str, str], *, ocr_excluded: bool = False) -> tuple[dict, Path]:
    pdf = resolve_literature_pdf(row["path"])
    source_kind = "guideline" if ("指南" in row.get("专题", "") or "共识" in row.get("专题", "") or "指南" in row.get("title", "")) else "paper"
    source = {
        "schema_version": "rehab.knowledge.source.v2",
        "source_id": f"PAPER-{row['uid']}",
        "uid": row["uid"],
        "source_type": source_kind,
        "evidence_level": "research",
        "knowledge_role": knowledge_role(source_kind, row["ingest_role"]),
        "classification": row["ingest_role"],
        "original_classification": row["分类"],
        "evidence_quality": row["evidence_quality"],
        "system_applicability": row["system_applicability"],
        "title": row["title"],
        "year": row.get("year", ""),
        "doi": row.get("doi", ""),
        "topic": row.get("专题", ""),
        "source_pdf": relative_project_path(pdf),
        "source_pdf_sha256": sha256_file(pdf),
        "page_count": page_count(pdf),
        "retrieval_role": "core" if row["ingest_role"].startswith("core_") else row["ingest_role"],
        "method_reference_policy": "low_priority_reference" if row["ingest_role"] == "method_reference_only" else None,
        "expert_review_status": "pending",
        "clinical_ready": False,
    }
    if row["uid"] in TEXTBOOK_ALIAS_TO_SOURCE:
        textbook_uid = TEXTBOOK_ALIAS_TO_SOURCE[row["uid"]]
        source.update({
            "rag_indexed": False,
            "replaced_by_source_id": f"TEXTBOOK-{textbook_uid}",
            "content_policy": "registered_for_classification_only; textbook_source_is_authoritative",
            "ocr_pending": ocr_excluded and textbook_uid in {"U179", "U180", "U181"},
        })
    return source, pdf


def build_paper_chunks(source: dict, pdf: Path) -> list[dict]:
    chunks: list[dict] = []
    with fitz.open(pdf) as doc:
        ordinal = 0
        for page_no, page in enumerate(doc, 1):
            text = normalize_native_text(page.get_text("text"))
            for piece in split_text(text):
                ordinal += 1
                chunk_id = f"{source['source_id']}-p{page_no:04d}-c{ordinal:04d}"
                content_hash = hashlib.sha256(piece.encode("utf-8")).hexdigest()
                chunks.append({
                    "schema_version": "rehab.knowledge.chunk.v2",
                    "chunk_id": chunk_id,
                    "content": piece,
                    "metadata": {
                        "source_id": source["source_id"],
                        "uid": source["uid"],
                        "source_type": source["source_type"],
                        "evidence_level": "research",
                        "knowledge_role": source["knowledge_role"],
                        "classification": source["classification"],
                        "evidence_quality": source["evidence_quality"],
                        "system_applicability": source["system_applicability"],
                        "title": source["title"],
                        "topic": source["topic"],
                        "page_start": page_no,
                        "page_end": page_no,
                        "page_numbers": [page_no],
                        "source_pdf": source["source_pdf"],
                        "source_file": source["source_pdf"],
                        "source_page_locator": f"{source['source_pdf']}#page={page_no}",
                        "source_view_route": f"/api/rag/sources/{source['source_id']}/pages/{page_no}",
                        "raw_ocr_text": None,
                        "corrected_text": None,
                        "page_image": None,
                        "bbox": None,
                        "text_derivation": "native_pdf_text_layer",
                        "content_hash": content_hash,
                        "expert_review_status": "pending",
                        "clinical_ready": False,
                    },
                })
    return chunks


def guide_records() -> tuple[list[dict], list[dict], list[dict]]:
    package = json.loads(GUIDE_PACKAGE.read_text(encoding="utf-8"))
    sources: list[dict] = []
    entries: list[dict] = []
    chunks: list[dict] = []
    for old in package["sources"]:
        source_path = BASE / "knowledge_base" / "guidelines" / old["file"]
        record = dict(old)
        record.update({
            "schema_version": "rehab.knowledge.source.v2",
            "uid": old.get("uid") or old["source_id"],
            "source_type": "guideline" if "指南" in old.get("source_type", "") or "共识" in old.get("source_type", "") or "Guideline" in old["title"] else "research",
            "evidence_level": "research",
            "source_pdf": f"knowledge_base/guidelines/{old['file']}" if old["file"].lower().endswith(".pdf") else None,
            "source_document": f"knowledge_base/guidelines/{old['file']}" if not old["file"].lower().endswith(".pdf") else None,
            "source_sha256": sha256_file(source_path) if source_path.exists() else None,
            "page_count": page_count(source_path) if source_path.suffix.lower() == ".pdf" and source_path.exists() else None,
            "retrieval_role": "guideline_or_research",
            "knowledge_role": knowledge_role("guideline" if "指南" in old.get("source_type", "") or "共识" in old.get("source_type", "") or "Guideline" in old["title"] else "research"),
            "expert_review_status": old.get("expert_decision", "pending"),
            "clinical_ready": False,
        })
        sources.append(record)
    for old in package["entries"]:
        source_ids = old.get("source_ids", [])
        entry = {
            "schema_version": "rehab.knowledge.entry.v2",
            "knowledge_id": old["knowledge_id"],
            "source_type": "guideline_or_research",
            "evidence_level": "research",
            "title": old.get("display_name", old.get("title", "")),
            "topic": old.get("domain", old.get("category", "")),
            "content": old.get("proposed_claim", old.get("content", "")),
            "evidence_summary": old.get("evidence_summary", ""),
            "source_ids": source_ids,
            "source_page_locators": old.get("source_page_locators", []),
            "evidence_tier_preserved": [next((s.get("evidence_tier") for s in package["sources"] if s["source_id"] == sid), None) for sid in source_ids],
            "expert_review_status": old.get("expert_decision", "pending"),
            "clinical_ready": False,
        }
        entries.append(entry)
        source = next((s for s in sources if s["source_id"] in source_ids), None)
        if source:
            locator = old.get("source_page_locators", [None])[0]
            page_match = re.search(r"第\s*(\d+)\s*页", locator or "")
            page = int(page_match.group(1)) if page_match else None
            content = "\n".join(x for x in (entry["title"], entry["content"], entry["evidence_summary"]) if x)
            chunks.append({
                "schema_version": "rehab.knowledge.chunk.v2",
                "chunk_id": f"{old['knowledge_id']}-guide-001",
                "content": content,
                "metadata": {
                    "source_id": source["source_id"],
                    "uid": source.get("uid") or source["source_id"],
                    "source_type": source["source_type"],
                    "evidence_level": "research",
                    "knowledge_role": source["knowledge_role"],
                    "title": entry["title"],
                    "topic": entry["topic"],
                    "page_start": page,
                    "page_end": page,
                    "page_numbers": [page] if page else [],
                    "source_pdf": source["source_pdf"],
                    "source_document": source["source_document"],
                    "source_file": source.get("source_pdf") or source.get("source_document"),
                    "source_page_locator": locator,
                    "source_view_route": f"/api/rag/sources/{source['source_id']}/pages/{page}" if page else f"/api/rag/sources/{source['source_id']}",
                    "knowledge_id": old["knowledge_id"],
                    "evidence_tier": source.get("evidence_tier"),
                    "expert_review_status": "pending",
                    "clinical_ready": False,
                },
            })
    return sources, entries, chunks


def build_entry_for_source(source: dict) -> dict:
    title = source.get("title") or source.get("book") or source.get("uid")
    return {
        "schema_version": "rehab.knowledge.entry.v2",
        "knowledge_id": source["source_id"],
        "source_id": source["source_id"],
        "uid": source.get("uid") or source["source_id"],
        "source_type": source["source_type"],
        "evidence_level": source["evidence_level"],
        "knowledge_role": source.get("knowledge_role") or knowledge_role(source["source_type"], source.get("classification")),
        "classification": source.get("classification"),
        "title": title,
        "topic": source.get("topic", ""),
        "book": source.get("book"),
        "source_pdf": source.get("source_pdf"),
        "page_count": source.get("page_count"),
        "evidence_quality": source.get("evidence_quality"),
        "evidence_tier": source.get("evidence_tier"),
        "expert_review_status": source.get("expert_review_status", "pending"),
        "clinical_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-method-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-incomplete-textbooks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-ocr-textbooks", action="store_true", help="Keep U179/U180/U181 source PDFs, but exclude their OCR text/chunks from this candidate build.")
    args = parser.parse_args()

    if OUT.exists():
        for child in OUT.iterdir():
            if child.name not in {"scripts", "tests", "README.md", "SOURCE_VIEW_INTEGRATION.md"}:
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    OUT.mkdir(parents=True, exist_ok=True)

    paper_sources: list[dict] = []
    paper_chunks: list[dict] = []
    paper_entries: list[dict] = []
    literature_rows = selected_literature_rows()
    for index, row in enumerate(literature_rows, 1):
        if row["ingest_role"] == "method_reference_only" and not args.include_method_reference:
            continue
        source, pdf = build_paper_source(row, ocr_excluded=args.exclude_ocr_textbooks)
        paper_sources.append(source)
        paper_entries.append(build_entry_for_source(source))
        # U177-U181 are textbook records in the same release. Preserve the
        # historical classification/source record, but avoid indexing the
        # same textbook content as a paper a second time.
        if row["uid"] not in TEXTBOOK_ALIAS_TO_SOURCE:
            paper_chunks.extend(build_paper_chunks(source, pdf))
        if index % 10 == 0 or index == len(literature_rows):
            print(f"literature {index}/{len(literature_rows)}: {row['uid']}", flush=True)

    guide_sources, guide_entries, guide_chunks = guide_records()

    textbook_sources: list[dict] = []
    textbook_entries: list[dict] = []
    textbook_chunks: list[dict] = []
    textbook_stats: dict[str, dict] = {}
    for uid in TEXTBOOKS:
        ocr_excluded = args.exclude_ocr_textbooks and uid in {"U179", "U180", "U181"}
        source = build_source_record(uid, ocr_excluded=ocr_excluded)
        textbook_sources.append(source)
        textbook_entries.append(build_entry_for_source(source))
        if uid == "U177":
            chunks = []
            stats = {
                "status": "reused_from_U178",
                "pages": source["page_count"],
                "success_pages": source.get("reused_page_count", 658),
                "failed_pages": [],
                "chunk_count": 0,
                "ocr_performed": False,
                "rag_indexed": False,
                "ocr_pending": False,
            }
        elif ocr_excluded:
            chunks = []
            stats = {
                "status": "ocr_pending",
                "pages": source["page_count"],
                "success_pages": 0,
                "failed_pages": [],
                "chunk_count": 0,
                "ocr_performed": False,
                "rag_indexed": False,
                "ocr_pending": True,
            }
        else:
            chunks, stats = build_textbook_chunks(uid, source)
            stats["status"] = "complete" if not stats["failed_pages"] else "partial"
            stats["ocr_performed"] = source["ocr_mode"] == "ocr_b"
            stats["rag_indexed"] = True
            stats["ocr_pending"] = False
        textbook_chunks.extend(chunks)
        textbook_stats[uid] = stats

    all_sources = paper_sources + guide_sources + textbook_sources
    all_entries = paper_entries + guide_entries + textbook_entries
    all_chunks = paper_chunks + guide_chunks + textbook_chunks
    write_jsonl(OUT / "sources.jsonl", all_sources)
    write_jsonl(OUT / "entries.jsonl", all_entries)
    write_jsonl(OUT / "chunks.jsonl", all_chunks)

    by_classification = Counter(s.get("classification") for s in paper_sources)
    by_type = Counter(s.get("source_type") for s in all_sources)
    chunk_by_type = Counter(c["metadata"].get("source_type") for c in all_chunks)
    manifest = {
        "schema_version": "rehab.knowledge.manifest.v2",
        "knowledge_base": "rehab_knowledge_v1",
        "release": "v1_candidate",
        "created_at_utc": now_utc(),
        "mode": "candidate_test_only",
        "architecture": {"embedding_model": "bge-m3", "embedding_dimensions": 1024, "distance": "Cosine", "reranker": "none", "retrieval": "dense_retrieval_plus_governance_filter"},
        "version_policy": {"v0.3_old": "existing production snapshot; immutable", "v1_candidate": "this additive release; selected by configuration", "default": "v0.3_old"},
        "ingestion_policy": {"reclassified": False, "evidence_levels_changed": False, "method_reference_included": args.include_method_reference, "method_reference_retrieval_role": "low_priority_reference", "raw_ocr_overwritten": False, "textbook_u177": "reuse U178 text layer; no OCR", "ocr_textbooks_excluded": args.exclude_ocr_textbooks, "textbook_alias_paper_chunks_excluded": True},
        "counts": {"sources": len(all_sources), "entries": len(all_entries), "chunks": len(all_chunks), "embeddings": 0, "paper_or_guideline_sources": len(paper_sources) + len(guide_sources), "textbook_source_records": len(textbook_sources), "paper_by_classification": dict(by_classification), "sources_by_type": dict(by_type), "chunks_by_type": dict(chunk_by_type)},
        "paper_counts": {"core": sum(v for k, v in by_classification.items() if k and k.startswith("core_")), "core_direct": by_classification.get("core_direct_candidate", 0), "core_background": by_classification.get("core_background_candidate", 0), "training": by_classification.get("training_recommendation_candidate", 0), "method": by_classification.get("method_reference_only", 0)},
        "textbook_ocr": textbook_stats,
        "artifacts": {"sources": "sources.jsonl", "entries": "entries.jsonl", "chunks": "chunks.jsonl", "embedding_manifest": "embedding_manifest.json", "source_view_contract": "source_view_contract.json", "retrieval_tests": "retrieval_tests.json"},
    }
    write_json(OUT / "manifest.json", manifest)
    write_json(OUT / "embedding_manifest.json", {"schema_version": "rehab.embedding.manifest.v1", "model": "bge-m3", "dimensions": 1024, "distance": "Cosine", "status": "pending_runner", "count": 0, "collection": "rehab_knowledge_v1_candidate", "source_chunks": len(all_chunks)})
    write_json(OUT / "source_view_contract.json", {"schema_version": "rehab.knowledge.source-view.v1", "route": "/api/rag/source-files/{source_file_id}/pages/{page}", "chain": "chunk_id -> uid -> source_file -> page_start/page_end -> original_pdf", "response": {"uid": "stable source UID", "source_file": "staged immutable source file", "source_pdf": "project-relative PDF path", "page": "1-based integer", "page_image": "optional rendered/OCR page image", "bbox": "optional page-level OCR boxes", "source_type": "paper|guideline|textbook", "evidence_level": "research|textbook_reference"}, "ui_requirements": ["显示原PDF或PDF内嵌页预览", "显示1-based页码", "点击来源直接打开对应PDF页", "教材和论文显示不同证据层级", "raw_ocr_text仅供审计，不能被corrected_text覆盖"]})
    write_json(OUT / "retrieval_tests.json", {"schema_version": "rehab.knowledge.retrieval-test.v1", "status": "pending_vector_build", "collection": "rehab_knowledge_v1_candidate", "queries": [], "note": "向量索引需使用既有 bge-m3/1024/Cosine runner；本次构建不替换旧 collection。"})
    print(json.dumps({"release": "v1_candidate", "sources": len(all_sources), "entries": len(all_entries), "chunks": len(all_chunks), "paper_counts": manifest["paper_counts"], "textbook_ocr": textbook_stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
