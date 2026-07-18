"""Build deterministic, reader-facing citations from governed RAG evidence.

The LLM cites internal knowledge IDs such as ``[KB-EMG-009]``.  Those IDs are
useful for auditing but are not suitable for a clinical report.  This module
maps them to a report-local numeric bibliography (``【1】`` / ``【2】``), while
keeping the knowledge-ID and source-ID relationships in structured metadata.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence


KNOWLEDGE_CITATION_RE = re.compile(r"\[(KB-[A-Za-z0-9._:-]+)\]")
NUMERIC_CITATION_RE = re.compile(r"【(\d+)】")
_SOURCE_PREFIX_RE = re.compile(
    r"^\s*[\[【](SRC-[A-Za-z0-9._:-]+)[\]】]\s*(?:[：:]\s*)?",
    re.IGNORECASE,
)
_LEGACY_NUMBER_PREFIX_RE = re.compile(r"^\s*\[\d+\]\s*")


def _unique_strings(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def extract_knowledge_ids(value: Any) -> List[str]:
    """Return internal knowledge IDs in first-appearance order."""
    texts: List[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return _unique_strings(
        match.group(1)
        for text in texts
        for match in KNOWLEDGE_CITATION_RE.finditer(text)
    )


def extract_numeric_citations(value: Any) -> List[int]:
    """Return report-local numeric citations in first-appearance order."""
    seen = set()
    numbers: List[int] = []
    for match in NUMERIC_CITATION_RE.finditer(str(value or "")):
        number = int(match.group(1))
        if number not in seen:
            seen.add(number)
            numbers.append(number)
    return numbers


def citation_markers(numbers: Iterable[Any]) -> str:
    values: List[int] = []
    seen = set()
    for value in numbers:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in seen:
            seen.add(number)
            values.append(number)
    return "".join(f"【{number}】" for number in values)


def _source_map(rag_evidence: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    sources: List[Any] = list(rag_evidence.get("sources") or [])
    sources.extend((rag_evidence.get("marker_sources") or {}).values())
    result: Dict[str, Dict[str, Any]] = {}
    for raw in sources:
        if not isinstance(raw, Mapping):
            continue
        knowledge_id = str(raw.get("knowledge_id") or "").strip()
        if not knowledge_id:
            continue
        source = result.setdefault(knowledge_id, {})
        for key, value in raw.items():
            if key == "references":
                source[key] = _unique_strings(
                    list(source.get(key) or []) + list(value or [])
                )
            elif value not in (None, "", [], {}):
                source[key] = value
    return result


def _reference_parts(raw: Any) -> tuple[str | None, str]:
    text = str(raw or "").strip()
    match = _SOURCE_PREFIX_RE.match(text)
    source_id = match.group(1).upper() if match else None
    if match:
        text = text[match.end():].strip()
    else:
        text = _LEGACY_NUMBER_PREFIX_RE.sub("", text).strip()
    return source_id, text


def _fallback_reference(source: Mapping[str, Any], knowledge_id: str) -> tuple[str | None, str]:
    title = str(source.get("title") or knowledge_id).strip()
    document_id = str(source.get("source_document_id") or "").strip()
    entry_number = source.get("source_entry_number")
    location = ""
    if document_id:
        location = f"；来源文档：{document_id}"
    if entry_number not in (None, ""):
        location += f"；条目：{entry_number}"
    return document_id or None, f"{title}（知识库条目 {knowledge_id}{location}；非外部文献）"


def build_reference_catalog(
    rag_evidence: Mapping[str, Any] | None,
    cited_knowledge_ids: Sequence[Any] | None = None,
    *,
    body: Any = None,
) -> Dict[str, Any]:
    """Create a stable numeric bibliography for one report.

    Numbering follows first appearance in the report body, then appends any
    declared-but-not-inline ``rag_citations``.  Duplicate source IDs (or, when a
    source ID is unavailable, duplicate citation strings) share one number.
    """
    packet = rag_evidence if isinstance(rag_evidence, Mapping) else {}
    sources_by_id = _source_map(packet)
    ordered_ids = _unique_strings(
        extract_knowledge_ids(body) + list(cited_knowledge_ids or [])
    )

    references: List[Dict[str, Any]] = []
    references_by_key: Dict[str, Dict[str, Any]] = {}
    knowledge: Dict[str, Dict[str, Any]] = {}

    for knowledge_id in ordered_ids:
        source = sources_by_id.get(knowledge_id)
        if not source:
            continue
        raw_references = list(source.get("references") or [])
        parsed = [_reference_parts(value) for value in raw_references]
        parsed = [(source_id, citation) for source_id, citation in parsed if citation]
        if not parsed:
            parsed = [_fallback_reference(source, knowledge_id)]

        numbers: List[int] = []
        for source_id, citation in parsed:
            normalized_citation = re.sub(r"\s+", " ", citation).strip().lower()
            identity = (
                f"source:{source_id.lower()}"
                if source_id
                else f"text:{normalized_citation}"
            )
            item = references_by_key.get(identity)
            if item is None:
                item = {
                    "number": len(references) + 1,
                    "marker": f"【{len(references) + 1}】",
                    "source_id": source_id,
                    "citation": citation,
                    "knowledge_ids": [],
                }
                references_by_key[identity] = item
                references.append(item)
            if knowledge_id not in item["knowledge_ids"]:
                item["knowledge_ids"].append(knowledge_id)
            numbers.append(int(item["number"]))

        citation_numbers = sorted(set(numbers))
        knowledge[knowledge_id] = {
            "knowledge_id": knowledge_id,
            "title": source.get("title") or knowledge_id,
            "knowledge_status": source.get("knowledge_status") or "",
            "knowledge_status_label": source.get("knowledge_status_label") or "",
            "clinical_ready": bool(source.get("clinical_ready")),
            "reviewed_by": source.get("reviewed_by") or "",
            "reviewed_at": source.get("reviewed_at") or "",
            "citation_numbers": citation_numbers,
            "source_ids": _unique_strings(
                references[number - 1].get("source_id")
                for number in citation_numbers
                if 0 < number <= len(references)
            ),
        }

    return {
        "style": "numeric_square_brackets_zh",
        "knowledge": knowledge,
        "references": references,
    }


def render_numeric_citations(value: Any, catalog: Mapping[str, Any]) -> str:
    """Replace safe internal ``[KB-*]`` tokens with report-local ``【n】``."""
    text = str(value or "")
    knowledge = catalog.get("knowledge") or {}

    def replace(match: re.Match[str]) -> str:
        item = knowledge.get(match.group(1)) or {}
        markers = citation_markers(item.get("citation_numbers") or [])
        return markers

    return KNOWLEDGE_CITATION_RE.sub(replace, text)


__all__ = [
    "KNOWLEDGE_CITATION_RE",
    "NUMERIC_CITATION_RE",
    "build_reference_catalog",
    "citation_markers",
    "extract_knowledge_ids",
    "extract_numeric_citations",
    "render_numeric_citations",
]
