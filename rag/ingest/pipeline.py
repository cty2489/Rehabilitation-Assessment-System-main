"""Convert the structured rehabilitation DOCX into governed JSONL records."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping

from .docx_parser import DocxBlock, ParsedDocx, parse_docx


_ENTRY_RE = re.compile(r"^条目\s*(\d+)\s*[：:]\s*(.+?)\s*$")
_FIELD_RE = re.compile(r"^[·•]\s*([^：:]+?)\s*[：:]\s*(.*)$")
_FIELD_NAMES = {
    "知识标题": "title",
    "知识类别": "category",
    "适用对象": "applicable_population",
    "正文内容": "content",
    "关键术语": "keywords",
    "注意事项": "cautions",
    "数据解释边界": "interpretation_boundary",
    "参考来源": "references_text",
    "审核专家": "reviewed_by",
    "审核日期": "reviewed_at",
    "版本号": "version",
}


def _split_list(value: str) -> List[str]:
    return [item.strip() for item in re.split(r"[、，,；;]", value) if item.strip()]


def _table_as_text(rows: Iterable[Iterable[str]]) -> str:
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)


def _append(fields: MutableMapping[str, str], key: str, value: str) -> None:
    value = value.strip()
    if not value:
        return
    fields[key] = f"{fields[key]}\n{value}".strip() if fields.get(key) else value


def parse_entries(blocks: Iterable[DocxBlock]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    active_field: str | None = None

    for block in blocks:
        if block.kind == "table":
            if current is not None:
                current["tables"].append(block.table_rows)
            continue

        text = block.text.strip()
        entry_match = _ENTRY_RE.match(text)
        if entry_match:
            if current is not None:
                entries.append(current)
            current = {
                "entry_number": int(entry_match.group(1)),
                "heading": entry_match.group(2).strip(),
                "fields": {},
                "images_by_field": {},
                "tables": [],
            }
            active_field = None
            continue
        if current is None:
            continue

        field_match = _FIELD_RE.match(text)
        if field_match:
            label = field_match.group(1).strip()
            active_field = _FIELD_NAMES.get(label)
            if active_field:
                _append(current["fields"], active_field, field_match.group(2))
        elif text and active_field:
            _append(current["fields"], active_field, text)

        if block.image_targets:
            image_field = active_field or "unassigned"
            current["images_by_field"].setdefault(image_field, []).extend(block.image_targets)

    if current is not None:
        entries.append(current)
    return entries


def _load_config(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _entry_record(
    raw: Dict[str, Any],
    parsed: ParsedDocx,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    number = str(raw["entry_number"])
    override = config.get("entry_overrides", {}).get(number, {})
    fields = dict(raw["fields"])
    for field_name, value in override.get("fields", {}).items():
        fields[field_name] = value

    title = fields.get("title") or raw["heading"]
    content = fields.get("content", "").strip()
    if override.get("include_tables_in_content") and raw["tables"]:
        table_text = "\n\n".join(_table_as_text(table) for table in raw["tables"])
        content = f"{content}\n{table_text}".strip()

    indexable = bool(override.get("indexable", True))
    kind = override.get("kind", "clinical_knowledge")
    issues: List[str] = []
    if not content:
        issues.append("missing_content")
        indexable = False
    if not fields.get("references_text"):
        issues.append("missing_reference_source")
    if not fields.get("reviewed_by"):
        issues.append("missing_expert_review")
    if raw["images_by_field"].get("content") and not override.get("content_image_transcribed"):
        issues.append("untranscribed_content_image")
        indexable = False
    issues.extend(override.get("issues", []))

    source_status = config.get("source_status", "pending")
    review_status = config.get("expert_review_status", "pending")
    demo_ready = indexable and bool(content)
    clinical_ready = (
        demo_ready
        and source_status == "verified"
        and review_status == "approved"
        and not issues
    )
    aliases = override.get("aliases", [])
    keywords = _split_list(fields.get("keywords", ""))
    return {
        "schema_version": "rehab.knowledge.entry.v1",
        "knowledge_id": override.get(
            "knowledge_id", f"KB-DEMO-{raw['entry_number']:03d}"
        ),
        "entry_version": fields.get("version") or config.get("entry_version", "0.1"),
        "kind": kind,
        "title": title,
        "category": fields.get("category", ""),
        "applicable_population": _split_list(fields.get("applicable_population", "")),
        "content": content,
        "keywords": keywords,
        "aliases": aliases,
        "cautions": fields.get("cautions", ""),
        "interpretation_boundary": fields.get("interpretation_boundary", ""),
        "references": (
            [fields["references_text"]] if fields.get("references_text") else []
        ),
        "review_notes": override.get("review_notes", []),
        "source": {
            "document_id": config["document_id"],
            "filename": parsed.filename,
            "sha256": parsed.sha256,
            "original_entry_number": raw["entry_number"],
            "image_refs": raw["images_by_field"],
        },
        "governance": {
            "source_status": source_status,
            "expert_review_status": review_status,
            "reviewed_by": fields.get("reviewed_by", ""),
            "reviewed_at": fields.get("reviewed_at", ""),
        },
        "status": {
            "indexable": indexable,
            "demo_ready": demo_ready,
            "clinical_ready": clinical_ready,
            "issues": sorted(set(issues)),
        },
    }


def _chunk_record(entry: Dict[str, Any]) -> Dict[str, Any]:
    sections = [
        f"标题：{entry['title']}",
        f"类别：{entry['category']}" if entry["category"] else "",
        f"适用对象：{'、'.join(entry['applicable_population'])}"
        if entry["applicable_population"]
        else "",
        f"正文：{entry['content']}",
        f"注意事项：{entry['cautions']}" if entry["cautions"] else "",
        f"解释边界：{entry['interpretation_boundary']}"
        if entry["interpretation_boundary"]
        else "",
    ]
    return {
        "schema_version": "rehab.knowledge.chunk.v1",
        "chunk_id": f"{entry['knowledge_id']}@{entry['entry_version']}#001",
        "knowledge_id": entry["knowledge_id"],
        "entry_version": entry["entry_version"],
        "text": "\n".join(section for section in sections if section),
        "metadata": {
            "title": entry["title"],
            "category": entry["category"],
            "keywords": entry["keywords"],
            "aliases": entry["aliases"],
            "clinical_ready": entry["status"]["clinical_ready"],
            "governance_issues": entry["status"]["issues"],
            "source_document_id": entry["source"]["document_id"],
            "source_filename": entry["source"]["filename"],
            "source_sha256": entry["source"]["sha256"],
            "source_entry_number": entry["source"]["original_entry_number"],
            "references": entry["references"],
            "reviewed_by": entry["governance"]["reviewed_by"],
            "reviewed_at": entry["governance"]["reviewed_at"],
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def prepare_knowledge_base(
    input_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    parsed = parse_docx(input_path)
    config = _load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assets_dir = output / "assets"
    assets_dir.mkdir(exist_ok=True)

    raw_entries = parse_entries(parsed.blocks)
    entries = [_entry_record(item, parsed, config) for item in raw_entries]
    chunks = [_chunk_record(item) for item in entries if item["status"]["indexable"]]

    for target, data in parsed.media.items():
        (assets_dir / Path(target).name).write_bytes(data)

    counts = {
        "total_entries": len(entries),
        "indexable_entries": sum(item["status"]["indexable"] for item in entries),
        "demo_ready_entries": sum(item["status"]["demo_ready"] for item in entries),
        "clinical_ready_entries": sum(item["status"]["clinical_ready"] for item in entries),
        "excluded_entries": sum(not item["status"]["indexable"] for item in entries),
        "chunks": len(chunks),
        "media_files": len(parsed.media),
    }
    quality_report = {
        "schema_version": "rehab.knowledge.quality.v1",
        "collection_id": config["collection_id"],
        "counts": counts,
        "entries": [
            {
                "knowledge_id": item["knowledge_id"],
                "title": item["title"],
                **item["status"],
            }
            for item in entries
        ],
        "release_decision": "demo_only" if not counts["clinical_ready_entries"] else "review_required",
    }
    manifest = {
        "schema_version": "rehab.knowledge.manifest.v1",
        "collection_id": config["collection_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "document_id": config["document_id"],
            "filename": parsed.filename,
            "sha256": parsed.sha256,
        },
        "counts": counts,
        "artifacts": ["entries.jsonl", "chunks.jsonl", "quality_report.json", "assets/"],
    }

    _write_jsonl(output / "entries.jsonl", entries)
    _write_jsonl(output / "chunks.jsonl", chunks)
    _write_json(output / "quality_report.json", quality_report)
    _write_json(output / "manifest.json", manifest)
    return {"manifest": manifest, "quality_report": quality_report}


__all__ = ["parse_entries", "prepare_knowledge_base"]
