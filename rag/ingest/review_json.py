"""Convert the expert-review JSON draft into governed RAG records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


_REQUIRED_ENTRY_FIELDS = {
    "knowledge_id",
    "domain",
    "display_name",
    "system_key",
    "status",
    "status_label",
    "proposed_claim",
    "allowed_interpretation",
    "prohibited_interpretation",
    "source_ids",
}


def _read_document(path: Path) -> tuple[Dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid review JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("review JSON root must be an object")
    return document, hashlib.sha256(raw).hexdigest()


def _require_text(value: Any, field: str, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}: {field} must be a non-empty string")
    return value.strip()


def _validate_document(
    document: Dict[str, Any],
    *,
    allow_internal_trial: bool,
) -> None:
    entries = document.get("entries")
    sources = document.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries must be a non-empty list")
    if not isinstance(sources, list):
        raise ValueError("sources must be a list")

    release = document.get("trial_release")
    if not document.get("clinical_ready"):
        if not allow_internal_trial:
            raise ValueError(
                "unreviewed knowledge requires --allow-internal-trial and an isolated collection"
            )
        if not isinstance(release, dict):
            raise ValueError("internal trial knowledge must include trial_release")
        if release.get("expert_verified") is not False:
            raise ValueError("trial_release.expert_verified must remain false")
        if release.get("clinical_ready") is not False:
            raise ValueError("trial_release.clinical_ready must remain false")
        _require_text(release.get("release_id"), "release_id", "trial_release")

    source_ids = {
        source.get("source_id")
        for source in sources
        if isinstance(source, dict) and source.get("source_id")
    }
    seen: set[str] = set()
    seen_system_keys: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"entries[{index}] must be an object")
        missing = _REQUIRED_ENTRY_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"entries[{index}] missing fields: {sorted(missing)}")
        knowledge_id = _require_text(
            entry.get("knowledge_id"), "knowledge_id", f"entries[{index}]"
        )
        if knowledge_id in seen:
            raise ValueError(f"duplicate knowledge_id: {knowledge_id}")
        seen.add(knowledge_id)
        system_key = _require_text(
            entry.get("system_key"), "system_key", f"entries[{index}]"
        )
        if system_key in seen_system_keys:
            raise ValueError(f"duplicate system_key: {system_key}")
        seen_system_keys.add(system_key)
        unknown_sources = set(entry.get("source_ids") or []) - source_ids
        if unknown_sources:
            raise ValueError(
                f"{knowledge_id} references unknown sources: {sorted(unknown_sources)}"
            )


def _reference_text(source: Dict[str, Any]) -> str:
    source_id = str(source.get("source_id") or "")
    title = str(source.get("title") or "")
    year = str(source.get("year") or "")
    tier = str(source.get("evidence_tier") or "")
    url = str(source.get("url") or "")
    details = ", ".join(
        value for value in (year, f"证据等级 {tier}" if tier else "") if value
    )
    suffix = f" ({details})" if details else ""
    link = f" {url}" if url else ""
    return f"[{source_id}] {title}{suffix}{link}".strip()


def _entry_text(entry: Dict[str, Any], warning: str) -> str:
    aliases = "、".join(str(value) for value in entry.get("aliases", []) if value)
    fields = [
        ("知识名称", entry["display_name"]),
        ("领域", entry["domain"]),
        ("系统指标键", entry["system_key"]),
        ("别名", aliases),
        ("当前知识状态", entry["status_label"]),
        ("核心结论", entry["proposed_claim"]),
        ("允许解释", entry["allowed_interpretation"]),
        ("禁止解释", entry["prohibited_interpretation"]),
        ("采集与算法要求", entry.get("acquisition_and_algorithm_requirements")),
        ("当前系统映射", entry.get("current_system_mapping")),
        ("参考范围政策", entry.get("reference_range_policy")),
        ("证据摘要", entry.get("evidence_summary")),
        ("实施动作", entry.get("implementation_action")),
        ("内部试运行边界", warning),
    ]
    return "\n".join(f"{label}：{value}" for label, value in fields if value)


def _entry_record(
    entry: Dict[str, Any],
    *,
    index: int,
    document: Dict[str, Any],
    source_path: Path,
    source_sha256: str,
    sources_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    release = document.get("trial_release") or {}
    warning = str(
        release.get("warning")
        or "仅供内部 RAG 功能试运行，尚未完成正式专家复核。"
    )
    references = [
        _reference_text(sources_by_id[source_id])
        for source_id in entry.get("source_ids", [])
        if source_id in sources_by_id
    ]
    expert_decision = str(entry.get("expert_decision") or "pending").strip()
    reviewer = str(entry.get("expert_reviewer") or "").strip()
    reviewed_at = str(entry.get("expert_review_date") or "").strip()
    clinical_ready = bool(
        document.get("clinical_ready")
        and entry.get("clinical_ready")
        and expert_decision == "approved"
        and reviewer
        and reviewed_at
    )
    issues = [f"knowledge_status:{entry['status']}"]
    if not clinical_ready:
        issues.extend(["internal_trial_only", "formal_expert_review_pending"])
    return {
        "schema_version": "rehab.knowledge.entry.v1",
        "knowledge_id": entry["knowledge_id"],
        "entry_version": str(document.get("version") or "0.1").lstrip("v"),
        "kind": "clinical_knowledge",
        "title": entry["display_name"],
        "category": entry["domain"],
        "applicable_population": [
            str(
                (document.get("scope") or {}).get("population")
                or "成人脑卒中后上肢康复"
            )
        ],
        "content": entry["proposed_claim"],
        "allowed_interpretation": entry["allowed_interpretation"],
        "prohibited_interpretation": entry["prohibited_interpretation"],
        "acquisition_and_algorithm_requirements": str(
            entry.get("acquisition_and_algorithm_requirements") or ""
        ),
        "reference_range_policy": str(
            entry.get("reference_range_policy") or ""
        ),
        "implementation_action": str(entry.get("implementation_action") or ""),
        "keywords": list(
            dict.fromkeys(
                [
                    entry["system_key"],
                    *[str(value) for value in entry.get("aliases", []) if value],
                ]
            )
        ),
        "aliases": [str(value) for value in entry.get("aliases", []) if value],
        "cautions": str(entry.get("acquisition_and_algorithm_requirements") or ""),
        "interpretation_boundary": "\n".join(
            [
                f"允许解释：{entry['allowed_interpretation']}",
                f"禁止解释：{entry['prohibited_interpretation']}",
                f"参考范围政策：{entry.get('reference_range_policy') or '未提供'}",
            ]
        ),
        "references": references,
        "review_notes": [
            str(value)
            for value in (
                entry.get("evidence_summary"),
                entry.get("expert_question"),
                entry.get("expert_comment"),
            )
            if value
        ],
        "source": {
            "document_id": str(document.get("schema_version") or source_path.stem),
            "filename": source_path.name,
            "sha256": source_sha256,
            "original_entry_number": index,
            "source_ids": list(entry.get("source_ids") or []),
        },
        "governance": {
            "source_status": str(document.get("status") or "pending_expert_review"),
            "expert_review_status": expert_decision,
            "reviewed_by": reviewer,
            "reviewed_at": reviewed_at,
            "expert_verified": clinical_ready,
            "trial_release_id": str(release.get("release_id") or ""),
        },
        "status": {
            "indexable": True,
            "demo_ready": True,
            "clinical_ready": clinical_ready,
            "issues": issues,
        },
        "system_key": entry["system_key"],
        "knowledge_status": entry["status"],
        "knowledge_status_label": entry["status_label"],
        "trial_text": _entry_text(entry, warning),
    }


def _chunk_record(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "rehab.knowledge.chunk.v1",
        "chunk_id": f"{entry['knowledge_id']}@{entry['entry_version']}#001",
        "knowledge_id": entry["knowledge_id"],
        "entry_version": entry["entry_version"],
        "text": entry["trial_text"],
        "metadata": {
            "title": entry["title"],
            "category": entry["category"],
            "keywords": entry["keywords"],
            "aliases": entry["aliases"],
            "system_key": entry["system_key"],
            "knowledge_status": entry["knowledge_status"],
            "knowledge_status_label": entry["knowledge_status_label"],
            "proposed_claim": entry["content"],
            "allowed_interpretation": str(
                entry.get("allowed_interpretation") or ""
            ),
            "prohibited_interpretation": str(
                entry.get("prohibited_interpretation") or ""
            ),
            "acquisition_and_algorithm_requirements": str(
                entry.get("acquisition_and_algorithm_requirements") or ""
            ),
            "reference_range_policy": str(
                entry.get("reference_range_policy") or ""
            ),
            "implementation_action": str(
                entry.get("implementation_action") or ""
            ),
            "clinical_ready": entry["status"]["clinical_ready"],
            "expert_verified": entry["governance"]["expert_verified"],
            "trial_release_id": entry["governance"]["trial_release_id"],
            "governance_issues": entry["status"]["issues"],
            "source_document_id": entry["source"]["document_id"],
            "source_filename": entry["source"]["filename"],
            "source_sha256": entry["source"]["sha256"],
            "source_entry_number": entry["source"]["original_entry_number"],
            "source_ids": entry["source"]["source_ids"],
            "references": entry["references"],
            "reviewed_by": entry["governance"]["reviewed_by"],
            "reviewed_at": entry["governance"]["reviewed_at"],
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, values: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def prepare_review_json_knowledge_base(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    collection_id: str = "rehab_knowledge_trial_v0_2",
    allow_internal_trial: bool = False,
) -> Dict[str, Any]:
    source_path = Path(input_path)
    document, source_sha256 = _read_document(source_path)
    _validate_document(document, allow_internal_trial=allow_internal_trial)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sources_by_id = {
        source["source_id"]: source
        for source in document["sources"]
        if isinstance(source, dict) and source.get("source_id")
    }
    entries = [
        _entry_record(
            raw_entry,
            index=index,
            document=document,
            source_path=source_path,
            source_sha256=source_sha256,
            sources_by_id=sources_by_id,
        )
        for index, raw_entry in enumerate(document["entries"], start=1)
    ]
    chunks = [
        _chunk_record(entry) for entry in entries if entry["status"]["indexable"]
    ]
    evaluation_questions: List[Dict[str, Any]] = []
    for question in document.get("evaluation_questions", []) or []:
        if not isinstance(question, dict) or not question.get("question"):
            continue
        evaluation_questions.append(
            {
                "question_id": question.get("question_id"),
                "category": question.get("category"),
                "query": question.get("question"),
                "expected_knowledge_ids": question.get("expected_knowledge_ids") or [],
                "expected_behavior": question.get("expected_behavior"),
                "disallowed_behavior": question.get("disallowed_behavior"),
            }
        )

    counts = {
        "total_entries": len(entries),
        "indexable_entries": sum(entry["status"]["indexable"] for entry in entries),
        "demo_ready_entries": sum(entry["status"]["demo_ready"] for entry in entries),
        "clinical_ready_entries": sum(
            entry["status"]["clinical_ready"] for entry in entries
        ),
        "excluded_entries": sum(
            not entry["status"]["indexable"] for entry in entries
        ),
        "chunks": len(chunks),
        "evaluation_questions": len(evaluation_questions),
    }
    quality_report = {
        "schema_version": "rehab.knowledge.quality.v1",
        "collection_id": collection_id,
        "counts": counts,
        "entries": [
            {
                "knowledge_id": entry["knowledge_id"],
                "title": entry["title"],
                "knowledge_status": entry["knowledge_status"],
                **entry["status"],
            }
            for entry in entries
        ],
        "release_decision": "internal_trial_only"
        if not counts["clinical_ready_entries"]
        else "review_required",
    }
    manifest = {
        "schema_version": "rehab.knowledge.manifest.v1",
        "collection_id": collection_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "document_id": str(document.get("schema_version") or source_path.stem),
            "filename": source_path.name,
            "sha256": source_sha256,
        },
        "trial_release": document.get("trial_release") or {},
        "counts": counts,
        "artifacts": [
            "entries.jsonl",
            "chunks.jsonl",
            "evaluation_queries.jsonl",
            "quality_report.json",
        ],
    }

    _write_jsonl(output / "entries.jsonl", entries)
    _write_jsonl(output / "chunks.jsonl", chunks)
    _write_jsonl(output / "evaluation_queries.jsonl", evaluation_questions)
    _write_json(output / "quality_report.json", quality_report)
    _write_json(output / "manifest.json", manifest)
    return {"manifest": manifest, "quality_report": quality_report}


__all__ = ["prepare_review_json_knowledge_base"]
