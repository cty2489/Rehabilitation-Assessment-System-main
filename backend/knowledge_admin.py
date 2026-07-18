"""Read-only, governed access to the active rehabilitation knowledge release."""

from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from biomarker_refs import REF_META

_COLLECTION_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_BIOMARKER_SYSTEM_KEYS = frozenset(REF_META)
_STATUS_ORDER = (
    "blocked_current_implementation",
    "research_only",
    "conditional_after_protocol_fix",
    "guideline_candidate_pending_expert",
)
_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, Tuple[Tuple[Tuple[str, int, int], ...], "KnowledgeSnapshot"]] = {}


class KnowledgeUnavailable(RuntimeError):
    """Raised when the governed runtime release cannot be read safely."""


@dataclass(frozen=True)
class KnowledgeSnapshot:
    root: Path
    manifest: Dict[str, Any]
    quality_report: Dict[str, Any]
    entries: Tuple[Dict[str, Any], ...]
    sources: Tuple[Dict[str, Any], ...]
    validation_issues: Tuple[str, ...]


def _runtime_root() -> Path:
    configured = os.getenv("KNOWLEDGE_RUNTIME_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "knowledge_base" / "runtime"


def active_collection_id() -> str:
    value = os.getenv("RAG_COLLECTION", "rehab_knowledge_trial_v0_2").strip()
    if not _COLLECTION_ID.fullmatch(value):
        raise KnowledgeUnavailable("RAG_COLLECTION 不是合法的集合标识")
    return value


def _collection_root() -> Path:
    return _runtime_root() / active_collection_id()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KnowledgeUnavailable(f"缺少知识发布文件：{path.name}") from exc
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise KnowledgeUnavailable(f"知识发布文件不可读：{path.name}") from exc
    if not isinstance(value, dict):
        raise KnowledgeUnavailable(f"知识发布文件格式错误：{path.name}")
    return value


def _read_jsonl(path: Path) -> Tuple[Dict[str, Any], ...]:
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record must be an object")
                records.append(value)
    except FileNotFoundError as exc:
        raise KnowledgeUnavailable(f"缺少知识发布文件：{path.name}") from exc
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        detail = f"第 {line_number} 行" if "line_number" in locals() else ""
        raise KnowledgeUnavailable(f"知识发布文件不可读：{path.name}{detail}") from exc
    return tuple(records)


def _signature(paths: Tuple[Path, ...]) -> Tuple[Tuple[str, int, int], ...]:
    signature = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError as exc:
            raise KnowledgeUnavailable(f"缺少知识发布文件：{path.name}") from exc
        signature.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _validate_snapshot(
    manifest: Dict[str, Any],
    quality_report: Dict[str, Any],
    entries: Tuple[Dict[str, Any], ...],
    sources: Tuple[Dict[str, Any], ...],
) -> Tuple[str, ...]:
    collection_id = active_collection_id()
    if manifest.get("schema_version") != "rehab.knowledge.manifest.v1":
        raise KnowledgeUnavailable("知识发布清单 schema_version 不受支持")
    if manifest.get("collection_id") != collection_id:
        raise KnowledgeUnavailable("知识发布清单与当前 RAG_COLLECTION 不一致")
    if quality_report.get("schema_version") != "rehab.knowledge.quality.v1":
        raise KnowledgeUnavailable("知识质量报告 schema_version 不受支持")
    if quality_report.get("collection_id") != collection_id:
        raise KnowledgeUnavailable("知识质量报告与当前 RAG_COLLECTION 不一致")

    entry_ids: set[str] = set()
    system_keys: set[str] = set()
    for entry in entries:
        knowledge_id = str(entry.get("knowledge_id") or "").strip()
        system_key = str(entry.get("system_key") or "").strip()
        if not knowledge_id or not system_key:
            raise KnowledgeUnavailable("知识条目缺少 knowledge_id 或 system_key")
        if knowledge_id in entry_ids:
            raise KnowledgeUnavailable(f"知识条目编号重复：{knowledge_id}")
        if system_key in system_keys:
            raise KnowledgeUnavailable(f"系统指标键重复：{system_key}")
        entry_ids.add(knowledge_id)
        system_keys.add(system_key)

    source_ids: set[str] = set()
    source_links: Dict[str, set[str]] = {}
    for source in sources:
        source_id = str(source.get("source_id") or "").strip()
        if not source_id or source_id in source_ids:
            raise KnowledgeUnavailable(f"来源编号缺失或重复：{source_id or '空值'}")
        source_ids.add(source_id)
        url = str(source.get("url") or "").strip()
        parsed_url = urlparse(url)
        if url and (
            parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc
        ):
            raise KnowledgeUnavailable(f"来源链接协议不安全：{source_id}")
        linked_ids = {
            str(value) for value in source.get("knowledge_ids", []) or [] if value
        }
        unknown_entries = linked_ids - entry_ids
        if unknown_entries:
            raise KnowledgeUnavailable(
                f"来源 {source_id} 关联了不存在的知识条目"
            )
        source_links[source_id] = linked_ids

    missing_sources: set[str] = set()
    inconsistent_links: set[str] = set()
    for entry in entries:
        knowledge_id = str(entry.get("knowledge_id") or "")
        for source_id in (entry.get("source") or {}).get("source_ids", []) or []:
            source_id_text = str(source_id)
            if source_id_text not in source_ids:
                missing_sources.add(source_id_text)
            elif knowledge_id not in source_links.get(source_id_text, set()):
                inconsistent_links.add(f"{knowledge_id}/{source_id_text}")
    if missing_sources:
        raise KnowledgeUnavailable(
            "知识条目引用了不存在的来源：" + "、".join(sorted(missing_sources))
        )
    if inconsistent_links:
        raise KnowledgeUnavailable(
            "知识条目与来源反向关联不一致：" + "、".join(sorted(inconsistent_links))
        )

    issues: List[str] = []
    manifest_counts = manifest.get("counts") or {}
    quality_counts = quality_report.get("counts") or {}
    expected_counts = {
        "total_entries": len(entries),
        "sources": len(sources),
        "clinical_ready_entries": sum(
            bool((entry.get("status") or {}).get("clinical_ready"))
            for entry in entries
        ),
    }
    for name, actual in expected_counts.items():
        if manifest_counts.get(name) != actual:
            issues.append(f"manifest.counts.{name} 与发布文件不一致")
        if name in quality_counts and quality_counts.get(name) != actual:
            issues.append(f"quality_report.counts.{name} 与发布文件不一致")

    biomarker_count = sum(
        str(entry.get("system_key") or "") in _BIOMARKER_SYSTEM_KEYS
        for entry in entries
    )
    if biomarker_count != 26:
        issues.append(f"系统指标映射为 {biomarker_count}/26")
    return tuple(issues)


def load_snapshot() -> KnowledgeSnapshot:
    root = _collection_root()
    paths = (
        root / "manifest.json",
        root / "quality_report.json",
        root / "entries.jsonl",
        root / "sources.jsonl",
    )
    signature = _signature(paths)
    cache_key = str(root.resolve())
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached and cached[0] == signature:
            return cached[1]

    manifest = _read_json(paths[0])
    quality_report = _read_json(paths[1])
    entries = _read_jsonl(paths[2])
    sources = _read_jsonl(paths[3])
    snapshot = KnowledgeSnapshot(
        root=root,
        manifest=manifest,
        quality_report=quality_report,
        entries=entries,
        sources=sources,
        validation_issues=_validate_snapshot(
            manifest, quality_report, entries, sources
        ),
    )
    with _CACHE_LOCK:
        _CACHE[cache_key] = (signature, snapshot)
    return snapshot


def _entry_summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    status = entry.get("status") or {}
    governance = entry.get("governance") or {}
    source = entry.get("source") or {}
    return {
        "knowledge_id": str(entry.get("knowledge_id") or ""),
        "entry_version": str(entry.get("entry_version") or ""),
        "title": str(entry.get("title") or ""),
        "category": str(entry.get("category") or ""),
        "system_key": str(entry.get("system_key") or ""),
        "knowledge_status": str(entry.get("knowledge_status") or ""),
        "knowledge_status_label": str(entry.get("knowledge_status_label") or ""),
        "clinical_ready": bool(status.get("clinical_ready")),
        "demo_ready": bool(status.get("demo_ready")),
        "expert_verified": bool(governance.get("expert_verified")),
        "expert_review_status": str(governance.get("expert_review_status") or ""),
        "source_ids": [str(value) for value in source.get("source_ids", []) or []],
        "issues": [str(value) for value in status.get("issues", []) or []],
    }


def _status_counts(entries: Tuple[Dict[str, Any], ...]) -> List[Dict[str, Any]]:
    counts = Counter(
        str(entry.get("knowledge_status") or "unknown") for entry in entries
    )
    biomarker_counts = Counter(
        str(entry.get("knowledge_status") or "unknown")
        for entry in entries
        if str(entry.get("system_key") or "") in _BIOMARKER_SYSTEM_KEYS
    )
    labels = {
        str(entry.get("knowledge_status") or "unknown"): str(
            entry.get("knowledge_status_label")
            or entry.get("knowledge_status")
            or "未知"
        )
        for entry in entries
    }
    keys = list(_STATUS_ORDER) + sorted(set(counts) - set(_STATUS_ORDER))
    return [
        {
            "status": key,
            "label": labels.get(key, key),
            "count": counts.get(key, 0),
            "biomarker_count": biomarker_counts.get(key, 0),
        }
        for key in keys
        if counts.get(key, 0)
    ]


def rag_service_status() -> Dict[str, Any]:
    mode = os.getenv("RAG_MODE", "off").strip().lower()
    result: Dict[str, Any] = {
        "mode": mode,
        "assist_approved": os.getenv("RAG_ASSIST_APPROVED", "").strip().lower()
        in {"1", "true", "yes", "on"},
        "demo_in_prompt": os.getenv("RAG_ALLOW_DEMO_IN_PROMPT", "").strip().lower()
        in {"1", "true", "yes", "on"},
        "service": {
            "reachable": False,
            "status": "not_checked" if mode == "off" else "unavailable",
            "collection": "",
            "collection_matches": False,
        },
    }
    if mode not in {"off", "shadow", "assist"}:
        result["service"]["status"] = "invalid_config"
        return result
    if mode == "off":
        result["service"]["status"] = "disabled"
        return result

    service_url = os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:8010").rstrip("/")
    parsed = urlparse(service_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        result["service"]["status"] = "invalid_config"
        return result
    try:
        import httpx

        response = httpx.get(f"{service_url}/health", timeout=1.5)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("health response is not an object")
        service_collection = str(payload.get("collection") or "")
        result["service"] = {
            "reachable": True,
            "status": str(payload.get("status") or "unknown"),
            "collection": service_collection,
            "collection_matches": service_collection == active_collection_id(),
        }
    except Exception:  # noqa: BLE001 - status endpoint must remain fail-open
        result["service"]["status"] = "unavailable"
    return result


def status_payload(
    *, app_version: str, build_commit: str, report_model: str
) -> Dict[str, Any]:
    rag = rag_service_status()
    base: Dict[str, Any] = {
        "schema_version": "rehab.knowledge.admin-status.v1",
        "available": False,
        "versions": {
            "application": app_version,
            "build_commit": build_commit,
            "report_model": report_model,
            "content_release": "",
            "source_document": "",
            "index_collection": os.getenv(
                "RAG_COLLECTION", "rehab_knowledge_trial_v0_2"
            ).strip(),
            "index_built_at_utc": "",
        },
        "rag": rag,
        "counts": {
            "total_entries": 0,
            "mapped_biomarkers": 0,
            "clinical_ready_biomarkers": 0,
            "expert_verified_entries": 0,
            "sources": 0,
        },
        "status_counts": [],
        "trial_release": {},
        "validation": {"valid": False, "issues": []},
    }
    try:
        snapshot = load_snapshot()
    except KnowledgeUnavailable as exc:
        base["error"] = str(exc)
        base["validation"]["issues"] = [str(exc)]
        return base

    entries = snapshot.entries
    biomarker_entries = tuple(
        entry
        for entry in entries
        if str(entry.get("system_key") or "") in _BIOMARKER_SYSTEM_KEYS
    )
    source = snapshot.manifest.get("source") or {}
    release = snapshot.manifest.get("trial_release") or {}
    base.update(
        {
            "available": True,
            "versions": {
                **base["versions"],
                "content_release": str(release.get("release_id") or ""),
                "source_document": str(source.get("document_id") or ""),
                "index_collection": str(snapshot.manifest.get("collection_id") or ""),
                "index_built_at_utc": str(
                    snapshot.manifest.get("created_at_utc") or ""
                ),
            },
            "counts": {
                "total_entries": len(entries),
                "mapped_biomarkers": len(biomarker_entries),
                "clinical_ready_biomarkers": sum(
                    bool((entry.get("status") or {}).get("clinical_ready"))
                    for entry in biomarker_entries
                ),
                "expert_verified_entries": sum(
                    bool((entry.get("governance") or {}).get("expert_verified"))
                    for entry in entries
                ),
                "sources": len(snapshot.sources),
            },
            "status_counts": _status_counts(entries),
            "trial_release": {
                "release_id": str(release.get("release_id") or ""),
                "expert_verified": bool(release.get("expert_verified")),
                "clinical_ready": bool(release.get("clinical_ready")),
                "warning": str(release.get("warning") or ""),
                "allowed_usage": [
                    str(value) for value in release.get("allowed_usage", []) or []
                ],
                "prohibited_usage": [
                    str(value) for value in release.get("prohibited_usage", []) or []
                ],
            },
            "validation": {
                "valid": not snapshot.validation_issues,
                "issues": list(snapshot.validation_issues),
            },
        }
    )
    return base


def entries_payload(
    *,
    category: Optional[str] = None,
    knowledge_status: Optional[str] = None,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    snapshot = load_snapshot()
    normalized_query = (query or "").strip().casefold()
    items = []
    for entry in snapshot.entries:
        if (
            category
            and str(entry.get("category") or "").casefold()
            != category.casefold()
        ):
            continue
        if knowledge_status and entry.get("knowledge_status") != knowledge_status:
            continue
        haystack = " ".join(
            str(value)
            for value in (
                entry.get("knowledge_id"),
                entry.get("title"),
                entry.get("system_key"),
                entry.get("category"),
                " ".join(entry.get("aliases", []) or []),
            )
        ).casefold()
        if normalized_query and normalized_query not in haystack:
            continue
        items.append(_entry_summary(entry))
    return {
        "schema_version": "rehab.knowledge.entries.v1",
        "total": len(items),
        "items": items,
        "filters": {
            "categories": sorted(
                {str(entry.get("category") or "") for entry in snapshot.entries}
            ),
            "statuses": _status_counts(snapshot.entries),
        },
    }


def coverage_payload() -> Dict[str, Any]:
    snapshot = load_snapshot()
    group_order = {"EMG": 0, "EEG": 1, "IMU": 2}
    items = [
        _entry_summary(entry)
        for entry in snapshot.entries
        if str(entry.get("system_key") or "") in _BIOMARKER_SYSTEM_KEYS
    ]
    items.sort(
        key=lambda item: (
            group_order.get(item["category"].upper(), 99),
            item["knowledge_id"],
        )
    )
    return {
        "schema_version": "rehab.knowledge.coverage.v1",
        "expected": 26,
        "mapped": len(items),
        "clinical_ready": sum(item["clinical_ready"] for item in items),
        "items": items,
    }


def sources_payload() -> Dict[str, Any]:
    snapshot = load_snapshot()
    items = sorted(
        snapshot.sources,
        key=lambda item: str(item.get("source_id") or ""),
    )
    return {
        "schema_version": "rehab.knowledge.sources.v1",
        "total": len(items),
        "items": list(items),
    }


def entry_payload(knowledge_id: str) -> Dict[str, Any]:
    snapshot = load_snapshot()
    entry = next(
        (item for item in snapshot.entries if item.get("knowledge_id") == knowledge_id),
        None,
    )
    if entry is None:
        raise KeyError(knowledge_id)
    source_ids = set((entry.get("source") or {}).get("source_ids", []) or [])
    sources = [
        source for source in snapshot.sources if source.get("source_id") in source_ids
    ]
    return {
        "schema_version": "rehab.knowledge.entry-detail.v1",
        "entry": {
            **_entry_summary(entry),
            "applicable_population": [
                str(value) for value in entry.get("applicable_population", []) or []
            ],
            "content": str(entry.get("content") or ""),
            "allowed_interpretation": str(entry.get("allowed_interpretation") or ""),
            "prohibited_interpretation": str(
                entry.get("prohibited_interpretation") or ""
            ),
            "acquisition_and_algorithm_requirements": str(
                entry.get("acquisition_and_algorithm_requirements") or ""
            ),
            "reference_range_policy": str(entry.get("reference_range_policy") or ""),
            "implementation_action": str(entry.get("implementation_action") or ""),
            "review_notes": [
                str(value) for value in entry.get("review_notes", []) or []
            ],
            "governance": {
                key: value
                for key, value in (entry.get("governance") or {}).items()
                if key
                in {
                    "source_status",
                    "expert_review_status",
                    "reviewed_by",
                    "reviewed_at",
                    "expert_verified",
                    "trial_release_id",
                }
            },
            "source_document": {
                key: value
                for key, value in (entry.get("source") or {}).items()
                if key in {"document_id", "filename", "sha256", "original_entry_number"}
            },
            "sources": sources,
        },
    }


__all__ = [
    "KnowledgeUnavailable",
    "active_collection_id",
    "coverage_payload",
    "entries_payload",
    "entry_payload",
    "load_snapshot",
    "sources_payload",
    "status_payload",
]
