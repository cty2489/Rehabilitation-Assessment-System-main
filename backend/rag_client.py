"""Fail-open client for report-time RAG shadow and governed assist modes."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse


_TRACE_LOCK = threading.Lock()
_MODES = {"off", "shadow", "assist"}
_CORRELATION_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class RagClientSettings:
    mode: str
    service_url: str
    timeout_seconds: float
    top_k_per_query: int
    max_sources: int
    max_context_chars: int
    assist_approved: bool
    shadow_include_demo: bool
    allow_demo_in_prompt: bool
    trace_enabled: bool
    trace_path: Path

    @classmethod
    def from_env(cls) -> "RagClientSettings":
        mode = os.getenv("RAG_MODE", "off").strip().lower()
        if mode not in _MODES:
            raise ValueError("RAG_MODE must be off, shadow or assist")
        service_url = os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:8010").rstrip("/")
        parsed = urlparse(service_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("RAG_SERVICE_URL must point to localhost")
        timeout_seconds = float(os.getenv("RAG_TIMEOUT_SECONDS", "20"))
        if timeout_seconds <= 0:
            raise ValueError("RAG_TIMEOUT_SECONDS must be positive")
        default_trace = (
            Path(__file__).resolve().parents[1]
            / "knowledge_base/runtime/rag_traces/report_retrieval.jsonl"
        )
        return cls(
            mode=mode,
            service_url=service_url,
            timeout_seconds=timeout_seconds,
            top_k_per_query=_positive_int("RAG_REPORT_TOP_K", 2),
            max_sources=_positive_int("RAG_MAX_SOURCES", 6),
            max_context_chars=_positive_int("RAG_MAX_CONTEXT_CHARS", 8000),
            assist_approved=_env_bool("RAG_ASSIST_APPROVED", False),
            shadow_include_demo=_env_bool("RAG_SHADOW_INCLUDE_DEMO", False),
            allow_demo_in_prompt=_env_bool("RAG_ALLOW_DEMO_IN_PROMPT", False),
            trace_enabled=_env_bool("RAG_TRACE_ENABLED", True),
            trace_path=Path(os.getenv("RAG_TRACE_PATH", str(default_trace))),
        )


def build_report_queries(context: Dict[str, Any]) -> List[Dict[str, str]]:
    """Create de-identified retrieval queries from code-owned assessment data."""
    predictions = context.get("predictions") or {}
    stage = str(context.get("stage_roman") or context.get("stage") or "")
    queries = [
        {
            "key": "clinical_scales",
            "text": (
                "脑卒中上肢康复评估中 FMA 手部分数、手部 MAS 肌张力和 "
                f"Brunnstrom 手功能 {stage} 期的解释边界与随访意义；"
                f"本次 FMA={predictions.get('FMA_UE')}，MAS={predictions.get('hand_tone')}。"
            ),
        }
    ]
    for group in (context.get("biomarkers") or {}).get("groups", []) or []:
        markers = [
            str(marker.get("name") or marker.get("key") or "").strip()
            for marker in group.get("markers", []) or []
            if marker.get("available", True)
        ]
        markers = [value for value in markers if value]
        if not markers:
            continue
        key = str(group.get("key") or "biomarker").strip()
        label = str(group.get("label") or key).strip()
        queries.append(
            {
                "key": key,
                "text": (
                    f"脑卒中上肢主动运动评估中的{label}指标解释边界、同设备复测意义与康复随访："
                    + "、".join(markers[:12])
                ),
            }
        )
    return queries[:8]


def build_marker_system_keys(context: Dict[str, Any]) -> List[str]:
    """Return available biomarker keys in deterministic report order."""
    keys: List[str] = []
    for group in (context.get("biomarkers") or {}).get("groups", []) or []:
        for marker in group.get("markers", []) or []:
            key = str(marker.get("key") or "").strip()
            if key and marker.get("available", True) and key not in keys:
                keys.append(key)
    return keys


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    import httpx

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("RAG service returned a non-object response")
    return value


def _packet(mode: str, status: str, **extra: Any) -> Dict[str, Any]:
    return {
        "schema_version": "rehab.rag.report-evidence.v1",
        "trace_id": uuid.uuid4().hex,
        "mode": mode,
        "status": status,
        "used_in_prompt": False,
        "used_in_report": False,
        "marker_grounding_used": False,
        "marker_grounding_complete": False,
        "collection": "",
        "queries": [],
        "sources": [],
        "marker_sources": {},
        **extra,
    }


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _source_from_hit(hit: Dict[str, Any], *, text: str) -> Dict[str, Any]:
    metadata = dict(hit.get("metadata") or {})
    return {
        "knowledge_id": str(hit.get("knowledge_id") or ""),
        "chunk_id": str(hit.get("chunk_id") or ""),
        "title": str(hit.get("title") or ""),
        "text": text,
        "score": float(hit.get("score") or 0.0),
        "system_key": str(metadata.get("system_key") or ""),
        "clinical_ready": bool(metadata.get("clinical_ready")),
        "expert_verified": bool(metadata.get("expert_verified")),
        "knowledge_status": str(metadata.get("knowledge_status") or ""),
        "knowledge_status_label": str(
            metadata.get("knowledge_status_label") or ""
        ),
        "proposed_claim": str(metadata.get("proposed_claim") or ""),
        "allowed_interpretation": str(
            metadata.get("allowed_interpretation") or ""
        ),
        "prohibited_interpretation": str(
            metadata.get("prohibited_interpretation") or ""
        ),
        "acquisition_and_algorithm_requirements": str(
            metadata.get("acquisition_and_algorithm_requirements") or ""
        ),
        "reference_range_policy": str(
            metadata.get("reference_range_policy") or ""
        ),
        "implementation_action": str(metadata.get("implementation_action") or ""),
        "trial_release_id": str(metadata.get("trial_release_id") or ""),
        "source_document_id": str(metadata.get("source_document_id") or ""),
        "source_sha256": str(metadata.get("source_sha256") or ""),
        "source_entry_number": metadata.get("source_entry_number"),
        "references": _string_list(metadata.get("references")),
        "reviewed_by": str(metadata.get("reviewed_by") or ""),
        "reviewed_at": str(metadata.get("reviewed_at") or ""),
    }


def _context_correlation_id(context: Dict[str, Any]) -> str:
    value = str(
        (context.get("assessment_context") or {}).get("rag_correlation_id") or ""
    ).strip()
    return value if _CORRELATION_ID.fullmatch(value) else ""


def _append_trace(packet: Dict[str, Any], settings: RagClientSettings) -> None:
    if not settings.trace_enabled or settings.mode == "off":
        return
    trace = {
        "schema_version": "rehab.rag.report-trace.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_id": packet.get("trace_id"),
        "correlation_id": packet.get("correlation_id"),
        "mode": packet.get("mode"),
        "status": packet.get("status"),
        "used_in_prompt": packet.get("used_in_prompt"),
        "used_in_report": packet.get("used_in_report"),
        "marker_grounding_used": packet.get("marker_grounding_used"),
        "marker_grounding_complete": packet.get("marker_grounding_complete"),
        "collection": packet.get("collection"),
        "elapsed_ms": packet.get("elapsed_ms"),
        "queries": packet.get("queries", []),
        "sources": [
            {
                "knowledge_id": source.get("knowledge_id"),
                "chunk_id": source.get("chunk_id"),
                "title": source.get("title"),
                "score": source.get("score"),
                "clinical_ready": source.get("clinical_ready"),
                "expert_verified": source.get("expert_verified"),
                "knowledge_status": source.get("knowledge_status"),
                "trial_release_id": source.get("trial_release_id"),
                "source_document_id": source.get("source_document_id"),
                "source_sha256": source.get("source_sha256"),
            }
            for source in packet.get("sources", [])
        ],
        "marker_sources": [
            {
                "system_key": system_key,
                "knowledge_id": source.get("knowledge_id"),
                "chunk_id": source.get("chunk_id"),
                "title": source.get("title"),
                "clinical_ready": source.get("clinical_ready"),
                "expert_verified": source.get("expert_verified"),
                "knowledge_status": source.get("knowledge_status"),
                "trial_release_id": source.get("trial_release_id"),
                "source_document_id": source.get("source_document_id"),
                "source_sha256": source.get("source_sha256"),
            }
            for system_key, source in (packet.get("marker_sources") or {}).items()
        ],
        "error": packet.get("error"),
    }
    path = settings.trace_path
    try:
        with _TRACE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n")
            os.chmod(path, 0o600)
    except OSError:
        # Retrieval must never break the clinical report because trace storage failed.
        return


def retrieve_report_evidence(
    context: Dict[str, Any],
    *,
    settings: Optional[RagClientSettings] = None,
    transport: Optional[Callable[[str, Dict[str, Any], float], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    correlation_id = _context_correlation_id(context)
    try:
        cfg = settings or RagClientSettings.from_env()
    except Exception as exc:  # noqa: BLE001
        return _packet(
            "off",
            "invalid_config",
            correlation_id=correlation_id,
            error=f"{type(exc).__name__}: {exc}",
        )
    if cfg.mode == "off":
        return _packet("off", "disabled", correlation_id=correlation_id)
    if cfg.mode == "assist" and not cfg.assist_approved:
        packet = _packet(
            "assist", "assist_not_approved", correlation_id=correlation_id
        )
        _append_trace(packet, cfg)
        return packet

    queries = build_report_queries(context)
    include_demo = (
        cfg.mode == "shadow" and cfg.shadow_include_demo
    ) or (
        cfg.mode == "assist" and cfg.allow_demo_in_prompt
    )
    request_payload = {
        "queries": queries,
        "top_k": cfg.top_k_per_query,
        "include_demo": include_demo,
    }
    started = time.perf_counter()
    try:
        request = transport or _post_json
        response = request(
            f"{cfg.service_url}/v1/retrieve",
            request_payload,
            cfg.timeout_seconds,
        )
        if response.get("schema_version") != "rehab.rag.retrieve.v1":
            raise RuntimeError("unsupported RAG response schema")

        sources: List[Dict[str, Any]] = []
        seen_chunks = set()
        remaining_chars = cfg.max_context_chars
        for query_result in response.get("results", []) or []:
            for hit in query_result.get("hits", []) or []:
                chunk_id = str(hit.get("chunk_id") or "")
                metadata = dict(hit.get("metadata") or {})
                clinical_ready = bool(metadata.get("clinical_ready"))
                if not chunk_id or chunk_id in seen_chunks:
                    continue
                if cfg.mode == "assist" and not cfg.allow_demo_in_prompt and not clinical_ready:
                    continue
                text = str(hit.get("text") or "").strip()
                if not text or remaining_chars <= 0:
                    continue
                text = text[:remaining_chars]
                remaining_chars -= len(text)
                seen_chunks.add(chunk_id)
                sources.append(_source_from_hit(hit, text=text))
                if len(sources) >= cfg.max_sources:
                    break
            if len(sources) >= cfg.max_sources:
                break

        marker_keys = build_marker_system_keys(context)
        marker_sources: Dict[str, Dict[str, Any]] = {}
        if marker_keys:
            lookup_response = request(
                f"{cfg.service_url}/v1/lookup",
                {"system_keys": marker_keys, "include_demo": include_demo},
                cfg.timeout_seconds,
            )
            if lookup_response.get("schema_version") != "rehab.rag.lookup.v1":
                raise RuntimeError("unsupported RAG lookup response schema")
            for result in lookup_response.get("results", []) or []:
                system_key = str(result.get("system_key") or "")
                hit = result.get("hit")
                if system_key not in marker_keys or not isinstance(hit, dict):
                    continue
                text = str(hit.get("text") or "").strip()
                source = _source_from_hit(hit, text=text)
                if not source["knowledge_id"] or source["system_key"] != system_key:
                    continue
                if (
                    cfg.mode == "assist"
                    and not cfg.allow_demo_in_prompt
                    and not source["clinical_ready"]
                ):
                    continue
                marker_sources[system_key] = source

        used_in_prompt = cfg.mode == "assist" and bool(sources)
        marker_grounding_used = cfg.mode == "assist" and bool(marker_sources)
        marker_grounding_complete = (
            marker_grounding_used
            and bool(marker_keys)
            and set(marker_sources) == set(marker_keys)
        )
        packet = _packet(
            cfg.mode,
            "retrieved" if sources or marker_sources else "no_eligible_evidence",
            correlation_id=correlation_id,
            used_in_prompt=used_in_prompt,
            used_in_report=used_in_prompt or marker_grounding_used,
            marker_grounding_used=marker_grounding_used,
            marker_grounding_complete=marker_grounding_complete,
            collection=str(response.get("collection") or ""),
            queries=queries,
            sources=sources,
            marker_sources=marker_sources,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
    except Exception as exc:  # noqa: BLE001
        packet = _packet(
            cfg.mode,
            "service_unavailable",
            correlation_id=correlation_id,
            queries=queries,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            error=f"{type(exc).__name__}: {str(exc)[:300]}",
        )
    _append_trace(packet, cfg)
    return packet


def augment_report_context(
    context: Dict[str, Any],
    **kwargs: Any,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    packet = retrieve_report_evidence(context, **kwargs)
    augmented = dict(context)
    augmented["rag_evidence"] = packet
    return augmented, packet


__all__ = [
    "RagClientSettings",
    "augment_report_context",
    "build_marker_system_keys",
    "build_report_queries",
    "retrieve_report_evidence",
]
