"""Isolated research-evidence retrieval service (HTTP proxy mode).

This module provides the backend for the knowledge and research evidence
retrieval endpoints.
It proxies requests to the independent RAG service running on port 8011
(RAG_GUIDELINE_TEST_SERVICE_URL), which owns the BGE-M3 embedder and Qdrant
vector store.

Key design decisions:
- Scope guard: screen_guideline_test_query is called BEFORE any upstream HTTP
  request, so out-of-scope queries are rejected cheaply without network overhead.
- Timeout/unreachable: upstream errors never leak exception trace, file paths,
  or environment variables to the browser.
- Feature gate: RAG_GUIDELINE_TEST_ENABLED must be "true" to enable.
- All upstream communication uses httpx async client with configurable timeout.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


RAG_GUIDELINE_TEST_ENABLED: bool = _env_bool("RAG_GUIDELINE_TEST_ENABLED", False)
RAG_GUIDELINE_TEST_SERVICE_URL: str = os.getenv(
    "RAG_GUIDELINE_TEST_SERVICE_URL", "http://127.0.0.1:8011"
).strip()
RAG_SERVICE_TIMEOUT: float = float(os.getenv("RAG_SERVICE_TIMEOUT", "30"))

TEST_COLLECTION: str = os.getenv(
    "RAG_GUIDELINE_TEST_COLLECTION", "rehab_knowledge_trial_v0_3"
).strip()

TEST_REPORT_BANNER = (
    "研究证据检索结果仅用于知识与证据展示，不构成临床诊断、治疗处方或医疗建议。"
)

# Safe error messages — never expose internals
_UPSTREAM_UNAVAILABLE = "知识库检索服务暂时不可用，请稍后再试。"
_UPSTREAM_TIMEOUT = "知识库检索服务响应超时，请稍后再试。"
_UPSTREAM_ERROR = "知识库检索服务返回异常，请稍后再试。"
_UPSTREAM_STRUCTURE_ERROR = "知识库检索服务返回数据格式异常，请稍后再试。"

_URL_RE = re.compile(r"https?://\S+")
_SOURCE_ID_RE = re.compile(r"^\[([A-Za-z0-9_-]+)\]")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuidelineTestStatus:
    enabled: bool
    service_reachable: bool
    collection: str
    clinical_ready: bool
    allow_demo: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "mode": "test_only",
            "allowed_rag_mode": "test_only",
            "enabled": self.enabled,
            "service_reachable": self.service_reachable,
            "collection": self.collection,
            "clinical_ready": self.clinical_ready,
            "allow_demo": self.allow_demo,
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass(frozen=True)
class GuidelineSearchHit:
    rank: int
    score: float
    source_id: str
    title: str
    year: str
    doi: str
    page_locator: str
    text: str
    citation_index: int
    citation_indices: List[int]
    chunk_id: str
    references: List[Dict[str, Any]]
    source_type: str
    knowledge_type: str
    evidence_scope: str
    research_type: str
    sample_size: str
    applicable_scope: str
    limitations: List[str]
    license: str
    non_clinical_statement: str
    research_only: bool
    expert_verified: bool
    source_detail: Dict[str, Any]


@dataclass(frozen=True)
class GuidelineSearchResponse:
    schema_version: str
    mode: str
    test_report_banner: str
    query: str
    top_k: int
    dataset: str
    clinical_ready: bool
    hits: List[GuidelineSearchHit]
    cached: bool
    elapsed_ms: int
    citations: List[Dict[str, Any]]
    reason_code: str = "in_scope"
    blocked_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "rehab.rag.guideline-test-api.v1",
            "mode": "test_only",
            "allowed_rag_mode": "test_only",
            "test_report_banner": self.test_report_banner,
            "query": self.query,
            "top_k": self.top_k,
            "dataset": self.dataset,
            "clinical_ready": self.clinical_ready,
            "results": [self._hit_dict(h) for h in self.hits],
            "cached": self.cached,
            "elapsed_ms": self.elapsed_ms,
            "citations": self.citations,
            "reason_code": self.reason_code,
            **({"blocked_message": self.blocked_message} if self.blocked_message else {}),
        }

    @staticmethod
    def _hit_dict(h: GuidelineSearchHit) -> Dict[str, Any]:
        return {
            "rank": h.rank,
            "score": round(h.score, 4),
            "source_id": h.source_id,
            "title": h.title,
            "year": h.year,
            "doi": h.doi,
            "page_locator": h.page_locator,
            "text": h.text,
            "citation_index": h.citation_index,
            "citation_indices": h.citation_indices,
            "chunk_id": h.chunk_id,
            "references": h.references,
            "source_type": h.source_type,
            "knowledge_type": h.knowledge_type,
            "evidence_scope": h.evidence_scope,
            "research_type": h.research_type,
            "sample_size": h.sample_size,
            "applicable_scope": h.applicable_scope,
            "limitations": h.limitations,
            "license": h.license,
            "non_clinical_statement": h.non_clinical_statement,
            "research_only": h.research_only,
            "expert_verified": h.expert_verified,
            "source_detail": h.source_detail,
        }


# ---------------------------------------------------------------------------
# Reference normalization
# ---------------------------------------------------------------------------


def _extract_source_id(raw: str) -> str:
    m = _SOURCE_ID_RE.match(raw)
    return m.group(1) if m else ""


def _normalize_reference(ref: Any) -> Dict[str, Any]:
    """Normalize a single reference to a structured dict.

    Handles:
    - Dict references: passed through with original fields.
    - String references: "[SRC-006] NICE NG236: Stroke rehabilitation... https://..."
      → source_id extracted, title cleaned, URL stored in doi, raw_text preserved.
    """
    if isinstance(ref, dict):
        normalized = dict(ref)
        for field in ("source_id", "title", "year", "doi", "page_locator", "raw_text"):
            value = normalized.get(field)
            if value is None:
                normalized[field] = ""
            elif not isinstance(value, str):
                raise ValueError(f"reference field {field} must be a string")
        return normalized
    if not isinstance(ref, str):
        raise ValueError("reference must be a string or object")
    raw = ref.strip()
    if not raw:
        return {"source_id": "", "title": "", "raw_text": ""}
    source_id = _extract_source_id(raw)
    urls = _URL_RE.findall(raw)
    # Remove [SOURCE-ID] prefix and trailing URLs for clean title
    title_part = raw
    if source_id:
        title_part = _SOURCE_ID_RE.sub("", title_part, count=1).strip()
    if urls:
        for u in urls:
            title_part = title_part.replace(u, "").strip()
    title_part = title_part.strip(" :").strip()
    result: Dict[str, Any] = {
        "source_id": source_id,
        "title": title_part,
        "raw_text": raw,
    }
    if urls:
        result["doi"] = urls[0]
    return result


def _normalize_references(refs: Any) -> List[Dict[str, Any]]:
    if refs is None or refs == [] or refs == "":
        return []
    if isinstance(refs, str):
        return [_normalize_reference(refs)]
    if isinstance(refs, dict):
        return [_normalize_reference(refs)]
    if isinstance(refs, list):
        return [_normalize_reference(r) for r in refs]
    raise ValueError("references must be a list, string, or object")


def _citation_dedup_key(ref: Dict[str, Any]) -> str:
    """Build a stable key without relying on a possibly empty source_id."""
    parts = [
        ref.get("source_id", ""),
        ref.get("raw_text", "") or ref.get("title", ""),
        ref.get("year", ""),
        ref.get("doi", ""),
    ]
    return "|".join(parts)


def _validate_score(score: Any) -> float:
    """Validate and convert score to finite float; reject bool/NaN/Inf."""
    if isinstance(score, bool):
        raise ValueError("score must not be a boolean")
    if isinstance(score, (int, float)):
        if math.isnan(score) or math.isinf(score):
            raise ValueError("score must be a finite number")
        return float(score)
    raise ValueError("score must be a finite number")


def _optional_string(value: Any, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _optional_string_list(value: Any, field: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return value


def _optional_bool(value: Any, field: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _source_detail(
    metadata: Dict[str, Any], reference: Dict[str, Any]
) -> Dict[str, Any]:
    """Return a safe, in-system evidence card for a retrieved source.

    The card stores a local evidence extract, not a source full text. Full texts
    are only appropriate when they are open access or separately licensed.
    """
    source_url = _optional_string(metadata.get("reference_url"), "metadata.reference_url")
    if not source_url:
        doi = _optional_string(reference.get("doi"), "reference.doi")
        source_url = doi if doi.startswith("https://") else (
            f"https://doi.org/{doi}" if doi.startswith("10.") else ""
        )
    if source_url and not source_url.startswith("https://"):
        source_url = ""
    raw_weight = metadata.get("authority_weight")
    if raw_weight is None:
        authority_weight = ""
    elif isinstance(raw_weight, bool) or not isinstance(raw_weight, (str, int, float)):
        raise ValueError("metadata.authority_weight must be a string or number")
    else:
        authority_weight = str(raw_weight)
    return {
        "source_type": _optional_string(metadata.get("source_type"), "metadata.source_type"),
        "evidence_tier": _optional_string(metadata.get("evidence_tier"), "metadata.evidence_tier"),
        "authority_weight": authority_weight,
        "source_url": source_url,
        "access_status": _optional_string(
            metadata.get("access_status"), "metadata.access_status"
        ) or "系统保存证据摘录；原始来源可用性需以访问时状态为准。",
        "rights_status": _optional_string(
            metadata.get("rights_status"), "metadata.rights_status"
        ) or "未保存受版权保护全文。",
        "local_excerpt": _optional_string(
            metadata.get("local_evidence_excerpt"), "metadata.local_evidence_excerpt"
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_status() -> GuidelineTestStatus:
    """Return status by probing the RAG service health endpoint.

    Validates: status/loaded/enabled, allow_demo, collection match.
    """
    if not RAG_GUIDELINE_TEST_ENABLED:
        return GuidelineTestStatus(
            enabled=False,
            service_reachable=False,
            collection=TEST_COLLECTION,
            clinical_ready=False,
        )

    try:
        async with httpx.AsyncClient(timeout=RAG_SERVICE_TIMEOUT) as client:
            resp = await client.get(f"{RAG_GUIDELINE_TEST_SERVICE_URL}/health")
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception:
                    return GuidelineTestStatus(
                        enabled=True,
                        service_reachable=False,
                        collection=TEST_COLLECTION,
                        clinical_ready=False,
                        error="知识库检索服务健康数据格式异常",
                    )
                if not isinstance(data, dict):
                    return GuidelineTestStatus(
                        enabled=True,
                        service_reachable=False,
                        collection=TEST_COLLECTION,
                        clinical_ready=False,
                        error="知识库检索服务健康数据格式异常",
                    )
                status = data.get("status")
                loaded = data.get("loaded")
                enabled = data.get("enabled")
                allow_demo = data.get("allow_demo")
                upstream_collection = data.get("collection", "")
                if status != "ok" or loaded is not True or enabled is not True:
                    return GuidelineTestStatus(
                        enabled=True,
                        service_reachable=False,
                        collection=TEST_COLLECTION,
                        clinical_ready=False,
                        allow_demo=allow_demo is True,
                        error="知识库检索服务未就绪",
                    )
                if upstream_collection != TEST_COLLECTION:
                    return GuidelineTestStatus(
                        enabled=True,
                        service_reachable=False,
                        collection=TEST_COLLECTION,
                        clinical_ready=False,
                        allow_demo=allow_demo is True,
                        error="知识库检索服务集合配置不匹配",
                    )
                if allow_demo is not True:
                    return GuidelineTestStatus(
                        enabled=True,
                        service_reachable=False,
                        collection=TEST_COLLECTION,
                        clinical_ready=False,
                        allow_demo=False,
                        error="知识库检索服务未开放演示模式",
                    )
                return GuidelineTestStatus(
                    enabled=True,
                    service_reachable=True,
                    collection=upstream_collection,
                    clinical_ready=False,
                    allow_demo=True,
                )
            return GuidelineTestStatus(
                enabled=True,
                service_reachable=False,
                collection=TEST_COLLECTION,
                clinical_ready=False,
                error="知识库检索服务响应异常",
            )
    except httpx.TimeoutException:
        return GuidelineTestStatus(
            enabled=True,
            service_reachable=False,
            collection=TEST_COLLECTION,
            clinical_ready=False,
            error="知识库检索服务连接超时",
        )
    except httpx.ConnectError:
        return GuidelineTestStatus(
            enabled=True,
            service_reachable=False,
            collection=TEST_COLLECTION,
            clinical_ready=False,
            error="知识库检索服务不可达",
        )
    except Exception:
        return GuidelineTestStatus(
            enabled=True,
            service_reachable=False,
            collection=TEST_COLLECTION,
            clinical_ready=False,
            error=_UPSTREAM_ERROR,
        )


def _build_citations(hits: List[GuidelineSearchHit]) -> List[Dict[str, Any]]:
    """Build globally deduplicated citation list from all hits' references."""
    seen_keys: set = set()
    citations: List[Dict[str, Any]] = []
    global_idx = 0
    for hit in hits:
        for ref in hit.references:
            key = _citation_dedup_key(ref)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            global_idx += 1
            citations.append({
                "index": global_idx,
                "source_id": ref.get("source_id", ""),
                "title": ref.get("title", ""),
                "year": ref.get("year", ""),
                "doi": ref.get("doi", ""),
                "page_locator": ref.get("page_locator", ""),
                "raw_text": ref.get("raw_text", ""),
            })
    return citations


def _assign_hit_citations(
    hits: List[GuidelineSearchHit], citations: List[Dict[str, Any]]
) -> List[GuidelineSearchHit]:
    """Assign each hit all globally deduplicated citation indices."""
    # Build lookup: dedup_key → global citation index
    lookup: Dict[str, int] = {}
    for c in citations:
        key = _citation_dedup_key(c)
        lookup[key] = c["index"]

    updated: List[GuidelineSearchHit] = []
    for hit in hits:
        citation_indices: List[int] = []
        for ref in hit.references:
            global_idx = lookup.get(_citation_dedup_key(ref))
            if global_idx is not None and global_idx not in citation_indices:
                citation_indices.append(global_idx)
        primary_index = citation_indices[0] if citation_indices else hit.citation_index
        updated.append(GuidelineSearchHit(
            rank=hit.rank,
            score=hit.score,
            source_id=hit.source_id,
            title=hit.title,
            year=hit.year,
            doi=hit.doi,
            page_locator=hit.page_locator,
            text=hit.text,
            citation_index=primary_index,
            citation_indices=citation_indices,
            chunk_id=hit.chunk_id,
            references=hit.references,
            source_type=hit.source_type,
            knowledge_type=hit.knowledge_type,
            evidence_scope=hit.evidence_scope,
            research_type=hit.research_type,
            sample_size=hit.sample_size,
            applicable_scope=hit.applicable_scope,
            limitations=hit.limitations,
            license=hit.license,
            non_clinical_statement=hit.non_clinical_statement,
            research_only=hit.research_only,
            expert_verified=hit.expert_verified,
            source_detail=hit.source_detail,
        ))
    return updated


async def search_guidelines(
    query: str,
    top_k: int = 3,
) -> GuidelineSearchResponse:
    """Run a guarded search against the research-evidence collection.

    Scope check happens BEFORE any upstream HTTP request.
    Upstream errors raise HTTPException(503) with safe Chinese message.
    """
    from rag.guideline_policy import screen_guideline_test_query

    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query must not be empty")
    if len(clean_query) > 2000:
        raise ValueError("query must not exceed 2000 characters")
    if top_k < 1 or top_k > 5:
        raise ValueError("top_k must be between 1 and 5")

    if not RAG_GUIDELINE_TEST_ENABLED:
        raise RuntimeError("rag_guideline_test_disabled")

    decision = screen_guideline_test_query(clean_query)
    if decision.action == "block":
        return GuidelineSearchResponse(
            schema_version="rehab.rag.guideline-test-api.v1",
            mode="test_only",
            test_report_banner=TEST_REPORT_BANNER,
            query=clean_query,
            top_k=top_k,
            dataset=TEST_COLLECTION,
            clinical_ready=False,
            hits=[],
            cached=False,
            elapsed_ms=0,
            citations=[],
            reason_code=decision.reason_code,
            blocked_message=decision.message,
        )

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=RAG_SERVICE_TIMEOUT) as client:
            resp = await client.post(
                f"{RAG_GUIDELINE_TEST_SERVICE_URL}/v1/retrieve",
                json={
                    "queries": [{"key": "guideline_test", "text": clean_query}],
                    "top_k": top_k,
                    "include_demo": True,
                },
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=503, detail=_UPSTREAM_TIMEOUT)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=_UPSTREAM_UNAVAILABLE)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail=_UPSTREAM_ERROR)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if resp.status_code != 200:
        raise HTTPException(status_code=503, detail=_UPSTREAM_ERROR)

    try:
        data = resp.json()
    except Exception:
        raise HTTPException(status_code=503, detail=_UPSTREAM_ERROR)

    # --- strict root structure validation ---
    if not isinstance(data, dict):
        raise HTTPException(status_code=503, detail=_UPSTREAM_STRUCTURE_ERROR)

    upstream_collection = data.get("collection", "")
    if not isinstance(upstream_collection, str) or not upstream_collection:
        raise HTTPException(status_code=503, detail=_UPSTREAM_STRUCTURE_ERROR)
    if upstream_collection != TEST_COLLECTION:
        raise HTTPException(status_code=503, detail=_UPSTREAM_STRUCTURE_ERROR)
    if data.get("demo_evidence_included") is not True:
        raise HTTPException(status_code=503, detail=_UPSTREAM_STRUCTURE_ERROR)

    results_list = data.get("results")
    if not isinstance(results_list, list) or not results_list:
        raise HTTPException(status_code=503, detail=_UPSTREAM_STRUCTURE_ERROR)

    first_result = results_list[0]
    if not isinstance(first_result, dict):
        raise HTTPException(status_code=503, detail=_UPSTREAM_STRUCTURE_ERROR)

    batch_key = first_result.get("key", "")
    if batch_key != "guideline_test":
        raise HTTPException(status_code=503, detail=_UPSTREAM_STRUCTURE_ERROR)

    upstream_hits = first_result.get("hits")
    if not isinstance(upstream_hits, list):
        raise HTTPException(status_code=503, detail=_UPSTREAM_STRUCTURE_ERROR)

    # --- build hits with strict per-hit validation ---
    result_hits: List[GuidelineSearchHit] = []
    for i, h in enumerate(upstream_hits, start=1):
        if not isinstance(h, dict):
            raise HTTPException(status_code=503, detail=_UPSTREAM_STRUCTURE_ERROR)

        metadata = h.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise HTTPException(status_code=503, detail=_UPSTREAM_STRUCTURE_ERROR)

        try:
            validated_score = _validate_score(h["score"]) if "score" in h else 0.0
        except ValueError:
            raise HTTPException(status_code=503, detail=_UPSTREAM_STRUCTURE_ERROR)

        refs_raw = (metadata or {}).get("references", []) if metadata else h.get("references", [])
        try:
            refs = _normalize_references(refs_raw)
            knowledge_id = _optional_string(h.get("knowledge_id"), "knowledge_id")
            chunk_id = _optional_string(h.get("chunk_id"), "chunk_id")
            text = _optional_string(h.get("text"), "text")
            hit_title = _optional_string(h.get("title"), "title")
            metadata_title = _optional_string(
                (metadata or {}).get("title"), "metadata.title"
            )
            source_type = _optional_string(
                (metadata or {}).get("source_type"), "metadata.source_type"
            )
            knowledge_type = _optional_string(
                (metadata or {}).get("knowledge_type"), "metadata.knowledge_type"
            )
            evidence_scope = _optional_string(
                (metadata or {}).get("evidence_scope"), "metadata.evidence_scope"
            )
            research_type = _optional_string(
                (metadata or {}).get("research_type"), "metadata.research_type"
            )
            sample_size = _optional_string(
                (metadata or {}).get("sample_size"), "metadata.sample_size"
            )
            applicable_scope = _optional_string(
                (metadata or {}).get("applicable_scope"), "metadata.applicable_scope"
            )
            limitations = _optional_string_list(
                (metadata or {}).get("limitations"), "metadata.limitations"
            )
            license_name = _optional_string(
                (metadata or {}).get("license"), "metadata.license"
            )
            non_clinical_statement = _optional_string(
                (metadata or {}).get("non_clinical_statement"),
                "metadata.non_clinical_statement",
            )
            research_only = _optional_bool(
                (metadata or {}).get("research_only"), "metadata.research_only"
            )
            expert_verified = _optional_bool(
                (metadata or {}).get("expert_verified"), "metadata.expert_verified"
            )
        except ValueError:
            raise HTTPException(status_code=503, detail=_UPSTREAM_STRUCTURE_ERROR)
        first_ref = refs[0] if refs else {}

        # Title priority: hit title → metadata title → first ref title (not whole ref)
        if not hit_title:
            hit_title = metadata_title
        if not hit_title:
            hit_title = first_ref.get("title", "")

        source_detail = _source_detail(metadata or {}, first_ref)
        result_hits.append(GuidelineSearchHit(
            rank=i,
            score=validated_score,
            source_id=first_ref.get("source_id", "") or knowledge_id,
            title=hit_title,
            year=first_ref.get("year", ""),
            doi=first_ref.get("doi", ""),
            page_locator=first_ref.get("page_locator", ""),
            text=text,
            citation_index=i,
            citation_indices=[],
            chunk_id=chunk_id,
            references=refs,
            source_type=source_type or source_detail["source_type"],
            knowledge_type=knowledge_type,
            evidence_scope=evidence_scope,
            research_type=research_type,
            sample_size=sample_size,
            applicable_scope=applicable_scope,
            limitations=limitations,
            license=license_name,
            non_clinical_statement=non_clinical_statement,
            research_only=research_only,
            expert_verified=expert_verified,
            source_detail=source_detail,
        ))

    citations = _build_citations(result_hits)
    result_hits = _assign_hit_citations(result_hits, citations)

    return GuidelineSearchResponse(
        schema_version="rehab.rag.guideline-test-api.v1",
        mode="test_only",
        test_report_banner=TEST_REPORT_BANNER,
        query=clean_query,
        top_k=top_k,
        dataset=upstream_collection,
        clinical_ready=False,
        hits=result_hits,
        cached=data.get("cached") is True,
        elapsed_ms=elapsed_ms,
        citations=citations,
    )


__all__ = [
    "RAG_GUIDELINE_TEST_ENABLED",
    "RAG_GUIDELINE_TEST_SERVICE_URL",
    "TEST_COLLECTION",
    "get_status",
    "search_guidelines",
]
