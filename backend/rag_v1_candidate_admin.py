"""Admin-only bridge for the isolated v1 candidate RAG release.

This module is intentionally outside the clinical pipeline. Version selection
is limited to admin evidence testing and never rewrites the production RAG
environment or report-generation configuration.
"""
from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(os.getenv("REHAB_PROJECT_ROOT", "/root/autodl-tmp/rehab_project"))
CANDIDATE_ROOT = Path(
    os.getenv(
        "RAG_V1_CANDIDATE_RELEASE",
        str(PROJECT_ROOT / "knowledge_base/v1_candidate_next"),
    )
).resolve()
VERSION_CONFIG = PROJECT_ROOT / "knowledge_base/rag_version_config.json"
SELECTION_FILE = PROJECT_ROOT / "rag_v1_admin_selection.json"
OLD_SERVICE = "http://127.0.0.1:8011"
CANDIDATE_SERVICE = "http://127.0.0.1:8013"

router = APIRouter()


def _require_admin_from_main(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    # Import lazily to avoid a main.py <-> router import cycle at startup.
    from main import _require_admin

    _require_admin(request, authorization)


class VersionSelectionRequest(BaseModel):
    rag_version: str = Field(min_length=1, max_length=64)


class RAGTestSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


def _config() -> dict[str, Any]:
    try:
        return json.loads(VERSION_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="RAG版本配置不可读") from exc


def _candidate_release_summary() -> dict[str, Any]:
    """Read the deployed candidate manifest for the admin UI."""
    try:
        manifest = json.loads((CANDIDATE_ROOT / "manifest.json").read_text(encoding="utf-8"))
        embeddings = json.loads((CANDIDATE_ROOT / "embedding_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="新版知识库发布清单不可读") from exc
    counts = manifest.get("counts") or {}
    return {
        "release": manifest.get("release") or "v1_candidate",
        "knowledge_base": manifest.get("knowledge_base") or "rehab_knowledge_v1",
        "sources": int(counts.get("sources") or 0),
        "chunks": int(counts.get("chunks") or 0),
        "embeddings": int(embeddings.get("count") or counts.get("embeddings") or 0),
        "qdrant_points": int(counts.get("qdrant_points") or 0),
        "collection": counts.get("collection") or embeddings.get("collection") or "",
        "source_files": int(counts.get("source_files") or 0),
        "embedding_dimensions": int(embeddings.get("dimensions") or 0),
        "distance": embeddings.get("distance") or "",
        "role_counts": counts.get("chunks_by_governance_role") or {},
        "textbook_ocr": manifest.get("textbook_ocr") or {},
        "failed_page_excluded": manifest.get("ingestion_policy", {}).get("failed_page_policy", {}),
    }


def _selected_version(config: dict[str, Any]) -> str:
    try:
        value = json.loads(SELECTION_FILE.read_text(encoding="utf-8")).get("rag_version")
    except (OSError, json.JSONDecodeError):
        value = config.get("rag_version", "v1_candidate")
    if value not in config.get("versions", {}):
        return str(config.get("rag_version") or "v1_candidate")
    return str(value)


def _service_for(version: str) -> str:
    return CANDIDATE_SERVICE if version == "v1_candidate" else OLD_SERVICE


def _enrich_source_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Expose stable citation fields alongside nested retrieval metadata."""
    for group in result.get("results", []):
        for hit in group.get("hits", []):
            metadata = hit.get("metadata") or {}
            if not hit.get("title"):
                hit["title"] = metadata.get("title") or metadata.get("book") or metadata.get("source_pdf") or ""
            if "uid" not in hit and metadata.get("uid"):
                hit["uid"] = metadata["uid"]
            if "page_number" not in hit:
                page = metadata.get("page_number") or metadata.get("page_start")
                if page is not None:
                    hit["page_number"] = page
            if "source_file_id" not in hit and metadata.get("source_file_id"):
                hit["source_file_id"] = metadata["source_file_id"]
            if "source_page_url" not in hit:
                route = metadata.get("source_page_url") or metadata.get("source_view_route")
                if route:
                    hit["source_page_url"] = route
            if "source_pdf_url" not in hit and metadata.get("source_pdf_url"):
                hit["source_pdf_url"] = metadata["source_pdf_url"]
    return result


def _source_record(source_file_id: str) -> dict[str, Any]:
    manifest = CANDIDATE_ROOT / "source_file_manifest.jsonl"
    try:
        rows = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HTTPException(status_code=503, detail="候选来源文件清单不可读") from exc
    for raw in rows:
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("source_file_id") == source_file_id:
            return row
    raise HTTPException(status_code=404, detail="来源文件不存在")


def _source_path(source_file_id: str) -> Path:
    row = _source_record(source_file_id)
    filename = Path(str(row.get("source_file") or "")).name
    candidate = (CANDIDATE_ROOT / "source_files" / filename).resolve()
    library = (CANDIDATE_ROOT / "source_files").resolve()
    if library not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="来源文件不可用")
    return candidate


@router.get("/api/admin/rag-versions", dependencies=[Depends(_require_admin_from_main)])
def get_rag_versions() -> dict[str, Any]:
    config = _config()
    selected = _selected_version(config)
    health: dict[str, Any] = {}
    for version in config.get("versions", {}):
        try:
            response = httpx.get(f"{_service_for(version)}/health", timeout=2.0)
            health[version] = response.json()
        except Exception as exc:  # noqa: BLE001 - admin status is diagnostic
            health[version] = {"status": "unavailable", "detail": str(exc)}
    return {
        "schema_version": "rehab.rag.admin-version-selection.v1",
        "selected": selected,
        "default": str(config.get("rag_version") or "v1_candidate"),
        "selection_scope": "active_rag",
        "production_report_rag_unchanged": False,
        "release": _candidate_release_summary(),
        "versions": config.get("versions", {}),
        "health": health,
    }


@router.post("/api/admin/rag-versions/select", dependencies=[Depends(_require_admin_from_main)])
def select_rag_version(body: VersionSelectionRequest) -> dict[str, Any]:
    config = _config()
    if body.rag_version not in config.get("versions", {}):
        raise HTTPException(status_code=422, detail="不支持的RAG版本")
    SELECTION_FILE.write_text(
        json.dumps(
            {"rag_version": body.rag_version, "scope": "admin_test_retrieval_only"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return get_rag_versions()


@router.post("/api/rag/v1-candidate/search", dependencies=[Depends(_require_admin_from_main)])
async def search_selected_rag(body: RAGTestSearchRequest) -> dict[str, Any]:
    config = _config()
    version = _selected_version(config)
    payload = {
        "queries": [{"key": "admin", "text": body.query}],
        "top_k": body.top_k,
        "include_demo": False,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{_service_for(version)}/v1/retrieve", json=payload)
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="RAG测试服务不可用") from exc
    result = _enrich_source_fields(result)
    result["selected_rag_version"] = version
    result["selection_scope"] = "admin_test_retrieval_only"
    return result


@router.get(
    "/api/rag/source-files/{source_file_id}/pages/{page_number}",
    dependencies=[Depends(_require_admin_from_main)],
)
def source_page(source_file_id: str, page_number: int) -> RedirectResponse:
    row = _source_record(source_file_id)
    page_count = int(row.get("page_count") or 0)
    if page_number < 1 or page_number > page_count:
        raise HTTPException(status_code=404, detail="PDF页码不存在")
    return RedirectResponse(
        f"/api/rag/source-files/{source_file_id}/pdf#page={page_number}",
        status_code=302,
    )


@router.get(
    "/api/rag/source-files/{source_file_id}/pdf",
    dependencies=[Depends(_require_admin_from_main)],
)
def source_pdf(source_file_id: str) -> FileResponse:
    path = _source_path(source_file_id)
    return FileResponse(path, media_type=mimetypes.guess_type(str(path))[0] or "application/octet-stream")


@router.get(
    "/api/rag/source-files/{source_file_id}",
    dependencies=[Depends(_require_admin_from_main)],
)
def source_file(source_file_id: str) -> FileResponse:
    """Serve a registered non-page source when an exact page is unavailable."""
    return source_pdf(source_file_id)


__all__ = ["router"]
