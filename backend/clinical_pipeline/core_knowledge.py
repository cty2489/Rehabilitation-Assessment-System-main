"""Read-only adapter for the existing governed core-knowledge release."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence
from uuid import uuid4

from pydantic import ValidationError

from knowledge_admin import KnowledgeSnapshot, load_snapshot

from .contracts import CoreKnowledgeBundle, CoreKnowledgeEntry


SnapshotLoader = Callable[[], KnowledgeSnapshot]


class CoreKnowledgeLoadError(RuntimeError):
    """Raised when required fixed knowledge cannot be loaded exactly."""


def _normalize_system_keys(system_keys: Sequence[str]) -> list[str]:
    if isinstance(system_keys, (str, bytes)):
        raise CoreKnowledgeLoadError("system_keys必须是键列表，不能是单个字符串")
    normalized: list[str] = []
    for value in system_keys:
        key = str(value).strip()
        if key and key not in normalized:
            normalized.append(key)
    if not normalized:
        raise CoreKnowledgeLoadError("至少需要一个system_key")
    return normalized


def _contract_entry(raw: Dict[str, Any], system_key: str) -> CoreKnowledgeEntry:
    source = raw.get("source") or {}
    if not isinstance(source, dict):
        raise CoreKnowledgeLoadError(
            f"固定核心知识格式错误：{system_key}.source必须是对象"
        )
    raw_source_ids = source.get("source_ids") or []
    if not isinstance(raw_source_ids, list):
        raise CoreKnowledgeLoadError(
            f"固定核心知识格式错误：{system_key}.source_ids必须是列表"
        )
    source_ids = [str(value).strip() for value in raw_source_ids]
    if any(not value for value in source_ids):
        raise CoreKnowledgeLoadError(
            f"固定核心知识格式错误：{system_key}.source_ids包含空值"
        )

    prohibited = str(raw.get("prohibited_interpretation") or "").strip()
    try:
        return CoreKnowledgeEntry(
            knowledge_id=str(raw.get("knowledge_id") or "").strip(),
            system_key=system_key,
            allowed_interpretation=str(
                raw.get("allowed_interpretation") or ""
            ).strip(),
            prohibited_interpretation=prohibited or None,
            source_ids=source_ids,
        )
    except ValidationError as exc:
        raise CoreKnowledgeLoadError(
            f"固定核心知识格式错误：{system_key}"
        ) from exc


class CoreKnowledgeProvider:
    """Select fixed knowledge by exact ``system_key`` from the active snapshot."""

    def __init__(self, snapshot_loader: Optional[SnapshotLoader] = None):
        self._snapshot_loader = snapshot_loader or load_snapshot

    def provide(self, system_keys: Sequence[str]) -> CoreKnowledgeBundle:
        requested = _normalize_system_keys(system_keys)
        try:
            snapshot = self._snapshot_loader()
        except Exception as exc:  # noqa: BLE001 - provider boundary adds context
            raise CoreKnowledgeLoadError(
                f"固定核心知识加载失败：{exc}"
            ) from exc
        if not isinstance(snapshot, KnowledgeSnapshot):
            raise CoreKnowledgeLoadError("固定核心知识加载失败：快照类型错误")

        by_key: Dict[str, Dict[str, Any]] = {}
        for raw in snapshot.entries:
            if not isinstance(raw, dict):
                raise CoreKnowledgeLoadError("固定核心知识格式错误：条目必须是对象")
            system_key = str(raw.get("system_key") or "").strip()
            if not system_key:
                raise CoreKnowledgeLoadError("固定核心知识格式错误：条目缺少system_key")
            if system_key in by_key:
                raise CoreKnowledgeLoadError(
                    f"固定核心知识格式错误：system_key重复：{system_key}"
                )
            by_key[system_key] = raw

        missing = [key for key in requested if key not in by_key]
        if missing:
            raise CoreKnowledgeLoadError(
                "固定核心知识缺少system_key：" + "、".join(missing)
            )

        if not isinstance(snapshot.manifest, dict):
            raise CoreKnowledgeLoadError("固定核心知识加载失败：发布清单格式错误")
        release = snapshot.manifest.get("trial_release") or {}
        if not isinstance(release, dict):
            raise CoreKnowledgeLoadError("固定核心知识加载失败：发布版本格式错误")
        version = str(
            release.get("release_id") or snapshot.manifest.get("collection_id") or ""
        ).strip()
        if not version:
            raise CoreKnowledgeLoadError("固定核心知识加载失败：发布版本缺失")

        try:
            return CoreKnowledgeBundle(
                bundle_id=f"core-{uuid4().hex}",
                version=version,
                entries=[_contract_entry(by_key[key], key) for key in requested],
            )
        except ValidationError as exc:
            raise CoreKnowledgeLoadError("固定核心知识加载失败：输出契约错误") from exc


__all__ = [
    "CoreKnowledgeLoadError",
    "CoreKnowledgeProvider",
    "SnapshotLoader",
]
