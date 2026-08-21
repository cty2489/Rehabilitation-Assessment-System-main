"""Gold Reference validation and deterministic reference manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonicalize import PLAN_FIELDS
from .normalization import normalize_medical_text

REFERENCE_SCHEMA_VERSION = "rehab.llm-benchmark-reference.v1"
REQUIRED_SECTIONS = (
    "biomarker_interpretation",
    "integrated_assessment",
    "rehabilitation_plan",
)


@dataclass(frozen=True)
class GoldReference:
    schema_version: str
    reference_id: str
    patient_id: str
    reference_version: str
    review_status: str
    sections: dict[str, Any]
    audit: dict[str, Any]
    source_path: str | None = None


@dataclass(frozen=True)
class ReferenceValidationResult:
    valid: bool
    reference: GoldReference | None
    errors: tuple[str, ...]


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(normalize_medical_text(value))


def validate_reference_payload(payload: Any, *, source_path: str | None = None) -> ReferenceValidationResult:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ReferenceValidationResult(False, None, ("reference_not_object",))
    for field in ("schema_version", "reference_id", "patient_id", "reference_version", "review_status"):
        if not _nonempty_string(payload.get(field)):
            errors.append(f"missing_or_empty:{field}")
    if payload.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if payload.get("review_status") != "approved":
        errors.append("review_status_not_approved")

    sections = payload.get("sections")
    if not isinstance(sections, Mapping):
        errors.append("sections_not_object")
        sections = {}
    for section in REQUIRED_SECTIONS[:2]:
        if not _nonempty_string(sections.get(section)):
            errors.append(f"missing_or_empty_section:{section}")
    plan = sections.get("rehabilitation_plan")
    if not isinstance(plan, list) or not plan:
        errors.append("missing_or_empty_section:rehabilitation_plan")
    elif not all(isinstance(item, Mapping) for item in plan):
        errors.append("rehabilitation_plan_item_not_object")
    else:
        for position, item in enumerate(plan, start=1):
            for field in PLAN_FIELDS:
                if not _nonempty_string(item.get(field)):
                    errors.append(f"missing_or_empty_plan_field:{position}:{field}")

    audit = payload.get("audit", {})
    if not isinstance(audit, Mapping):
        errors.append("audit_not_object")
        audit = {}
    if errors:
        return ReferenceValidationResult(False, None, tuple(errors))
    reference = GoldReference(
        schema_version=str(payload["schema_version"]),
        reference_id=str(payload["reference_id"]),
        patient_id=str(payload["patient_id"]),
        reference_version=str(payload["reference_version"]),
        review_status=str(payload["review_status"]),
        sections={key: value for key, value in sections.items()},
        audit={key: value for key, value in audit.items()},
        source_path=source_path,
    )
    return ReferenceValidationResult(True, reference, ())


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # pragma: no cover - exact parser errors vary
        return None, f"reference_json_read_failed:{type(exc).__name__}:{exc}"


def load_reference_set(root: str | Path) -> tuple[dict[str, GoldReference], list[dict[str, Any]], str]:
    """Load all JSON references and return valid references, errors, and hash."""

    root_path = Path(root)
    paths = sorted(root_path.rglob("*.json")) if root_path.exists() else []
    references: dict[str, GoldReference] = {}
    errors: list[dict[str, Any]] = []
    manifest_items: list[dict[str, str]] = []
    if not paths:
        errors.append({
            "stage": "reference",
            "source_path": str(root_path),
            "error_code": "REFERENCE_EMPTY",
            "error_message": "reference root contains no JSON files",
        })
    for path in paths:
        relative = str(path.relative_to(root_path))
        data, read_error = _read_json(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_items.append({"path": relative, "sha256": digest})
        if read_error:
            errors.append({"stage": "reference", "source_path": str(path), "error_code": "REFERENCE_INVALID", "error_message": read_error})
            continue
        result = validate_reference_payload(data, source_path=str(path))
        if not result.valid or result.reference is None:
            errors.append({"stage": "reference", "source_path": str(path), "error_code": "REFERENCE_INVALID", "error_message": ";".join(result.errors)})
            continue
        reference = result.reference
        if reference.patient_id in references:
            errors.append({"stage": "reference", "source_path": str(path), "error_code": "REFERENCE_DUPLICATE_PATIENT", "patient_id": reference.patient_id, "error_message": "duplicate patient_id"})
            references.pop(reference.patient_id, None)
            continue
        references[reference.patient_id] = reference
    manifest_hash = hashlib.sha256(
        json.dumps(manifest_items, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return references, errors, manifest_hash


__all__ = [
    "GoldReference",
    "REFERENCE_SCHEMA_VERSION",
    "REQUIRED_SECTIONS",
    "ReferenceValidationResult",
    "load_reference_set",
    "validate_reference_payload",
]
