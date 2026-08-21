"""Gold Reference v2 validation; v1 references remain separate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..canonicalize import PLAN_FIELDS
from ..normalization import normalize_medical_text

REFERENCE_SCHEMA_VERSION = "rehab.llm-benchmark-reference.v2"
REQUIRED_SECTIONS = ("integrated_assessment", "rehabilitation_plan")
EXPECTED_PLAN_COUNT = 3
FORBIDDEN_SECTIONS = (
    "biomarker_interpretation",
    "clinical_subtype",
    "subtype",
    "evidence_summary",
    "limitations",
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


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(normalize_medical_text(value))


def validate_reference_payload(payload: Any, *, source_path: str | None = None) -> ReferenceValidationResult:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ReferenceValidationResult(False, None, ("reference_not_object",))
    for field in ("schema_version", "reference_id", "patient_id", "reference_version", "review_status"):
        if not _nonempty(payload.get(field)):
            errors.append(f"missing_or_empty:{field}")
    if payload.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if payload.get("review_status") != "approved":
        errors.append("review_status_not_approved")
    sections = payload.get("sections")
    if not isinstance(sections, Mapping):
        errors.append("sections_not_object")
        sections = {}
    for section in REQUIRED_SECTIONS:
        if section == "rehabilitation_plan":
            continue
        if not _nonempty(sections.get(section)):
            errors.append(f"missing_or_empty_section:{section}")
    for forbidden in FORBIDDEN_SECTIONS:
        if forbidden in sections:
            errors.append(f"forbidden_section:{forbidden}")
    unexpected = sorted(str(key) for key in sections.keys() if key not in REQUIRED_SECTIONS)
    for section in unexpected:
        if f"forbidden_section:{section}" not in errors:
            errors.append(f"unexpected_section:{section}")
    plan = sections.get("rehabilitation_plan")
    if not isinstance(plan, list) or not plan:
        errors.append("missing_or_empty_section:rehabilitation_plan")
    elif len(plan) != EXPECTED_PLAN_COUNT:
        errors.append(f"rehabilitation_plan_count_must_be:{EXPECTED_PLAN_COUNT}")
    elif not all(isinstance(item, Mapping) for item in plan):
        errors.append("rehabilitation_plan_item_not_object")
    else:
        for position, item in enumerate(plan, start=1):
            for field in PLAN_FIELDS:
                if not _nonempty(item.get(field)):
                    errors.append(f"missing_or_empty_plan_field:{position}:{field}")
    audit = payload.get("audit", {})
    if not isinstance(audit, Mapping):
        errors.append("audit_not_object")
        audit = {}
    if errors:
        return ReferenceValidationResult(False, None, tuple(errors))
    return ReferenceValidationResult(
        True,
        GoldReference(
            schema_version=str(payload["schema_version"]),
            reference_id=str(payload["reference_id"]),
            patient_id=str(payload["patient_id"]),
            reference_version=str(payload["reference_version"]),
            review_status=str(payload["review_status"]),
            sections=dict(sections),
            audit=dict(audit),
            source_path=source_path,
        ),
        (),
    )


def load_reference_set(root: str | Path) -> tuple[dict[str, GoldReference], list[dict[str, Any]], str]:
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
        data_path = str(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_items.append({"path": str(path.relative_to(root_path)), "sha256": digest})
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            errors.append({"stage": "reference", "source_path": data_path, "error_code": "REFERENCE_INVALID", "error_message": f"{type(exc).__name__}: {exc}"})
            continue
        result = validate_reference_payload(payload, source_path=data_path)
        if not result.valid or result.reference is None:
            errors.append({"stage": "reference", "source_path": data_path, "error_code": "REFERENCE_INVALID", "error_message": ";".join(result.errors)})
            continue
        reference = result.reference
        if reference.patient_id in references:
            references.pop(reference.patient_id, None)
            errors.append({"stage": "reference", "source_path": data_path, "patient_id": reference.patient_id, "error_code": "REFERENCE_DUPLICATE_PATIENT", "error_message": "duplicate patient_id"})
            continue
        references[reference.patient_id] = reference
    manifest_hash = hashlib.sha256(json.dumps(manifest_items, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return references, errors, manifest_hash


__all__ = [
    "FORBIDDEN_SECTIONS",
    "EXPECTED_PLAN_COUNT",
    "GoldReference",
    "REFERENCE_SCHEMA_VERSION",
    "REQUIRED_SECTIONS",
    "ReferenceValidationResult",
    "load_reference_set",
    "validate_reference_payload",
]
