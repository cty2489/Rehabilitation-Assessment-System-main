"""Deterministic extraction of approved Gold Reference from doctor column L."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .clinical_input import Stage1Case, read_doctor_sheet

REFERENCE_SCHEMA_VERSION = "rehab.llm-benchmark-reference.v2"
IMPORTER_VERSION = "doctor_reference_parser_v1"
_PLACEHOLDER = "【填写】"
_PLAN_FIELDS = ("action", "goal", "reason", "precaution")


@dataclass(frozen=True)
class DoctorReferenceParseResult:
    patient_id: str
    reference_ready: bool
    sections: dict[str, Any] | None
    errors: tuple[str, ...]


def _clean(value: Any) -> str:
    text = str(value or "").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text).strip()
    text = text.strip(" \t\n，,；;。")
    if text.startswith("【") and text.endswith("】"):
        text = text[1:-1].strip()
    return text.strip(" \t\n，,；;。")


def _has_placeholder(text: str) -> bool:
    return _PLACEHOLDER in text


def _extract_assessment(text: str) -> tuple[str, str] | None:
    first = re.search(r"当前腕手功能主要表现为\s*(.*?)\s*(?=[；;]\s*主要功能受限为)", text, flags=re.S)
    second = re.search(r"主要功能受限为\s*(.*?)\s*(?=建议1\s*[:：]|$)", text, flags=re.S)
    if not first or not second:
        return None
    function = _clean(first.group(1))
    limitation = _clean(second.group(1))
    if not function or not limitation:
        return None
    return function, limitation


def _extract_plan(segment: str) -> dict[str, str] | None:
    goal_match = re.search(r"目标是\s*", segment)
    reason_match = re.search(r"原因是\s*", segment)
    precaution_match = re.search(r"(?:必要注意事项|注意)\s*[:：]?\s*", segment)
    if not goal_match or not reason_match or not precaution_match:
        return None
    action_text = segment[: goal_match.start()]
    action_text = re.sub(r"^建议\s*[1-3]\s*[:：]?\s*", "", action_text).strip()
    action_text = re.sub(r"^进行\s*", "", action_text).strip()
    action_text = _clean(action_text)
    if action_text.endswith("训练"):
        action_text = _clean(action_text[:-2])
    goal = _clean(segment[goal_match.end() : reason_match.start()])
    reason = _clean(segment[reason_match.end() : precaution_match.start()])
    precaution = _clean(segment[precaution_match.end() :])
    values = {"action": action_text, "goal": goal, "reason": reason, "precaution": precaution}
    return values if all(values.values()) else None


def _extract_plans(text: str) -> list[dict[str, str]]:
    starts = list(re.finditer(r"建议\s*([1-3])\s*[:：]", text))
    if len(starts) != 3 or [int(match.group(1)) for match in starts] != [1, 2, 3]:
        return []
    plans: list[dict[str, str]] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        plan = _extract_plan(text[match.start() : end])
        if plan is None:
            return []
        plans.append(plan)
    return plans


def parse_doctor_review(review_text: str, *, patient_id: str) -> DoctorReferenceParseResult:
    """Parse only actual doctor text from L; no LLM or auto-repair is used."""

    text = str(review_text or "").strip()
    errors: list[str] = []
    if not text:
        return DoctorReferenceParseResult(
            patient_id=patient_id,
            reference_ready=False,
            sections=None,
            errors=("WAITING_FOR_REVIEW",),
        )
    if _has_placeholder(text):
        return DoctorReferenceParseResult(
            patient_id=patient_id,
            reference_ready=False,
            sections=None,
            errors=("WAITING_FOR_REVIEW",),
        )
    assessment = _extract_assessment(text)
    if assessment is None:
        errors.append("ASSESSMENT_PARSE_FAILED")
    plans = _extract_plans(text)
    if len(plans) != 3:
        errors.append("REHABILITATION_PLAN_PARSE_FAILED")
    if errors:
        return DoctorReferenceParseResult(
            patient_id=patient_id,
            reference_ready=False,
            sections=None,
            errors=tuple(dict.fromkeys(errors)),
        )
    assert assessment is not None
    return DoctorReferenceParseResult(
        patient_id=patient_id,
        reference_ready=True,
        sections={
            "integrated_assessment": assessment[0] + "。" + assessment[1],
            "rehabilitation_plan": plans,
        },
        errors=(),
    )


def build_reference_payload(
    case: Stage1Case,
    parsed: DoctorReferenceParseResult,
    *,
    generated_at: str | None = None,
) -> dict[str, Any] | None:
    if not parsed.reference_ready or parsed.sections is None:
        return None
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "reference_id": f"REF-{case.patient_id}-V2",
        "patient_id": case.patient_id,
        "reference_version": "gold_v2",
        "review_status": "approved",
        "sections": parsed.sections,
        "audit": {
            "source": "doctor_excel",
            "draft_column": "K",
            "review_column": "L",
            "importer_version": IMPORTER_VERSION,
            "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        },
    }


def validate_doctor_sheet_progress(cases: Iterable[Stage1Case]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        parsed = parse_doctor_review(case.doctor2_text, patient_id=case.patient_id)
        rows.append({
            "patient_id": case.patient_id,
            "doctor1_complete": bool(case.doctor1_text and not _has_placeholder(case.doctor1_text)),
            "doctor2_complete": bool(case.doctor2_text and not _has_placeholder(case.doctor2_text)),
            "reference_ready": parsed.reference_ready,
            "error": "|".join(parsed.errors),
        })
    return rows


def write_gold_references(
    excel_path: str | Path,
    output_root: str | Path,
    *,
    generated_at: str | None = None,
) -> tuple[Path, ...]:
    cases = read_doctor_sheet(excel_path)
    output = Path(output_root)
    written: list[Path] = []
    for case in cases:
        parsed = parse_doctor_review(case.doctor2_text, patient_id=case.patient_id)
        payload = build_reference_payload(case, parsed, generated_at=generated_at)
        if payload is None:
            continue
        path = output / f"{case.patient_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return tuple(written)


def write_progress_csv(cases: Iterable[Stage1Case], output_path: str | Path) -> Path:
    rows = validate_doctor_sheet_progress(cases)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("patient_id", "doctor1_complete", "doctor2_complete", "reference_ready", "error"))
        writer.writeheader()
        writer.writerows({
            **row,
            "doctor1_complete": str(row["doctor1_complete"]).lower(),
            "doctor2_complete": str(row["doctor2_complete"]).lower(),
            "reference_ready": str(row["reference_ready"]).lower(),
        } for row in rows)
    return path


__all__ = [
    "DoctorReferenceParseResult",
    "IMPORTER_VERSION",
    "REFERENCE_SCHEMA_VERSION",
    "build_reference_payload",
    "parse_doctor_review",
    "validate_doctor_sheet_progress",
    "write_gold_references",
    "write_progress_csv",
]
