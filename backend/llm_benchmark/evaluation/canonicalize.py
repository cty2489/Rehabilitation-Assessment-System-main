"""Canonical text conversion for the structured rehabilitation plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .normalization import normalize_medical_text

PLAN_FIELDS = ("action", "goal", "reason", "precaution")


@dataclass(frozen=True)
class CanonicalPlan:
    text: str
    count: int
    missing: bool
    missing_fields: tuple[str, ...]
    plan_count_violation: bool
    structural_error: str | None = None


def canonicalize_rehabilitation_plan(plan: Any, *, expected_count: int | None = None) -> CanonicalPlan:
    """Join actual field values only, retaining original action order.

    The historical default remains 3–5 for v1 compatibility. Newer callers
    can pass an exact expected count without changing the shared text metric.
    """

    if plan is None:
        return CanonicalPlan("", 0, True, (), False)
    if not isinstance(plan, list):
        return CanonicalPlan("", 0, True, (), False, "rehabilitation_plan_not_list")

    blocks: list[str] = []
    missing_fields: list[str] = []
    for position, item in enumerate(plan, start=1):
        if not isinstance(item, Mapping):
            return CanonicalPlan(
                "",
                len(plan),
                not plan,
                (f"{position}:item",),
                bool(plan) and (
                    len(plan) != expected_count
                    if expected_count is not None
                    else (len(plan) < 3 or len(plan) > 5)
                ),
                "rehabilitation_action_not_object",
            )
        values: list[str] = []
        for field in PLAN_FIELDS:
            value = item.get(field, "")
            if not isinstance(value, str):
                value = ""
            normalized = normalize_medical_text(value)
            if not normalized:
                missing_fields.append(f"{position}:{field}")
            values.append(normalized)
        # Empty fields are represented by an empty string; labels are never
        # inserted.  Blank lines are only separators between real values.
        blocks.append("\n".join(value for value in values if value))

    return CanonicalPlan(
        "\n".join(block for block in blocks if block),
        len(plan),
        len(plan) == 0,
        tuple(missing_fields),
        bool(plan) and (
            len(plan) != expected_count
            if expected_count is not None
            else (len(plan) < 3 or len(plan) > 5)
        ),
    )


__all__ = ["CanonicalPlan", "PLAN_FIELDS", "canonicalize_rehabilitation_plan"]
