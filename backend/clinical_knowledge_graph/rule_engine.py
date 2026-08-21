"""Data-availability rules kept separate from the static graph relations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


_RULES_PATH = Path(__file__).resolve().parent / "clinical_rules.json"


class RuleDataError(ValueError):
    """Raised when the editable prototype rule file is malformed."""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuleDataError(f"无法读取临床规则：{path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("rules"), list):
        raise RuleDataError("clinical_rules.json必须包含rules列表")
    return value


def _lookup(value: Mapping[str, Any], field: str) -> Any:
    current: Any = value
    for piece in field.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(piece)
    return current


def _matches(condition: Mapping[str, Any], values: Mapping[str, Any]) -> bool:
    actual = _lookup(values, str(condition.get("field") or ""))
    expected = condition.get("value")
    operator = condition.get("operator")
    if operator == "equals":
        return actual == expected
    if operator == "greater_than":
        return isinstance(actual, (int, float)) and actual > expected
    if operator == "greater_than_or_equal":
        return isinstance(actual, (int, float)) and actual >= expected
    raise RuleDataError(f"不支持的规则操作符：{operator}")


class RuleEngine:
    """Evaluate non-clinical, reviewable structure and quality rules."""

    def __init__(self, rules_path: Optional[Path] = None) -> None:
        self._data = _read_json(rules_path or _RULES_PATH)

    def evaluate(self, evidence_bundle: Mapping[str, Any]) -> list[Dict[str, Any]]:
        patient = evidence_bundle.get("patient_summary") or {}
        values = {
            "quality": patient.get("quality") or {},
            "summary": {
                "available_by_modality": patient.get("available_by_modality") or {},
                "missing_non_imu_marker_count": patient.get("missing_non_imu_marker_count", 0),
            },
        }
        results: list[Dict[str, Any]] = []
        for rule in self._data["rules"]:
            conditions: Iterable[Mapping[str, Any]]
            if "when_all" in rule:
                conditions = rule["when_all"]
            else:
                conditions = [rule["when"]]
            matched = all(_matches(condition, values) for condition in conditions)
            results.append(
                {
                    "rule_id": rule["rule_id"],
                    "matched": matched,
                    "result": dict(rule["result"]),
                    "evidence_level": rule["evidence_level"],
                    "expert_review_status": rule["expert_review_status"],
                    "rule_scope": "data_availability_and_quality_only",
                }
            )
        return results


__all__ = ["RuleDataError", "RuleEngine"]
