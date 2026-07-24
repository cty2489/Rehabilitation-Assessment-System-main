"""Literature-backed reference ranges for the 18 BJH biomarkers (pure data).

Replaces the previous per-Brunnstrom-stage lo/hi tables. The new biomarker
program (``biomarkers/biomarkers.py``) ships a literature-sourced reference file
``biomarkers/out/biomarker_reference_ranges.json`` whose semantics differ:

* ``reference_type == "healthy_norm"``    → has a quantitative healthy range/mean.
* ``reference_type == "directional_trend"``→ no absolute threshold; literature only
  states the expected direction with recovery (↑/↓).
* ``reference_type == "none"``            → device/protocol-specific quantity, no
  literature standard at all (most raw EMG voltages, IEMG, IMU gyro, power changes).

This module is dependency-free (json + stdlib) so ``report_builder.py``'s rule
fallback can import it without pulling in the scipy DSP stack.

Public API:
    marker_ref(key)   -> dict | None   (raw entry from the JSON, normalised)
    ref_display(key)  -> str           (legacy/storage-only reference summary)
    judge(key, value) -> str           (rule-fallback verdict, ref-type aware)
    REF_META                            (the parsed {key: entry} map)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

# biomarkers/out/biomarker_reference_ranges.json lives at the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_REF_JSON = _REPO_ROOT / "biomarkers" / "out" / "biomarker_reference_ranges.json"


def _load() -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(_REF_JSON.read_text(encoding="utf-8"))
        return dict(data.get("biomarkers", {}))
    except Exception as exc:  # noqa: BLE001 — missing refs must not break the report
        print(f"[biomarker_refs][warn] failed to read {_REF_JSON}: {exc}")
        return {}


REF_META: Dict[str, Dict[str, Any]] = _load()

_DIR_ARROW = {"increase": "↑", "decrease": "↓"}
_DIR_WORD = {"increase": "升高", "decrease": "下降"}

# The current implementation computes SPARC from acceleration magnitude, while
# the literature range was derived from velocity SPARC. Keep the literature
# metadata for audit, but never compare the two absolute scales.
_NONCOMPARABLE_HEALTHY_NORMS = set()


def normalize_evidence_note(value: Any) -> str:
    """Replace stale cohort-ranking wording with the supported comparison basis."""
    note = str(value or "")
    replacements = {
        "绝对值仅供队列内排名": "绝对值仅用于同一患者在同设备、同流程下复测比较",
        "绝对值仅队列内排名": "绝对值仅用于同一患者在同设备、同流程下复测比较",
        "仅供队列内方向比较": "仅用于同一患者在同设备、同流程下复测比较",
        "仅供队列内方向参考": "仅用于同一患者在同设备、同流程下复测比较",
    }
    for old, new in replacements.items():
        note = note.replace(old, new)
    return note


def _range_bounds(entry: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Pull (lo, hi) out of a healthy_norm entry's range, if any."""
    hr = entry.get("healthy_reference") or {}
    rng = hr.get("range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        return float(rng[0]), float(rng[1])
    return None, None


def marker_ref(key: str) -> Optional[Dict[str, Any]]:
    """Normalised reference entry for one biomarker, or None if unknown."""
    entry = REF_META.get(key)
    if entry is None:
        return None
    lo, hi = _range_bounds(entry)
    return {
        "units": entry.get("units", ""),
        "reference_type": entry.get("reference_type", "none"),
        "expected_direction": entry.get("expected_direction_with_recovery", "n/a"),
        "lo": lo,
        "hi": hi,
        "confidence": entry.get("confidence", "none"),
        "note": normalize_evidence_note(entry.get("note_zh", "")),
        "source": entry.get("source", []),
        "absolute_comparison_applicable": (
            entry.get("reference_type") == "healthy_norm"
            and key not in _NONCOMPARABLE_HEALTHY_NORMS
            and (lo is not None or hi is not None)
        ),
    }


def _fmt_num(x: float) -> str:
    return f"{x:g}"


def ref_display(key: str) -> str:
    """Compact legacy/storage summary; user-facing reports hide this field."""
    ref = marker_ref(key)
    if ref is None:
        return "—"
    rtype = ref["reference_type"]
    direction = ref["expected_direction"]
    arrow = _DIR_ARROW.get(direction, "")
    expect = f"（恢复期望{arrow}）" if arrow else ""

    if rtype == "healthy_norm":
        if not ref["absolute_comparison_applicable"]:
            return "当前算法与文献常模尺度不可直接比较"
        if ref["lo"] is not None and ref["hi"] is not None:
            return f"健康常模 {_fmt_num(ref['lo'])}–{_fmt_num(ref['hi'])}{expect}"
        hr = (REF_META.get(key, {}) or {}).get("healthy_reference") or {}
        mean = hr.get("mean")
        if mean is not None:
            return f"健康常模均值 {_fmt_num(float(mean))}{expect}"
        return f"健康常模参考{expect}"

    if rtype == "directional_trend":
        if arrow:
            return f"无绝对阈值；康复过程通常{arrow}"
        return "无绝对阈值；仅支持同条件复测"

    # reference_type == "none"
    return "不适用；仅支持同设备、同流程复测"


def judge(key: str, value: Optional[float]) -> str:
    """Rule-fallback verdict for a biomarker value, aware of reference type.

    healthy_norm with a range → 低于/处于/高于参考范围.
    directional_trend / none  → 方向性判语 (无绝对阈值, only direction matters);
    these never produce a hard 超标 verdict.
    """
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return "数据不可用"
    ref = marker_ref(key)
    if ref is None:
        return "暂无量化参考"

    lo, hi = ref["lo"], ref["hi"]
    if ref["absolute_comparison_applicable"]:
        if lo is not None and value < lo:
            return "低于参考范围"
        if hi is not None and value > hi:
            return "高于参考范围"
        return "处于参考范围内"

    # No valid absolute comparator: never label a single value normal/abnormal.
    direction = ref["expected_direction"]
    word = _DIR_WORD.get(direction)
    if word:
        return f"单次值不作正常/异常判断；康复过程通常{word}"
    return "单次值不作正常/异常判断；需同条件复测"


__all__ = ["REF_META", "marker_ref", "ref_display", "judge", "normalize_evidence_note"]
