"""Deterministic medical-text normalization for metrics v1.

Only presentation artifacts are removed.  No medical terminology, numbers,
directions, negation, or clinical meaning is rewritten.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

NORMALIZATION_VERSION = "medical_text_normalizer_v1"

_CODE_FENCE_RE = re.compile(r"^\s*```(?:[A-Za-z0-9_+-]+)?\s*$")
_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
_BULLET_RE = re.compile(r"^\s{0,3}[-*+]\s+")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_SPACE_RE = re.compile(r"[ \t]+")


def normalize_medical_text(value: object) -> str:
    """Normalize formatting without changing medical semantics."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines: list[str] = []
    previous_blank = False
    for raw_line in text.split("\n"):
        if _CODE_FENCE_RE.match(raw_line):
            continue
        # A line that is solely a Markdown heading is a fixed presentation
        # label, not generated medical content.
        if _HEADING_LINE_RE.match(raw_line):
            continue
        line = raw_line
        line = _HEADING_RE.sub("", line)
        line = _BULLET_RE.sub("", line)
        if _TABLE_SEPARATOR_RE.match(line):
            continue
        # Pipes are Markdown table controls.  A space keeps adjacent cells
        # separate without adding a literal template token.
        line = line.replace("|", " ")
        line = _SPACE_RE.sub(" ", line).strip()
        if not line:
            if not previous_blank and normalized_lines:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(line)
        previous_blank = False
    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()
    return "\n".join(normalized_lines).strip()


def normalization_rules_hash() -> str:
    rules = "\n".join(
        [
            NORMALIZATION_VERSION,
            _CODE_FENCE_RE.pattern,
            _HEADING_LINE_RE.pattern,
            _HEADING_RE.pattern,
            _BULLET_RE.pattern,
            _TABLE_SEPARATOR_RE.pattern,
            _SPACE_RE.pattern,
            "unicode=NFKC",
            "pipes=space",
            "preserve_medical_content=true",
        ]
    )
    return hashlib.sha256(rules.encode("utf-8")).hexdigest()


__all__ = [
    "NORMALIZATION_VERSION",
    "normalize_medical_text",
    "normalization_rules_hash",
]
