"""Versioned deterministic tokenizer for Chinese medical text."""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

TOKENIZER_VERSION = "ChineseMedicalTokenizer_v1"

# Longest/most specific patterns are tested first.  Punctuation is not a
# metric token, while the symbols inside a medical numeric expression are.
_RANGE_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?[\-–—]\d+(?:\.\d+)?(?:%|\+)?")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:%|\+)?")
_LATIN_RE = re.compile(r"[A-Za-z]+(?:[A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*)?")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_GREEK_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")


class ChineseMedicalTokenizer:
    """Chinese characters plus intact Latin/medical/numeric tokens."""

    version = TOKENIZER_VERSION

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        index = 0
        value = str(text or "")
        while index < len(value):
            match = _RANGE_NUMBER_RE.match(value, index)
            if match:
                tokens.append(match.group(0))
                index = match.end()
                continue
            match = _NUMBER_RE.match(value, index)
            if match:
                tokens.append(match.group(0))
                index = match.end()
                continue
            match = _LATIN_RE.match(value, index)
            if match:
                tokens.append(match.group(0))
                index = match.end()
                continue
            match = _CJK_RE.match(value, index)
            if match:
                tokens.append(match.group(0))
                index = match.end()
                continue
            match = _GREEK_RE.match(value, index)
            if match:
                tokens.append(match.group(0))
                index = match.end()
                continue
            # Whitespace and all punctuation are delimiters, not tokens.
            index += 1
        return tokens

    def tokenize_many(self, texts: Sequence[str]) -> list[list[str]]:
        return [self.tokenize(text) for text in texts]


def tokenizer_rules_hash() -> str:
    rules = "\n".join(
        [
            TOKENIZER_VERSION,
            _RANGE_NUMBER_RE.pattern,
            _NUMBER_RE.pattern,
            _LATIN_RE.pattern,
            _CJK_RE.pattern,
            _GREEK_RE.pattern,
            "punctuation=delimiter",
        ]
    )
    return hashlib.sha256(rules.encode("utf-8")).hexdigest()


__all__ = [
    "ChineseMedicalTokenizer",
    "TOKENIZER_VERSION",
    "tokenizer_rules_hash",
]
