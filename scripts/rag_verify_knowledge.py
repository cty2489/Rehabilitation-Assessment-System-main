#!/usr/bin/env python3
"""Run deterministic ingestion checks before adding an embedding model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _search(entries: Iterable[Dict[str, Any]], terms: List[str]) -> List[Dict[str, Any]]:
    scored = []
    for entry in entries:
        searchable = "\n".join(
            [
                entry["title"],
                entry["content"],
                " ".join(entry["keywords"]),
                " ".join(entry["aliases"]),
            ]
        ).lower()
        score = sum(1 for term in terms if term.lower() in searchable)
        if score:
            scored.append((score, entry["knowledge_id"], entry))
    return [item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify prepared RAG knowledge records")
    parser.add_argument("--knowledge-dir", required=True)
    parser.add_argument(
        "--queries",
        default=str(Path(__file__).resolve().parents[1] / "knowledge_base/eval/demo_queries.jsonl"),
    )
    args = parser.parse_args()

    root = Path(args.knowledge_dir)
    entries = _read_jsonl(root / "entries.jsonl")
    cases = _read_jsonl(Path(args.queries))
    failures = []
    for case in cases:
        results = _search(entries, case["match_terms"])
        actual = results[0]["knowledge_id"] if results else None
        passed = actual == case["expected_knowledge_id"]
        print(f"{'PASS' if passed else 'FAIL'} {case['query']} -> {actual}")
        if not passed:
            failures.append({"case": case, "actual": actual})

    quality = json.loads((root / "quality_report.json").read_text(encoding="utf-8"))
    counts = quality["counts"]
    print(
        "QUALITY "
        f"total={counts['total_entries']} demo_ready={counts['demo_ready_entries']} "
        f"clinical_ready={counts['clinical_ready_entries']} chunks={counts['chunks']}"
    )
    if counts["clinical_ready_entries"] != 0:
        failures.append({"reason": "unreviewed demo unexpectedly marked clinical_ready"})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
