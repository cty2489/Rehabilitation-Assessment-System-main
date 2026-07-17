#!/usr/bin/env python3
"""Prepare the structured expert-review JSON for an isolated RAG trial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.ingest import prepare_review_json_knowledge_base


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare governed RAG review JSON")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--collection", default="rehab_knowledge_trial_v0_1")
    parser.add_argument(
        "--allow-internal-trial",
        action="store_true",
        help="Explicitly allow unreviewed entries in a non-clinical trial collection",
    )
    args = parser.parse_args()
    result = prepare_review_json_knowledge_base(
        args.input,
        args.output_dir,
        collection_id=args.collection,
        allow_internal_trial=args.allow_internal_trial,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
