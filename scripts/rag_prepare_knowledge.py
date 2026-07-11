#!/usr/bin/env python3
"""Prepare a private DOCX knowledge source for later vector indexing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.ingest import prepare_knowledge_base


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse a rehabilitation DOCX into governed JSONL knowledge records."
    )
    parser.add_argument("--input", required=True, help="Private source .docx path")
    parser.add_argument(
        "--config",
        default=str(ROOT / "knowledge_base/config/rehab_knowledge_demo_v0_1.json"),
        help="Ingestion governance and manual transcription config",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "knowledge_base/runtime/rehab_knowledge_demo_v0_1"),
        help="Generated private output directory",
    )
    args = parser.parse_args()

    result = prepare_knowledge_base(args.input, args.config, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
