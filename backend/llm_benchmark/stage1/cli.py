"""Non-generative Stage 1 preparation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clinical_input import read_doctor_sheet
from .doctor_reference import write_gold_references, write_progress_csv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第一阶段Clinical-only Excel准备工具")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-doctor-sheet", help="输出doctor_progress.csv")
    validate.add_argument("--excel", required=True)
    validate.add_argument("--output", required=True)
    gold = sub.add_parser("build-gold-references", help="仅从L列已复核文本生成Gold v2")
    gold.add_argument("--excel", required=True)
    gold.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-doctor-sheet":
        cases = read_doctor_sheet(args.excel)
        path = write_progress_csv(cases, args.output)
        print(json.dumps({"patient_count": len(cases), "progress_csv": str(path)}, ensure_ascii=False))
        return 0
    if args.command == "build-gold-references":
        paths = write_gold_references(args.excel, args.output_root)
        print(json.dumps({"reference_count": len(paths), "references": [str(path) for path in paths]}, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
