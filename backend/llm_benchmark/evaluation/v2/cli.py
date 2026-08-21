"""CLI for the isolated Benchmark v2 evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from ..metrics import corpus_bleu
from .aggregate import EvaluationRun, _manifest, _summary, evaluate_batch, evaluate_single_case, write_evaluation_outputs
from .candidate_reader import read_candidate_report
from .reference_schema import load_reference_set, validate_reference_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="康复大模型 Benchmark v2 ROUGE + BLEU 评价工具")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-reference")
    validate.add_argument("--reference-root", required=True)
    case = sub.add_parser("score-case")
    case.add_argument("--reference", required=True)
    case.add_argument("--candidate", required=True)
    case.add_argument("--patient-id", required=True)
    case.add_argument("--model-id", required=True)
    case.add_argument("--output-root", required=True)
    case.add_argument("--evaluation-id", default=None)
    batch = sub.add_parser("score-batch")
    batch.add_argument("--batch-root", required=True)
    batch.add_argument("--reference-root", required=True)
    batch.add_argument("--output-root", required=True)
    batch.add_argument("--model-id", action="append", dest="model_ids")
    batch.add_argument("--evaluation-id", default=None)
    return parser


def _score_case(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Reference读取失败：{exc}", file=sys.stderr)
        return 2
    validation = validate_reference_payload(payload, source_path=args.reference)
    if not validation.valid or validation.reference is None:
        print(json.dumps({"valid": False, "errors": list(validation.errors)}, ensure_ascii=False))
        return 2
    if validation.reference.patient_id != args.patient_id:
        print("Reference patient_id与--patient-id不一致", file=sys.stderr)
        return 2
    candidate = read_candidate_report(args.candidate, patient_id=args.patient_id, model_id=args.model_id)
    case = evaluate_single_case(validation.reference, candidate)
    corpus = {section: corpus_bleu([case.reference_texts[section]], [case.candidate_texts[section]]) for section in case.reference_texts}
    evaluation_id = args.evaluation_id or "evaluation-v2-single-case"
    candidate_file = Path(args.candidate)
    item = {"model_id": args.model_id, "patient_id": args.patient_id, "path": str(candidate_file)}
    if candidate_file.exists():
        item["sha256"] = hashlib.sha256(candidate_file.read_bytes()).hexdigest()
    else:
        item["missing"] = True
    candidate_manifest = [item]
    manifest = _manifest(
        evaluation_id=evaluation_id,
        patients=[args.patient_id],
        models=[args.model_id],
        references={args.patient_id: validation.reference},
        reference_hash=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
        candidate_hash=hashlib.sha256(json.dumps(candidate_manifest, sort_keys=True).encode("utf-8")).hexdigest(),
        candidate_manifest=candidate_manifest,
        prompt_versions=[candidate.prompt_version] if candidate.prompt_version else None,
    )
    run = EvaluationRun(
        evaluation_id=evaluation_id,
        output_root=Path(args.output_root),
        rows=case.rows,
        summaries=(_summary(args.model_id, case.rows, corpus, 1),),
        corpus_bleu_results={args.model_id: corpus},
        errors=case.errors,
        manifest=manifest,
    )
    output = write_evaluation_outputs(run)
    print(json.dumps({"output_root": str(output), "evaluation_id": evaluation_id}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-reference":
        references, errors, manifest_hash = load_reference_set(args.reference_root)
        print(json.dumps({"valid_reference_count": len(references), "reference_errors": errors, "reference_manifest_hash": manifest_hash}, ensure_ascii=False, indent=2))
        return 0 if not errors else 2
    if args.command == "score-case":
        return _score_case(args)
    if args.command == "score-batch":
        try:
            run = evaluate_batch(
                batch_root=args.batch_root,
                reference_root=args.reference_root,
                output_root=args.output_root,
                model_ids=args.model_ids,
                evaluation_id=args.evaluation_id,
            )
            output = write_evaluation_outputs(run)
        except Exception as exc:
            print(f"evaluation v2 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"output_root": str(output), "evaluation_id": run.evaluation_id}, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
