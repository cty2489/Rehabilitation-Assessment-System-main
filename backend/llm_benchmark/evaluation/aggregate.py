"""Single-case, batch, aggregation, and artifact writing for metrics v1."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidate_reader import CANONICAL_SECTIONS, CandidateRecord, missing_candidate_record, read_candidate_report
from .canonicalize import canonicalize_rehabilitation_plan
from .metrics import METRICS_VERSION, ROUGE_NAMES, corpus_bleu, score_rouge, sentence_bleu
from .normalization import NORMALIZATION_VERSION, normalize_medical_text, normalization_rules_hash
from .reference_schema import GoldReference, load_reference_set
from .tokenizer import ChineseMedicalTokenizer, TOKENIZER_VERSION, tokenizer_rules_hash

ALL_SECTIONS = (*CANONICAL_SECTIONS, "overall_generated_content")
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class CaseEvaluation:
    rows: tuple[dict[str, Any], ...]
    reference_texts: dict[str, str]
    candidate_texts: dict[str, str]
    errors: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EvaluationRun:
    evaluation_id: str
    output_root: Path
    rows: tuple[dict[str, Any], ...]
    summaries: tuple[dict[str, Any], ...]
    corpus_bleu_results: dict[str, Any]
    errors: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _slugify(value: str) -> str:
    return _SLUG_RE.sub("_", value).strip("._") or "model"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _empty_metric() -> dict[str, float]:
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def _reference_texts(reference: GoldReference) -> dict[str, str]:
    plan = canonicalize_rehabilitation_plan(reference.sections["rehabilitation_plan"])
    texts = {
        "biomarker_interpretation": normalize_medical_text(reference.sections["biomarker_interpretation"]),
        "integrated_assessment": normalize_medical_text(reference.sections["integrated_assessment"]),
        "rehabilitation_plan": plan.text,
    }
    texts["overall_generated_content"] = "\n".join(texts[name] for name in CANONICAL_SECTIONS if texts[name]).strip()
    return texts


def _candidate_texts(candidate: CandidateRecord) -> dict[str, str]:
    payload = candidate.parsed_output or {}
    texts = {
        "biomarker_interpretation": normalize_medical_text(payload.get("biomarker_interpretation", "")),
        "integrated_assessment": normalize_medical_text(payload.get("integrated_assessment", "")),
        "rehabilitation_plan": canonicalize_rehabilitation_plan(payload.get("rehabilitation_plan")).text,
    }
    if candidate.invalid_candidate:
        texts = {name: "" for name in CANONICAL_SECTIONS}
    else:
        for section in candidate.missing_sections:
            texts[section] = ""
    if candidate.invalid_candidate or candidate.missing_sections:
        texts["overall_generated_content"] = ""
    else:
        texts["overall_generated_content"] = "\n".join(
            texts[name] for name in CANONICAL_SECTIONS if texts[name]
        ).strip()
    return texts


def evaluate_single_case(
    reference: GoldReference,
    candidate: CandidateRecord,
    *,
    tokenizer: ChineseMedicalTokenizer | None = None,
) -> CaseEvaluation:
    """Evaluate one valid Gold and one Candidate, retaining zero rows on failure."""

    tokenizer = tokenizer or ChineseMedicalTokenizer()
    reference_texts = _reference_texts(reference)
    candidate_texts = _candidate_texts(candidate)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for error in candidate.errors:
        errors.append({
            "patient_id": candidate.patient_id,
            "model_id": candidate.model_id,
            "stage": "candidate",
            **error,
        })

    for section in ALL_SECTIONS:
        reference_text = reference_texts[section]
        candidate_text = candidate_texts[section]
        rouge = {name: _empty_metric() for name in ROUGE_NAMES}
        if candidate_text and reference_text:
            values = score_rouge(reference_text, candidate_text, tokenizer=tokenizer)
            rouge = {
                name: {
                    "precision": values[name].precision,
                    "recall": values[name].recall,
                    "f1": values[name].f1,
                }
                for name in ROUGE_NAMES
            }
        row = {
            "patient_id": candidate.patient_id,
            "model_id": candidate.model_id,
            "section": section,
            "reference_version": reference.reference_version,
            "rouge1_precision": rouge["rouge1"]["precision"],
            "rouge1_recall": rouge["rouge1"]["recall"],
            "rouge1_f1": rouge["rouge1"]["f1"],
            "rouge2_precision": rouge["rouge2"]["precision"],
            "rouge2_recall": rouge["rouge2"]["recall"],
            "rouge2_f1": rouge["rouge2"]["f1"],
            "rougeL_precision": rouge["rougeL"]["precision"],
            "rougeL_recall": rouge["rougeL"]["recall"],
            "rougeL_f1": rouge["rougeL"]["f1"],
            "sentence_bleu4": sentence_bleu(reference_text, candidate_text) if candidate_text else 0.0,
            "schema_valid": candidate.schema_valid,
            "parse_success": candidate.parse_success,
            "invalid_candidate": candidate.invalid_candidate,
            "missing_section": section in candidate.missing_sections or (
                section == "overall_generated_content" and bool(candidate.missing_sections)
            ),
            "plan_count_violation": candidate.plan_count_violation,
            "missing_plan_field": "|".join(candidate.missing_plan_fields),
            "unexpected_sections": "|".join(candidate.unexpected_sections),
            "candidate_token_count": len(tokenizer.tokenize(candidate_text)),
            "reference_token_count": len(tokenizer.tokenize(reference_text)),
            "candidate_source_path": candidate.source_path or "",
            "normalization_version": NORMALIZATION_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "candidate_text_hash": _sha256_text(candidate_text),
            "reference_text_hash": _sha256_text(reference_text),
            "error_code": "|".join(str(error.get("error_code", "")) for error in candidate.errors),
            "error_details": "|".join(str(error.get("error_message", "")) for error in candidate.errors),
        }
        rows.append(row)
    return CaseEvaluation(tuple(rows), reference_texts, candidate_texts, tuple(errors))


def _discover_patient_ids(batch_root: Path) -> list[str]:
    patients_root = batch_root / "patients"
    if not patients_root.exists():
        return []
    return sorted(path.name for path in patients_root.iterdir() if path.is_dir())


def _discover_model_ids(batch_root: Path) -> list[str]:
    models: set[str] = set()
    patients_root = batch_root / "patients"
    if patients_root.exists():
        for report in patients_root.glob("*/reports/*.json"):
            models.add(report.stem)
    return sorted(models)


def _candidate_path(batch_root: Path, patient_id: str, model_id: str) -> Path:
    reports = batch_root / "patients" / patient_id / "reports"
    exact = reports / f"{model_id}.json"
    slugged = reports / f"{_slugify(model_id)}.json"
    if exact.exists():
        return exact
    return slugged


def _candidate_manifest(batch_root: Path, patient_ids: Sequence[str], model_ids: Sequence[str]) -> tuple[list[dict[str, Any]], str]:
    items: list[dict[str, Any]] = []
    for model_id in sorted(model_ids):
        for patient_id in patient_ids:
            path = _candidate_path(batch_root, patient_id, model_id)
            item: dict[str, Any] = {"model_id": model_id, "patient_id": patient_id, "path": str(path)}
            if path.exists():
                item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                item["missing"] = True
            items.append(item)
    return items, _sha256_json(items)


def _git_metadata() -> dict[str, Any]:
    source_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source_root, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except Exception:
        commit = None
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("rouge-score", "sacrebleu"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _build_manifest(
    *,
    evaluation_id: str,
    patient_ids: Sequence[str],
    model_ids: Sequence[str],
    reference_version: str | list[str],
    reference_manifest_hash: str,
    candidate_manifest_hash: str,
    candidate_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "metrics_version": METRICS_VERSION,
        "evaluation_id": evaluation_id,
        "timestamp": _utc_timestamp(),
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "unicode": "NFKC",
            "rules_hash": normalization_rules_hash(),
        },
        "rouge": {
            "implementation": "rouge-score",
            "version": importlib.metadata.version("rouge-score"),
            "metrics": list(ROUGE_NAMES),
            "reported_value": "precision_recall_f1; primary=f1",
            "tokenizer": TOKENIZER_VERSION,
            "tokenizer_rules_hash": tokenizer_rules_hash(),
            "stemming": False,
            "aggregation": "macro_mean_per_case_f1_with_fixed_denominator",
        },
        "bleu": {
            "implementation": "sacrebleu",
            "version": importlib.metadata.version("sacrebleu"),
            "metric": "BLEU-4",
            "tokenizer": "zh",
            "lowercase": False,
            "smooth_method": "exp",
            "nrefs": 1,
            "primary": "corpus_bleu",
            "corpus_effective_order": False,
            "sentence_effective_order": True,
        },
        "included_sections": list(ALL_SECTIONS),
        "excluded_content": [
            "patient facts",
            "raw clinical scores",
            "raw biomarker table",
            "fixed headings",
            "JSON field labels",
            "RAG citations",
            "KG metadata",
        ],
        "missing_candidate_policy": "retain_case_with_empty_candidate_and_zero_metrics",
        "reference_invalid_policy": "exclude_invalid_reference_from_formal_denominator_and_log_error",
        "plan_count_policy": "score_text_and_record_violation",
        "reference_version": reference_version,
        "case_order": list(patient_ids),
        "expected_patient_ids": list(patient_ids),
        "case_order_hash": _sha256_json(list(patient_ids)),
        "model_ids": list(model_ids),
        "reference_manifest_hash": reference_manifest_hash,
        "candidate_manifest_hash": candidate_manifest_hash,
        "candidate_manifest": candidate_manifest,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        **_git_metadata(),
    }


def _model_summary(model_id: str, rows: Sequence[dict[str, Any]], corpus: Mapping[str, Any], expected: int) -> dict[str, Any]:
    model_rows = [row for row in rows if row["model_id"] == model_id]
    by_case = {row["patient_id"]: row for row in model_rows if row["section"] == "overall_generated_content"}
    n_invalid = sum(bool(row["invalid_candidate"]) for row in by_case.values())
    n_missing = sum(bool(row["missing_section"]) for row in by_case.values())
    n_valid = sum(bool(row["schema_valid"]) for row in by_case.values())
    summary: dict[str, Any] = {
        "model_id": model_id,
        "n_expected": expected,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "n_missing": n_missing,
        "generation_failure_rate": (n_invalid / expected) if expected else 0.0,
    }
    for section in ALL_SECTIONS:
        section_rows = [row for row in model_rows if row["section"] == section]
        for metric in ("rouge1", "rouge2", "rougeL"):
            values = [float(row[f"{metric}_f1"]) for row in section_rows]
            prefix = "overall" if section == "overall_generated_content" else section
            summary[f"{prefix}_{metric}_mean"] = (sum(values) / len(values)) if values else 0.0
            summary[f"{prefix}_{metric}_std"] = _std(values)
    for section in ALL_SECTIONS:
        result = corpus.get(section, {})
        prefix = "overall" if section == "overall_generated_content" else section
        summary[f"{prefix}_corpus_bleu4"] = result.get("bleu", 0.0)
    summary["overall_corpus_bleu4"] = corpus.get("overall_generated_content", {}).get("bleu", 0.0)
    return summary


def evaluate_batch(
    *,
    batch_root: str | Path,
    reference_root: str | Path,
    output_root: str | Path,
    model_ids: Sequence[str] | None = None,
    evaluation_id: str | None = None,
) -> EvaluationRun:
    """Evaluate all valid Gold cases against every requested model."""

    batch_path = Path(batch_root)
    references, reference_errors, reference_hash = load_reference_set(reference_root)
    if not references:
        raise ValueError("没有可用于正式评价的approved Gold Reference")
    patient_ids = sorted(references)
    discovered = _discover_model_ids(batch_path)
    resolved_models = sorted(set(model_ids or discovered))
    if not resolved_models:
        raise ValueError("未发现Candidate模型；请使用--model-id显式指定")
    candidate_manifest, candidate_hash = _candidate_manifest(batch_path, patient_ids, resolved_models)
    tokenizer = ChineseMedicalTokenizer()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = list(reference_errors)
    corpus_inputs: dict[str, dict[str, list[str]]] = {
        model: {section: [] for section in ALL_SECTIONS} for model in resolved_models
    }
    for model_id in resolved_models:
        for patient_id in patient_ids:
            path = _candidate_path(batch_path, patient_id, model_id)
            candidate = read_candidate_report(path, patient_id=patient_id, model_id=model_id)
            reference = references[patient_id]
            case = evaluate_single_case(reference, candidate, tokenizer=tokenizer)
            rows.extend(case.rows)
            errors.extend(case.errors)
            for section in ALL_SECTIONS:
                corpus_inputs[model_id][section].append(case.candidate_texts[section])
                if len(corpus_inputs[model_id].get(f"reference:{section}", [])) < len(patient_ids):
                    corpus_inputs[model_id].setdefault(f"reference:{section}", []).append(case.reference_texts[section])

    corpus_results: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    for model_id in resolved_models:
        corpus_results[model_id] = {}
        for section in ALL_SECTIONS:
            references_for_section = corpus_inputs[model_id][f"reference:{section}"]
            candidates_for_section = corpus_inputs[model_id][section]
            corpus_results[model_id][section] = corpus_bleu(references_for_section, candidates_for_section)
        summaries.append(_model_summary(model_id, rows, corpus_results[model_id], len(patient_ids)))

    resolved_evaluation_id = evaluation_id or f"evaluation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    reference_versions = sorted({reference.reference_version for reference in references.values()})
    manifest = _build_manifest(
        evaluation_id=resolved_evaluation_id,
        patient_ids=patient_ids,
        model_ids=resolved_models,
        reference_version=reference_versions[0] if len(reference_versions) == 1 else reference_versions,
        reference_manifest_hash=reference_hash,
        candidate_manifest_hash=candidate_hash,
        candidate_manifest=candidate_manifest,
    )
    return EvaluationRun(
        evaluation_id=resolved_evaluation_id,
        output_root=Path(output_root),
        rows=tuple(rows),
        summaries=tuple(summaries),
        corpus_bleu_results=corpus_results,
        errors=tuple(errors),
        manifest=manifest,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


CASE_FIELDS = [
    "patient_id", "model_id", "section", "reference_version",
    "rouge1_precision", "rouge1_recall", "rouge1_f1",
    "rouge2_precision", "rouge2_recall", "rouge2_f1",
    "rougeL_precision", "rougeL_recall", "rougeL_f1", "sentence_bleu4",
    "schema_valid", "parse_success", "invalid_candidate", "missing_section",
    "plan_count_violation", "missing_plan_field", "unexpected_sections",
    "candidate_token_count", "reference_token_count", "candidate_source_path",
    "normalization_version", "tokenizer_version", "candidate_text_hash",
    "reference_text_hash", "error_code", "error_details",
]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_evaluation_outputs(run: EvaluationRun, output_root: str | Path | None = None) -> Path:
    """Write the six required artifacts and return their directory."""

    target = Path(output_root) if output_root is not None else run.output_root
    target.mkdir(parents=True, exist_ok=True)
    _write_csv(target / "per_case_metrics.csv", run.rows, CASE_FIELDS)
    _write_jsonl(target / "per_case_metrics.jsonl", run.rows)
    summary_fields = sorted({key for row in run.summaries for key in row.keys()})
    _write_csv(target / "per_model_summary.csv", run.summaries, summary_fields)
    _write_json(target / "corpus_bleu.json", run.corpus_bleu_results)
    errors = []
    for error in run.errors:
        errors.append({"timestamp": _utc_timestamp(), "evaluation_id": run.evaluation_id, **error})
    _write_jsonl(target / "evaluation_errors.jsonl", errors)
    _write_json(target / "metric_manifest.json", run.manifest)
    return target


__all__ = [
    "ALL_SECTIONS",
    "EvaluationRun",
    "evaluate_batch",
    "evaluate_single_case",
    "write_evaluation_outputs",
]
