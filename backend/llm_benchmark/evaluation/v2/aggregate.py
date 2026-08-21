"""ROUGE/BLEU metrics v2: two generated sections plus overall."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..aggregate import (
    _git_metadata,
    _package_versions,
    _sha256_json,
    _sha256_text,
    _slugify,
    _std,
    _utc_timestamp,
)
from ..canonicalize import canonicalize_rehabilitation_plan
from ..metrics import ROUGE_NAMES, corpus_bleu, score_rouge, sentence_bleu
from ..normalization import NORMALIZATION_VERSION, normalize_medical_text, normalization_rules_hash
from ..tokenizer import TOKENIZER_VERSION, ChineseMedicalTokenizer, tokenizer_rules_hash
from .candidate_reader import CANONICAL_SECTIONS, CandidateRecord, read_candidate_report
from .reference_schema import GoldReference, REFERENCE_SCHEMA_VERSION, load_reference_set

METRICS_VERSION = "rehab_llm_metrics_v2"
PROMPT_VERSION = "rehab_llm_benchmark_v2"
ALL_SECTIONS = (*CANONICAL_SECTIONS, "overall_generated_content")
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


def _reference_texts(reference: GoldReference) -> dict[str, str]:
    plan = canonicalize_rehabilitation_plan(reference.sections["rehabilitation_plan"])
    texts = {
        "integrated_assessment": normalize_medical_text(reference.sections["integrated_assessment"]),
        "rehabilitation_plan": plan.text,
    }
    texts["overall_generated_content"] = "\n".join(texts[name] for name in CANONICAL_SECTIONS if texts[name]).strip()
    return texts


def _candidate_texts(candidate: CandidateRecord) -> dict[str, str]:
    payload = candidate.parsed_output or {}
    texts = {
        "integrated_assessment": normalize_medical_text(payload.get("integrated_assessment", "")),
        "rehabilitation_plan": canonicalize_rehabilitation_plan(payload.get("rehabilitation_plan")).text,
    }
    if candidate.invalid_candidate or candidate.candidate_missing:
        texts = {name: "" for name in CANONICAL_SECTIONS}
    else:
        for section in candidate.missing_sections:
            texts[section] = ""
    if candidate.invalid_candidate or candidate.candidate_missing or candidate.missing_sections:
        texts["overall_generated_content"] = ""
    else:
        texts["overall_generated_content"] = "\n".join(texts[name] for name in CANONICAL_SECTIONS if texts[name]).strip()
    return texts


def evaluate_single_case(reference: GoldReference, candidate: CandidateRecord, *, tokenizer: ChineseMedicalTokenizer | None = None) -> CaseEvaluation:
    tokenizer = tokenizer or ChineseMedicalTokenizer()
    reference_texts = _reference_texts(reference)
    candidate_texts = _candidate_texts(candidate)
    errors = tuple({"patient_id": candidate.patient_id, "model_id": candidate.model_id, "stage": "candidate", **error} for error in candidate.errors)
    rows: list[dict[str, Any]] = []
    for section in ALL_SECTIONS:
        reference_text = reference_texts[section]
        candidate_text = candidate_texts[section]
        values = {name: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for name in ROUGE_NAMES}
        if candidate_text and reference_text:
            scored = score_rouge(reference_text, candidate_text, tokenizer=tokenizer)
            values = {name: {"precision": scored[name].precision, "recall": scored[name].recall, "f1": scored[name].f1} for name in ROUGE_NAMES}
        rows.append({
            "patient_id": candidate.patient_id,
            "model_id": candidate.model_id,
            "section": section,
            "metrics_version": METRICS_VERSION,
            "reference_version": reference.reference_version,
            "rouge1_precision": values["rouge1"]["precision"],
            "rouge1_recall": values["rouge1"]["recall"],
            "rouge1_f1": values["rouge1"]["f1"],
            "rouge2_precision": values["rouge2"]["precision"],
            "rouge2_recall": values["rouge2"]["recall"],
            "rouge2_f1": values["rouge2"]["f1"],
            "rougeL_precision": values["rougeL"]["precision"],
            "rougeL_recall": values["rougeL"]["recall"],
            "rougeL_f1": values["rougeL"]["f1"],
            "sentence_bleu4": sentence_bleu(reference_text, candidate_text) if candidate_text else 0.0,
            "schema_valid": candidate.schema_valid,
            "parse_success": candidate.parse_success,
            "invalid_candidate": candidate.invalid_candidate,
            "candidate_missing": candidate.candidate_missing,
            "missing_section": section in candidate.missing_sections or (section == "overall_generated_content" and bool(candidate.missing_sections)),
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
        })
    return CaseEvaluation(tuple(rows), reference_texts, candidate_texts, errors)


def _candidate_path(batch_root: Path, patient_id: str, model_id: str) -> Path:
    reports = batch_root / "patients" / patient_id / "reports"
    exact = reports / f"{model_id}.json"
    slugged = reports / f"{_slugify(model_id)}.json"
    return exact if exact.exists() else slugged


def _discover_models(batch_root: Path) -> list[str]:
    return sorted({path.stem for path in (batch_root / "patients").glob("*/reports/*.json")})


def _candidate_manifest(batch_root: Path, patients: Sequence[str], models: Sequence[str]) -> tuple[list[dict[str, Any]], str]:
    items: list[dict[str, Any]] = []
    for model_id in sorted(models):
        for patient_id in patients:
            path = _candidate_path(batch_root, patient_id, model_id)
            item: dict[str, Any] = {"model_id": model_id, "patient_id": patient_id, "path": str(path)}
            if path.exists():
                item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                item["missing"] = True
            items.append(item)
    return items, _sha256_json(items)


def _manifest(*, evaluation_id: str, patients: Sequence[str], models: Sequence[str], references: Mapping[str, GoldReference], reference_hash: str, candidate_hash: str, candidate_manifest: list[dict[str, Any]], prompt_versions: Sequence[str] | None = None) -> dict[str, Any]:
    versions = sorted({item.reference_version for item in references.values()})
    return {
        "metrics_version": METRICS_VERSION,
        "evaluation_id": evaluation_id,
        "timestamp": _utc_timestamp(),
        "prompt_version": (sorted(set(prompt_versions or (PROMPT_VERSION,))) if len(set(prompt_versions or (PROMPT_VERSION,))) > 1 else next(iter(set(prompt_versions or (PROMPT_VERSION,))))),
        "reference_schema_version": REFERENCE_SCHEMA_VERSION,
        "normalization": {"version": NORMALIZATION_VERSION, "unicode": "NFKC", "rules_hash": normalization_rules_hash()},
        "rouge": {
            "implementation": "rouge-score",
            "version": _package_versions()["rouge-score"],
            "metrics": list(ROUGE_NAMES),
            "reported_value": "precision_recall_f1; primary=f1",
            "tokenizer": TOKENIZER_VERSION,
            "tokenizer_rules_hash": tokenizer_rules_hash(),
            "stemming": False,
            "aggregation": "macro_mean_per_case_f1_with_fixed_denominator",
        },
        "bleu": {
            "implementation": "sacrebleu",
            "version": _package_versions()["sacrebleu"],
            "metric": "BLEU-4",
            "tokenizer": "zh",
            "lowercase": False,
            "smooth_method": "exp",
            "effective_order": False,
            "sentence_effective_order": True,
            "nrefs": 1,
            "primary": "corpus_bleu",
        },
        "included_sections": list(ALL_SECTIONS),
        "excluded_sections": ["biomarker_interpretation", "clinical_subtype"],
        "biomarker_policy": {
            "single_visit": True,
            "standalone_biomarker_interpretation_scored": False,
            "absolute_normality_inference_allowed": False,
            "longitudinal_comparison_available": False,
        },
        "excluded_content": ["patient facts", "raw clinical scores", "raw biomarker table", "fixed headings", "RAG citations", "KG metadata", "subtype metadata"],
        "missing_candidate_policy": "retain_case_with_empty_candidate_and_zero_metrics",
        "reference_invalid_policy": "exclude_invalid_reference_from_formal_denominator_and_log_error",
        "plan_count_policy": "exactly_3_score_text_and_record_violation",
        "reference_version": versions[0] if len(versions) == 1 else versions,
        "case_order": list(patients),
        "expected_patient_ids": list(patients),
        "case_order_hash": _sha256_json(list(patients)),
        "model_ids": list(models),
        "reference_manifest_hash": reference_hash,
        "candidate_manifest_hash": candidate_hash,
        "candidate_manifest": candidate_manifest,
        "package_versions": _package_versions(),
        **_git_metadata(),
    }


def _summary(model_id: str, rows: Sequence[dict[str, Any]], corpus: Mapping[str, Any], expected: int) -> dict[str, Any]:
    overall = [row for row in rows if row["model_id"] == model_id and row["section"] == "overall_generated_content"]
    n_schema_valid = sum(bool(row["schema_valid"]) for row in overall)
    n_invalid = sum(bool(row["invalid_candidate"]) for row in overall)
    n_missing = sum(bool(row["candidate_missing"]) for row in overall)
    result: dict[str, Any] = {
        "model_id": model_id,
        "n_expected": expected,
        "n_scored": expected,
        "n_schema_valid": n_schema_valid,
        "n_invalid": n_invalid,
        "n_missing": n_missing,
        "generation_failure_rate": ((n_invalid + n_missing) / expected) if expected else 0.0,
    }
    for section in ALL_SECTIONS:
        section_rows = [row for row in rows if row["model_id"] == model_id and row["section"] == section]
        prefix = "overall" if section == "overall_generated_content" else section
        for metric in ("rouge1", "rouge2", "rougeL"):
            values = [float(row[f"{metric}_f1"]) for row in section_rows]
            result[f"{prefix}_{metric}_mean"] = sum(values) / len(values) if values else 0.0
            result[f"{prefix}_{metric}_std"] = _std(values)
        result[f"{prefix}_corpus_bleu4"] = corpus.get(section, {}).get("bleu", 0.0)
    result["overall_corpus_bleu4"] = corpus.get("overall_generated_content", {}).get("bleu", 0.0)
    return result


def evaluate_batch(*, batch_root: str | Path, reference_root: str | Path, output_root: str | Path, model_ids: Sequence[str] | None = None, evaluation_id: str | None = None) -> EvaluationRun:
    batch_path = Path(batch_root)
    references, reference_errors, reference_hash = load_reference_set(reference_root)
    if not references:
        raise ValueError("没有可用于正式v2评价的approved Gold Reference")
    patients = sorted(references)
    models = sorted(set(model_ids or _discover_models(batch_path)))
    if not models:
        raise ValueError("未发现v2 Candidate模型；请使用--model-id显式指定")
    candidate_manifest, candidate_hash = _candidate_manifest(batch_path, patients, models)
    tokenizer = ChineseMedicalTokenizer()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = list(reference_errors)
    corpus_inputs = {model: {section: [] for section in ALL_SECTIONS} for model in models}
    prompt_versions: set[str] = set()
    for model_id in models:
        for patient_id in patients:
            candidate = read_candidate_report(_candidate_path(batch_path, patient_id, model_id), patient_id=patient_id, model_id=model_id)
            if candidate.prompt_version:
                prompt_versions.add(candidate.prompt_version)
            case = evaluate_single_case(references[patient_id], candidate, tokenizer=tokenizer)
            rows.extend(case.rows)
            errors.extend(case.errors)
            for section in ALL_SECTIONS:
                corpus_inputs[model_id][section].append(case.candidate_texts[section])
                corpus_inputs[model_id].setdefault(f"reference:{section}", []).append(case.reference_texts[section])
    corpus_results: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    for model_id in models:
        corpus_results[model_id] = {}
        for section in ALL_SECTIONS:
            corpus_results[model_id][section] = corpus_bleu(
                corpus_inputs[model_id][f"reference:{section}"],
                corpus_inputs[model_id][section],
            )
        summaries.append(_summary(model_id, rows, corpus_results[model_id], len(patients)))
    resolved_id = evaluation_id or f"evaluation-v2-{_utc_timestamp().replace(':', '').replace('+00:00', 'Z')}"
    manifest = _manifest(
        evaluation_id=resolved_id,
        patients=patients,
        models=models,
        references=references,
        reference_hash=reference_hash,
        candidate_hash=candidate_hash,
        candidate_manifest=candidate_manifest,
        prompt_versions=sorted(prompt_versions),
    )
    return EvaluationRun(resolved_id, Path(output_root), tuple(rows), tuple(summaries), corpus_results, tuple(errors), manifest)


CASE_FIELDS = [
    "patient_id", "model_id", "section", "metrics_version", "reference_version",
    "rouge1_precision", "rouge1_recall", "rouge1_f1", "rouge2_precision", "rouge2_recall", "rouge2_f1", "rougeL_precision", "rougeL_recall", "rougeL_f1", "sentence_bleu4",
    "schema_valid", "parse_success", "invalid_candidate", "candidate_missing", "missing_section", "plan_count_violation", "missing_plan_field", "unexpected_sections", "candidate_token_count", "reference_token_count", "candidate_source_path", "normalization_version", "tokenizer_version", "candidate_text_hash", "reference_text_hash", "error_code", "error_details",
]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_evaluation_outputs(run: EvaluationRun, output_root: str | Path | None = None) -> Path:
    target = Path(output_root) if output_root is not None else run.output_root
    target.mkdir(parents=True, exist_ok=True)
    _write_csv(target / "per_case_metrics.csv", run.rows, CASE_FIELDS)
    _write_jsonl(target / "per_case_metrics.jsonl", run.rows)
    summary_fields = sorted({key for row in run.summaries for key in row.keys()})
    _write_csv(target / "per_model_summary.csv", run.summaries, summary_fields)
    _write_json(target / "corpus_bleu.json", run.corpus_bleu_results)
    _write_jsonl(target / "evaluation_errors.jsonl", [{"timestamp": _utc_timestamp(), "evaluation_id": run.evaluation_id, **error} for error in run.errors])
    _write_json(target / "metric_manifest.json", run.manifest)
    return target


__all__ = ["ALL_SECTIONS", "EvaluationRun", "evaluate_batch", "evaluate_single_case", "write_evaluation_outputs"]
