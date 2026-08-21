"""Append-only JSONL storage helpers for future benchmark runs.

This module is intentionally passive: it only writes records when an explicit
caller invokes one of the functions.  It is not imported by the production
assessment path and does not create or alter experiment data during startup.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Union

from .schemas import (
    BenchmarkEvaluationInput,
    BenchmarkExperimentLog,
    BenchmarkRegistryMapping,
    BlindReviewPacket,
    DoctorReviewScore,
)


JsonRecord = Union[
    BenchmarkExperimentLog,
    BenchmarkRegistryMapping,
    BlindReviewPacket,
]


def append_jsonl(path: str | Path, record: JsonRecord) -> None:
    """Append one validated record without overwriting previous records."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
        handle.write("\n")


def append_experiment_log(path: str | Path, record: BenchmarkExperimentLog) -> None:
    append_jsonl(path, record)


def append_registry_mapping(path: str | Path, record: BenchmarkRegistryMapping) -> None:
    append_jsonl(path, record)


def append_blind_review_packet(path: str | Path, record: BlindReviewPacket) -> None:
    append_jsonl(path, record)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def materialize_batch_manifest(
    root: str | Path,
    *,
    batch_id: str,
    manifest: dict,
    validation_report: dict,
) -> Path:
    """Create the required batch directory without touching old runs."""
    batch_root = Path(root) / batch_id
    batch_root.mkdir(parents=True, exist_ok=False)
    _write_json(batch_root / "manifest.json", manifest)
    _write_json(batch_root / "validation_report.json", validation_report)
    (batch_root / "patients").mkdir()
    (batch_root / "logs").mkdir()
    return batch_root


def materialize_benchmark_input(
    batch_root: str | Path,
    evaluation_input: BenchmarkEvaluationInput,
) -> Path:
    """Store fixed input facts separately from any future model output."""
    patient_root = Path(batch_root) / "patients" / evaluation_input.patient.patient_id
    patient_root.mkdir(parents=True, exist_ok=True)
    _write_json(patient_root / "input.json", evaluation_input.model_dump(mode="json"))
    _write_json(
        patient_root / "biomarkers.json",
        {
            "schema_version": "rehab.llm-benchmark-biomarkers.v1",
            "items": [marker.model_dump(mode="json") for marker in evaluation_input.biomarkers],
        },
    )
    (patient_root / "reports").mkdir(exist_ok=True)
    return patient_root


def materialize_benchmark_generation(
    batch_root: str | Path,
    result: Any,
    *,
    doctor_score: Optional[DoctorReviewScore] = None,
) -> Path:
    """Persist one generated report without replacing raw model output.

    ``result`` is kept duck-typed to avoid importing the generation service
    back into this passive storage module.  The caller must pass a validated
    ``BenchmarkGenerationResult``.
    """
    required = (
        "evaluation_input",
        "anonymous_report_id",
        "parsed_model_output",
        "report_markdown",
        "experiment_log",
        "registry_mapping",
    )
    missing = [name for name in required if not hasattr(result, name)]
    if missing:
        raise TypeError("生成结果缺少字段：" + "、".join(missing))

    evaluation_input = result.evaluation_input
    patient_root = materialize_benchmark_input(batch_root, evaluation_input)
    report_dir = patient_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_id = str(result.experiment_log.llm_model_id or "model").strip()
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._") or "model"
    report_json = report_dir / f"{model_slug}.json"
    report_md = report_dir / f"{model_slug}.md"
    _write_json(
        report_json,
        {
            "schema_version": "rehab.llm-benchmark-report.v1",
            "anonymous_report_id": result.anonymous_report_id,
            "parsed_model_output": result.parsed_model_output.model_dump(mode="json"),
            "report_markdown": result.report_markdown,
        },
    )
    report_md.write_text(result.report_markdown, encoding="utf-8")

    logs_dir = Path(batch_root) / "logs"
    append_experiment_log(logs_dir / "experiment.jsonl", result.experiment_log)
    append_registry_mapping(logs_dir / "registry.jsonl", result.registry_mapping)
    from .blind_review import build_blind_review_packet

    packet = build_blind_review_packet(
        report_id=result.anonymous_report_id,
        evaluation_input=evaluation_input,
        report_markdown=result.report_markdown,
        doctor_score=doctor_score,
    )
    append_blind_review_packet(logs_dir / "blind_review.jsonl", packet)
    return report_json


__all__ = [
    "append_blind_review_packet",
    "append_experiment_log",
    "append_jsonl",
    "append_registry_mapping",
    "materialize_batch_manifest",
    "materialize_benchmark_input",
    "materialize_benchmark_generation",
]
