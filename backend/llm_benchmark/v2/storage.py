"""Versioned materialization for v2 benchmark reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from ..blind_review import build_blind_review_packet
from ..schemas import DoctorReviewScore
from ..storage import append_jsonl
from .schemas import BenchmarkEvaluationInput, OUTPUT_SCHEMA_VERSION, PROMPT_VERSION


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def materialize_benchmark_input(
    batch_root: str | Path,
    evaluation_input: BenchmarkEvaluationInput,
) -> Path:
    patient_root = Path(batch_root) / "patients" / evaluation_input.patient.patient_id
    patient_root.mkdir(parents=True, exist_ok=True)
    _write_json(patient_root / "input.json", evaluation_input.model_dump(mode="json"))
    _write_json(patient_root / "biomarkers.json", {
        "schema_version": "rehab.llm-benchmark-biomarkers.v2",
        "items": [marker.model_dump(mode="json") for marker in evaluation_input.biomarkers],
    })
    (patient_root / "reports").mkdir(exist_ok=True)
    return patient_root


def materialize_benchmark_generation(
    batch_root: str | Path,
    result: Any,
    *,
    doctor_score: Optional[DoctorReviewScore] = None,
) -> Path:
    patient_root = materialize_benchmark_input(batch_root, result.evaluation_input)
    report_dir = patient_root / "reports"
    model_id = str(result.experiment_log.llm_model_id or "model").strip()
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._") or "model"
    report_json = report_dir / f"{model_slug}.json"
    report_md = report_dir / f"{model_slug}.md"
    _write_json(report_json, {
        "schema_version": f"rehab.llm-benchmark-report.v2",
        "prompt_version": PROMPT_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "anonymous_report_id": result.anonymous_report_id,
        "parsed_model_output": result.parsed_model_output.model_dump(mode="json"),
        "report_markdown": result.report_markdown,
    })
    report_md.write_text(result.report_markdown, encoding="utf-8")
    logs_dir = Path(batch_root) / "logs"
    append_jsonl(logs_dir / "experiment_v2.jsonl", result.experiment_log)
    append_jsonl(logs_dir / "registry_v2.jsonl", result.registry_mapping)
    packet = build_blind_review_packet(
        report_id=result.anonymous_report_id,
        evaluation_input=result.evaluation_input,
        report_markdown=result.report_markdown,
        doctor_score=doctor_score,
    )
    append_jsonl(logs_dir / "blind_review_v2.jsonl", packet)
    return report_json


__all__ = ["materialize_benchmark_generation", "materialize_benchmark_input"]
