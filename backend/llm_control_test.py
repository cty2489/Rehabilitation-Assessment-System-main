"""Contracts and audit metadata for the administrator-only LLM control test.

The control test keeps the normal signal-derived biomarker calculation, skips
only the three DL score models, and supplies their ground-truth values manually.
This module contains no inference, persistence, retrieval, or report code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from schemas import PatientInfo


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ManualPredictionValues(_StrictModel):
    FMA_UE: float = Field(..., ge=0.0, le=20.0)
    hand_tone: Literal["0", "1", "1+", "2", "3", "4"]
    hand_function: int = Field(..., ge=1, le=6)


class LlmControlTestSessionResponse(_StrictModel):
    session_id: str
    input_fingerprint: str
    n_trials: int = Field(ge=1)
    report_model_id: str
    biomarker_source: Literal["computed_from_uploaded_signals"] = (
        "computed_from_uploaded_signals"
    )
    persisted: Literal[False] = False
    dl_inference_skipped: Literal[True] = True


def _update_file_hash(digest: "hashlib._Hash", path: Path) -> None:
    digest.update(path.name.encode("utf-8"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def input_fingerprint(
    patient: PatientInfo,
    predictions: ManualPredictionValues,
    eeg_paths: Sequence[Path],
    emg_paths: Sequence[Path],
) -> str:
    digest = hashlib.sha256()
    canonical = json.dumps(
        {
            "patient": patient.model_dump(mode="json"),
            "manual_predictions": predictions.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(canonical)
    for kind, paths in (("eeg", eeg_paths), ("emg_imu", emg_paths)):
        digest.update(kind.encode("ascii"))
        for path in paths:
            _update_file_hash(digest, Path(path))
    return digest.hexdigest()


def quality_metadata(fingerprint: str, coverage: dict, n_trials: int) -> dict:
    return {
        "status": "llm_control_test",
        "input_source": "manual_predictions_with_signal_biomarkers",
        "prediction_source": "manual_ground_truth",
        "clinical_score_source": "clinician_provided",
        "biomarker_source": "computed_from_uploaded_signals",
        "input_fingerprint": fingerprint,
        "manual_ground_truth": True,
        "dl_inference_skipped": True,
        "biomarkers_computed_from_signal": True,
        "test_only": True,
        "persisted": False,
        "trial_count": int(n_trials),
        "short_trial_count": 0,
        "sync_fallback_count": 0,
        "sampling_rate_mismatch_count": 0,
        "biomarker_coverage": dict(coverage),
    }


__all__ = [
    "LlmControlTestSessionResponse",
    "ManualPredictionValues",
    "input_fingerprint",
    "quality_metadata",
]
