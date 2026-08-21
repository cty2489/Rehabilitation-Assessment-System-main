"""Run the non-IMU graph prototype without model weights, RAG or a database."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from clinical_pipeline.contracts import (
    CanonicalAssessmentContext,
    CanonicalBiomarker,
    CanonicalPatientInfo,
    CanonicalPredictions,
    QualityDecision,
)
from clinical_pipeline.interpreter import Interpreter

from .graph_rag_adapter import GraphRagAdapter


_SAMPLE = Path(__file__).resolve().parent / "patient_sample_non_imu.json"


def sample_context(path: Path = _SAMPLE) -> CanonicalAssessmentContext:
    """Load the synthetic fixture using the same canonical structures as production."""
    raw: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    patient = CanonicalPatientInfo(**raw["patient"])
    predictions = CanonicalPredictions(**raw["predictions"])
    markers = []
    for group in raw["biomarkers"]["groups"]:
        for marker in group["markers"]:
            # The fixture deliberately carries an IMU row to prove exclusion.
            markers.append(
                CanonicalBiomarker(
                    metric_key=marker["key"],
                    name=marker["name"],
                    value=marker["value"] if marker["available"] else None,
                    unit=marker.get("unit"),
                    modality=group["key"],
                    available=marker["available"],
                    n_valid=marker["n_valid"],
                )
            )
    quality = raw["quality"]
    return CanonicalAssessmentContext(
        context_id="kg-demo-context",
        assessment_id="kg-demo-assessment",
        quality_decision=(
            QualityDecision.REVIEW
            if quality.get("status") == "needs_review"
            else QualityDecision.PASS
        ),
        patient=patient,
        predictions=predictions,
        biomarkers=markers,
        quality_metadata=quality,
    )


def run_demo() -> Dict[str, Any]:
    context = sample_context()
    interpretation = Interpreter().interpret(context)
    return GraphRagAdapter().build_evidence(context, interpretation)


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2))
