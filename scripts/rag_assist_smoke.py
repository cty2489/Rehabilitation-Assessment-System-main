#!/usr/bin/env python3
"""Run one de-identified end-to-end RAG Assist report smoke test."""

from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import rag_client
import report
import report_builder
from schemas import PatientInfo, PredictionResult


def _biomarkers() -> dict:
    return {
        "flat": {
            "emg_activation_rms": 0.31,
            "wrist_co_contraction_index": 0.42,
            "movement_smoothness_sparc": -2.1,
        },
        "groups": [
            {
                "key": "emg",
                "label": "肌电",
                "markers": [
                    {
                        "key": "emg_activation_rms",
                        "name": "肌肉激活幅度（RMS）",
                        "value": 0.31,
                        "unit": "V(RMS)",
                    },
                    {
                        "key": "wrist_co_contraction_index",
                        "name": "腕屈伸肌共收缩指数",
                        "value": 0.42,
                        "unit": "比值[0,1]",
                    },
                ],
            },
            {
                "key": "imu",
                "label": "运动学",
                "markers": [
                    {
                        "key": "movement_smoothness_sparc",
                        "name": "运动平滑度（SPARC）",
                        "value": -2.1,
                        "unit": "",
                    }
                ],
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated RAG Assist smoke test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    settings = rag_client.RagClientSettings.from_env()
    if settings.mode != "assist":
        raise SystemExit("RAG_MODE must be assist for this smoke test")
    if not settings.assist_approved or not settings.allow_demo_in_prompt:
        raise SystemExit(
            "RAG_ASSIST_APPROVED=1 and RAG_ALLOW_DEMO_IN_PROMPT=1 are required"
        )

    patient = PatientInfo(
        patient_id="RAG-SMOKE",
        name="内部测试",
        sex="男",
        age=62,
        diagnosis="脑梗死",
        disease_days=90,
        paralysis_side="右",
    )
    predictions = PredictionResult(
        FMA_UE=16.0,
        BI=80.0,
        hand_tone="0",
        hand_function=6,
    )
    context = report_builder.build_context(
        patient,
        predictions,
        _biomarkers(),
        history=None,
        assessment_context={
            "rag_correlation_id": "rag-assist-smoke",
            "validation_status": "engineering_validation_only",
        },
    )
    context, packet = rag_client.augment_report_context(context)
    if not packet.get("used_in_prompt") or not packet.get("sources"):
        raise RuntimeError(f"RAG evidence was not eligible for Assist: {packet}")

    events: "queue.Queue[dict]" = queue.Queue()
    started = time.perf_counter()
    clinical, generation_mode = report._reason_clinical(context, events)
    markdown = report_builder.render_markdown(context, clinical)
    elapsed = round(time.perf_counter() - started, 3)
    cited_ids = sorted(
        {
            *[
                str(value).strip()
                for value in clinical.get("rag_citations", [])
                if str(value).strip()
            ],
            *re.findall(
                r"\[(KB-[A-Za-z0-9._:-]+)\]",
                json.dumps(clinical, ensure_ascii=False),
            ),
        }
    )
    source_ids = [str(source.get("knowledge_id") or "") for source in packet["sources"]]
    result = {
        "schema_version": "rehab.rag.assist-smoke.v1",
        "status": "ok",
        "generation_mode": generation_mode,
        "elapsed_seconds": elapsed,
        "collection": packet.get("collection"),
        "used_in_prompt": packet.get("used_in_prompt"),
        "source_ids": source_ids,
        "cited_ids": cited_ids,
        "citations_valid": bool(cited_ids) and set(cited_ids).issubset(source_ids),
        "trial_warning_rendered": "内部技术验证" in markdown
        and "未完成正式专家审核" in markdown,
        "overall_interpretation": clinical.get("overall_interpretation"),
        "overall_subtype": clinical.get("overall_subtype"),
    }
    if not result["citations_valid"] or not result["trial_warning_rendered"]:
        raise RuntimeError(f"RAG Assist smoke assertions failed: {result}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {**result, "clinical": clinical, "markdown": markdown},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
