"""Run the isolated planner_rag v0.1 pipeline with deterministic fake services."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import rag_client  # noqa: E402
from clinical_pipeline.config import (  # noqa: E402
    CoreKnowledgeConfig,
    LlmRoleConfig,
    PipelineConfig,
)
from clinical_pipeline.contracts import (  # noqa: E402
    CanonicalBiomarker,
    CanonicalPredictions,
    CoreKnowledgeBundle,
    CoreKnowledgeEntry,
)
from clinical_pipeline.knowledge_planner import (  # noqa: E402
    KnowledgePlanner,
    PlannerMessage,
)
from clinical_pipeline.orchestrator import (  # noqa: E402
    ClinicalPipelineOrchestrator,
    ModuleExecutionStatus,
    PipelineAssessmentInput,
    PipelinePatientInput,
    PipelineRunStatus,
)
from clinical_pipeline.report_generator import (  # noqa: E402
    ReportGenerator,
    ReportMessage,
)
from clinical_pipeline.retriever import Retriever  # noqa: E402


PLANNER_MODEL_ID = "demo-planner-llm"
REPORT_MODEL_ID = "demo-report-generator-llm"


class DemoCoreKnowledgeProvider:
    def provide(self, system_keys: Sequence[str]) -> CoreKnowledgeBundle:
        return CoreKnowledgeBundle(
            bundle_id="demo-core-knowledge",
            version="demo-v0.1",
            entries=[
                CoreKnowledgeEntry(
                    knowledge_id=f"CORE-DEMO-{index:03d}",
                    system_key=system_key,
                    allowed_interpretation="仅描述输入中已有的结构化观察。",
                    prohibited_interpretation="不得作确定性诊断或给出训练处方。",
                    source_ids=[f"SRC-CORE-DEMO-{index:03d}"],
                )
                for index, system_key in enumerate(system_keys, start=1)
            ],
        )


class DemoPlannerLlm:
    model_id = PLANNER_MODEL_ID

    def generate(
        self,
        messages: Sequence[PlannerMessage],
        *,
        attempt: int,
    ) -> str:
        del messages, attempt
        return json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "topic-fma-hand",
                        "label": "FMA手部子量表解释边界",
                        "finding_ids": ["prediction:FMA_UE"],
                        "priority": "medium",
                    }
                ],
                "queries": [
                    {
                        "query_id": "query-fma-hand",
                        "topic_id": "topic-fma-hand",
                        "text": "FMA手部子量表模型预测结果的解释边界",
                    }
                ],
                "reason": "检索量表解释边界和可引用证据。",
                "generation_mode": "llm",
            },
            ensure_ascii=False,
        )


class DemoReportLlm:
    model_id = REPORT_MODEL_ID

    def generate(
        self,
        messages: Sequence[ReportMessage],
        *,
        attempt: int,
    ) -> str:
        del messages, attempt
        return json.dumps(
            {
                "summary": (
                    "量表结果来自模型预测，本报告仅描述已有结构化观察。"
                ),
                "findings": [
                    {
                        "finding_id": "prediction:FMA_UE",
                        "statement": "模型预测的FMA手部子量表结果为8分。",
                        "citations": ["SRC-DEMO-001"],
                    }
                ],
                "evidence_summary": "检索证据覆盖本次知识主题。",
                "limitations": ["模型预测仍需结合临床实测进行人工复核。"],
                "recommendations": ["建议由康复专业人员结合完整病史复核。"],
                "citations": ["SRC-DEMO-001"],
            },
            ensure_ascii=False,
        )


def fake_rag_transport(url: str, payload: dict, timeout: float) -> dict:
    del url, timeout
    query = payload["queries"][0]
    return {
        "schema_version": "rehab.rag.retrieve.v1",
        "collection": "demo-knowledge-collection",
        "results": [
            {
                "key": query["key"],
                "query": query["text"],
                "hits": [
                    {
                        "rank": 1,
                        "score": 0.91,
                        "knowledge_id": "KB-DEMO-001",
                        "chunk_id": "KB-DEMO-001@1#001",
                        "title": "FMA手部子量表解释说明",
                        "text": "量表预测结果应结合标准化临床评估解释。",
                        "metadata": {
                            "clinical_ready": True,
                            "source_ids": ["SRC-DEMO-001"],
                        },
                    }
                ],
            }
        ],
    }


def build_demo_orchestrator() -> ClinicalPipelineOrchestrator:
    settings = rag_client.RagClientSettings(
        mode="off",
        service_url="http://demo.invalid",
        timeout_seconds=1.0,
        top_k_per_query=2,
        max_sources=6,
        max_context_chars=8000,
        assist_approved=False,
        shadow_include_demo=False,
        allow_demo_in_prompt=False,
        trace_enabled=False,
        trace_path=Path("/unused/demo-rag-trace.jsonl"),
    )
    config = PipelineConfig(
        config_version="demo-planner-rag-v0.1",
        core_knowledge=CoreKnowledgeConfig(bundle_version="demo-v0.1"),
        planner=LlmRoleConfig(model_id=PLANNER_MODEL_ID),
        report_generator=LlmRoleConfig(model_id=REPORT_MODEL_ID),
    )
    return ClinicalPipelineOrchestrator(
        config=config,
        core_knowledge_provider=DemoCoreKnowledgeProvider(),
        knowledge_planner=KnowledgePlanner(DemoPlannerLlm()),
        retriever=Retriever(settings=settings, transport=fake_rag_transport),
        report_generator=ReportGenerator(DemoReportLlm()),
    )


def demo_patient() -> PipelineAssessmentInput:
    return PipelineAssessmentInput(
        assessment_id="ASSESSMENT-DEMO-001",
        patient=PipelinePatientInput(
            patient_id="PATIENT-DEMO-001",
            age=62,
            sex="男",
            diagnosis="脑梗死",
            disease_days=120,
            paralysis_side="左",
        ),
        predictions=CanonicalPredictions(
            FMA_UE=8,
            hand_tone="2",
            hand_function=3,
        ),
        biomarkers=[
            CanonicalBiomarker(
                metric_key="movement_smoothness_sparc",
                name="运动平滑度SPARC",
                value=-1.4,
                modality="imu",
                available=True,
                n_valid=1,
            )
        ],
    )


def main() -> None:
    result = build_demo_orchestrator().run(demo_patient())
    if result.status != PipelineRunStatus.COMPLETED:
        raise RuntimeError(result.model_dump_json(indent=2))

    print("=== Module execution order ===")
    started = [
        event
        for event in result.module_events
        if event.status == ModuleExecutionStatus.STARTED
    ]
    for index, event in enumerate(started, start=1):
        print(f"{index}. {event.module.value}")

    print("\n=== Planner queries ===")
    for query in result.knowledge_plan.queries:
        print(f"- [{query.query_id}] {query.text}")

    print("\n=== Retriever status ===")
    print(result.retrieval.status.value)

    print("\n=== Final report JSON ===")
    print(result.report.model_dump_json(indent=2))

    print("\n=== Validator status ===")
    print(result.validation.status.value)


if __name__ == "__main__":
    main()
