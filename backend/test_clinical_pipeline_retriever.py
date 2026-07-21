from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock

import rag_client

from clinical_pipeline.contracts import (
    KnowledgePlan,
    KnowledgeTopic,
    RetrievalEvidence,
    RetrievalQuery,
    RetrievalStatus,
)
from clinical_pipeline.retriever import Retriever


def _settings() -> rag_client.RagClientSettings:
    return rag_client.RagClientSettings(
        mode="off",
        service_url="http://127.0.0.1:8010",
        timeout_seconds=2.0,
        top_k_per_query=2,
        max_sources=6,
        max_context_chars=8000,
        assist_approved=False,
        shadow_include_demo=False,
        allow_demo_in_prompt=False,
        trace_enabled=False,
        trace_path=Path("/unused/retriever-trace.jsonl"),
    )


def _plan(topic_count: int = 1) -> KnowledgePlan:
    topics = [
        KnowledgeTopic(
            topic_id=f"topic-{index}",
            label=f"测试主题{index}",
            finding_ids=[f"finding-{index}"],
        )
        for index in range(1, topic_count + 1)
    ]
    queries = [
        RetrievalQuery(
            query_id=f"query-{index}",
            topic_id=f"topic-{index}",
            text=f"测试检索查询{index}",
        )
        for index in range(1, topic_count + 1)
    ]
    return KnowledgePlan(
        planner_model_id="planner-test-model",
        topics=topics,
        queries=queries,
        reason="测试检索计划",
    )


def _hit(index: int = 1) -> dict:
    return {
        "rank": 1,
        "score": 0.82,
        "knowledge_id": f"KB-TEST-{index:03d}",
        "chunk_id": f"KB-TEST-{index:03d}@1#001",
        "title": f"测试证据{index}",
        "text": f"测试证据正文{index}",
        "metadata": {
            "clinical_ready": True,
            "source_ids": [f"SRC-{index:03d}"],
            "system_key": f"metric-{index}",
        },
    }


def _response(*hit_groups: list[dict]) -> dict:
    return {
        "schema_version": "rehab.rag.retrieve.v1",
        "collection": "rehab-core-test",
        "results": [
            {
                "key": f"q{index}",
                "query": f"测试检索查询{index}",
                "hits": hits,
            }
            for index, hits in enumerate(hit_groups, start=1)
        ],
    }


class RetrieverTests(unittest.TestCase):
    def test_single_query_returns_contract_evidence(self) -> None:
        transport = Mock(return_value=_response([_hit(1)]))

        result = Retriever(settings=_settings(), transport=transport).retrieve(
            _plan(),
            attempt_id="attempt-single",
        )

        self.assertEqual(result.status, RetrievalStatus.COMPLETE)
        self.assertEqual(result.request_count, 1)
        self.assertEqual(len(result.evidence), 1)
        self.assertIsInstance(result.evidence[0], RetrievalEvidence)
        self.assertEqual(result.evidence[0].query_id, "query-1")
        self.assertEqual(result.evidence[0].source_ids, ["SRC-001"])
        self.assertEqual(result.evidence[0].metadata["knowledge_id"], "KB-TEST-001")
        transport.assert_called_once()
        url, payload, timeout = transport.call_args.args
        self.assertEqual(url, "http://127.0.0.1:8010/v1/retrieve")
        self.assertNotIn("lookup", url)
        self.assertEqual(len(payload["queries"]), 1)
        self.assertFalse(payload["include_demo"])
        self.assertEqual(timeout, 2.0)

    def test_multiple_queries_use_one_batch_http_request(self) -> None:
        transport = Mock(return_value=_response([_hit(1)], [_hit(2)]))

        result = Retriever(settings=_settings(), transport=transport).retrieve(
            _plan(2),
            attempt_id="attempt-batch",
        )

        self.assertEqual(result.status, RetrievalStatus.COMPLETE)
        self.assertEqual(len(result.evidence), 2)
        transport.assert_called_once()
        url, payload, _ = transport.call_args.args
        self.assertTrue(url.endswith("/v1/retrieve"))
        self.assertEqual(
            payload["queries"],
            [
                {"key": "q1", "text": "测试检索查询1"},
                {"key": "q2", "text": "测试检索查询2"},
            ],
        )

    def test_partial_topic_coverage_returns_partial(self) -> None:
        transport = Mock(return_value=_response([_hit(1)], []))

        result = Retriever(settings=_settings(), transport=transport).retrieve(
            _plan(2),
            attempt_id="attempt-partial",
        )

        self.assertEqual(result.status, RetrievalStatus.PARTIAL)
        self.assertEqual(result.covered_topic_ids, ["topic-1"])
        self.assertEqual(result.uncovered_topic_ids, ["topic-2"])
        self.assertEqual(len(result.evidence), 1)
        transport.assert_called_once()

    def test_no_qualified_evidence_returns_insufficient(self) -> None:
        transport = Mock(return_value=_response([], []))

        result = Retriever(settings=_settings(), transport=transport).retrieve(
            _plan(2),
            attempt_id="attempt-insufficient",
        )

        self.assertEqual(result.status, RetrievalStatus.INSUFFICIENT)
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.covered_topic_ids, [])
        self.assertEqual(result.uncovered_topic_ids, ["topic-1", "topic-2"])
        transport.assert_called_once()

    def test_rag_failure_returns_unavailable_without_fabricated_evidence(self) -> None:
        transport = Mock(side_effect=TimeoutError("RAG offline"))

        result = Retriever(settings=_settings(), transport=transport).retrieve(
            _plan(2),
            attempt_id="attempt-unavailable",
        )

        self.assertEqual(result.status, RetrievalStatus.UNAVAILABLE)
        self.assertTrue(result.attempted)
        self.assertEqual(result.request_count, 1)
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.covered_topic_ids, [])
        self.assertEqual(result.uncovered_topic_ids, ["topic-1", "topic-2"])
        transport.assert_called_once()


if __name__ == "__main__":
    unittest.main()
