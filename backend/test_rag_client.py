from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rag_client


def _context() -> dict:
    return {
        "patient": {"patient_id": "P001", "name": "不应进入查询"},
        "predictions": {"FMA_UE": 16, "hand_tone": "1", "hand_function": 5},
        "stage": 5,
        "stage_roman": "V",
        "assessment_context": {"rag_correlation_id": "session-123"},
        "biomarkers": {
            "groups": [
                {
                    "key": "emg",
                    "label": "肌电标志物",
                    "markers": [
                        {"key": "fcr_mdf", "name": "FCR 中位频率", "available": True},
                        {"key": "missing", "name": "缺失指标", "available": False},
                    ],
                }
            ]
        },
    }


def _settings(trace_path: Path, **overrides) -> rag_client.RagClientSettings:
    values = {
        "mode": "shadow",
        "service_url": "http://127.0.0.1:8010",
        "timeout_seconds": 2.0,
        "top_k_per_query": 2,
        "max_sources": 6,
        "max_context_chars": 8000,
        "assist_approved": False,
        "shadow_include_demo": False,
        "allow_demo_in_prompt": False,
        "trace_enabled": True,
        "trace_path": trace_path,
    }
    values.update(overrides)
    return rag_client.RagClientSettings(**values)


def _response(*, clinical_ready: bool = False) -> dict:
    hit = {
        "rank": 1,
        "score": 0.82,
        "knowledge_id": "KB-EMG-001",
        "chunk_id": "KB-EMG-001@0.1#001",
        "title": "肌电解释",
        "text": "肌电中位频率用于同条件复测。",
        "metadata": {
            "clinical_ready": clinical_ready,
            "source_document_id": "doc-1",
            "source_sha256": "abc",
            "source_entry_number": 1,
            "references": ["参考资料A"],
            "reviewed_by": "专家" if clinical_ready else "",
            "reviewed_at": "2026-07-16" if clinical_ready else "",
        },
    }
    return {
        "schema_version": "rehab.rag.retrieve.v1",
        "collection": "demo",
        "results": [
            {"key": "clinical_scales", "query": "q1", "hits": [hit]},
            {"key": "emg", "query": "q2", "hits": [hit]},
        ],
    }


class RagClientTests(unittest.TestCase):
    def test_settings_default_to_off_and_local_service(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = rag_client.RagClientSettings.from_env()
        self.assertEqual(settings.mode, "off")
        self.assertEqual(settings.service_url, "http://127.0.0.1:8010")
        self.assertFalse(settings.assist_approved)
        self.assertFalse(settings.shadow_include_demo)

    def test_remote_service_url_is_rejected(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"RAG_SERVICE_URL": "https://example.com"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "localhost"):
                rag_client.RagClientSettings.from_env()

    def test_queries_are_deidentified_and_skip_unavailable_markers(self) -> None:
        queries = rag_client.build_report_queries(_context())
        text = "\n".join(item["text"] for item in queries)
        self.assertNotIn("P001", text)
        self.assertNotIn("不应进入查询", text)
        self.assertIn("FCR 中位频率", text)
        self.assertNotIn("缺失指标", text)

    def test_shadow_retrieves_demo_without_prompt_injection_and_traces(self) -> None:
        calls = []

        def transport(url, payload, timeout):
            calls.append((url, payload, timeout))
            return _response(clinical_ready=False)

        with tempfile.TemporaryDirectory() as temporary_dir:
            trace_path = Path(temporary_dir) / "trace.jsonl"
            packet = rag_client.retrieve_report_evidence(
                _context(),
                settings=_settings(trace_path, shadow_include_demo=True),
                transport=transport,
            )
            trace = json.loads(trace_path.read_text(encoding="utf-8"))

        self.assertEqual(packet["status"], "retrieved")
        self.assertEqual(packet["correlation_id"], "session-123")
        self.assertFalse(packet["used_in_prompt"])
        self.assertEqual(len(packet["sources"]), 1)
        self.assertGreaterEqual(packet["elapsed_ms"], 0)
        self.assertTrue(calls[0][1]["include_demo"])
        self.assertNotIn("text", trace["sources"][0])
        self.assertGreaterEqual(trace["elapsed_ms"], 0)
        self.assertEqual(trace["correlation_id"], "session-123")
        self.assertNotIn("P001", json.dumps(trace, ensure_ascii=False))
        self.assertNotIn("不应进入查询", json.dumps(trace, ensure_ascii=False))

    def test_shadow_does_not_request_demo_by_default(self) -> None:
        calls = []

        def transport(url, payload, timeout):
            calls.append((url, payload, timeout))
            return _response(clinical_ready=True)

        with tempfile.TemporaryDirectory() as temporary_dir:
            packet = rag_client.retrieve_report_evidence(
                _context(),
                settings=_settings(Path(temporary_dir) / "trace.jsonl"),
                transport=transport,
            )
        self.assertEqual(packet["status"], "retrieved")
        self.assertFalse(calls[0][1]["include_demo"])

    def test_assist_requires_explicit_approval(self) -> None:
        called = False

        def transport(url, payload, timeout):
            nonlocal called
            called = True
            return _response(clinical_ready=True)

        with tempfile.TemporaryDirectory() as temporary_dir:
            packet = rag_client.retrieve_report_evidence(
                _context(),
                settings=_settings(Path(temporary_dir) / "trace.jsonl", mode="assist"),
                transport=transport,
            )
        self.assertEqual(packet["status"], "assist_not_approved")
        self.assertFalse(called)

    def test_assist_uses_only_clinical_ready_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            settings = _settings(
                Path(temporary_dir) / "trace.jsonl",
                mode="assist",
                assist_approved=True,
            )
            demo = rag_client.retrieve_report_evidence(
                _context(), settings=settings, transport=lambda *_: _response(clinical_ready=False)
            )
            reviewed = rag_client.retrieve_report_evidence(
                _context(), settings=settings, transport=lambda *_: _response(clinical_ready=True)
            )
        self.assertEqual(demo["status"], "no_eligible_evidence")
        self.assertFalse(demo["used_in_prompt"])
        self.assertTrue(reviewed["used_in_prompt"])
        self.assertTrue(reviewed["sources"][0]["clinical_ready"])

    def test_service_failure_is_fail_open(self) -> None:
        def transport(*_):
            raise TimeoutError("offline")

        with tempfile.TemporaryDirectory() as temporary_dir:
            packet = rag_client.retrieve_report_evidence(
                _context(),
                settings=_settings(Path(temporary_dir) / "trace.jsonl"),
                transport=transport,
            )
        self.assertEqual(packet["status"], "service_unavailable")
        self.assertFalse(packet["used_in_prompt"])


if __name__ == "__main__":
    unittest.main()
