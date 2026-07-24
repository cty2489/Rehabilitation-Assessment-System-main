"""Tests for the RAG guideline test demo API endpoints.

Covers:
  - Unauthenticated access returns 401
  - Feature-disabled returns stable error
  - Status endpoint does NOT call upstream
  - Status validates loaded/enabled/allow_demo/collection
  - Out-of-scope queries blocked before upstream call
  - top_k strict validation (rejects bool/string/float/digit-string)
  - Empty query rejected
  - Normal proxy response with multiple references and dedup citations
  - Batch request body sent to upstream
  - String references normalized (source_id extracted, title cleaned, URL in doi)
  - Multiple string references across multiple hits with independent numbering
  - dataset uses upstream collection, not hardcoded
  - Upstream structural errors all return safe HTTP 503
  - Score validation (bool/NaN/Inf/string rejected)
  - All HTTP upstream calls are mocked (no real network)
"""

from __future__ import annotations

import math
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def _make_stub(name: str, attrs: dict | None = None) -> types.ModuleType:
    m = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    return m


# ---------------------------------------------------------------------------
# Module stubs for heavy dependencies
# ---------------------------------------------------------------------------

_bjh_loader_stub = _make_stub("bjh_io.bjh_loader", {
    "EEG_CHANNELS": [],
    "EEG_CHANNELS_BDF_30": [],
    "EEG_FS_DEFAULT": 250,
    "EMG_MUSCLES": [],
    "IMU_AXES_PER_MUSCLE": [],
    "load_bjh_trial": lambda *a, **kw: None,
})

_device_loader_stub = _make_stub("bjh_io.device_loader", {
    "load_device_trial": lambda *a, **kw: None,
})

_bjh_io_stub = _make_stub("bjh_io", {})
_bjh_io_stub.bjh_loader = _bjh_loader_stub
_bjh_io_stub.device_loader = _device_loader_stub

_alignment_tri_stub = _make_stub("alignment.tri_strategies", {
    "align_by_strategy_tri": lambda *a, **kw: None,
})
_alignment_wby_stub = _make_stub("alignment.wby_dtw", {
    "WBYDTWConfig": type("WBYDTWConfig", (), {}),
})
_alignment_stub = _make_stub("alignment", {})
_alignment_stub.tri_strategies = _alignment_tri_stub
_alignment_stub.wby_dtw = _alignment_wby_stub

_clinical_model_stub = _make_stub("clinical_model", {
    "ClinicalPredictionModel": type("ClinicalPredictionModel", (), {}),
})

_task_config_stub = _make_stub("task_config", {
    "ALL_TASK_NAMES": ("FMA_UE", "hand_tone", "hand_function"),
    "clip_regression": lambda *a, **kw: None,
    "get_encoder": lambda *a, **kw: None,
    "get_task": lambda *a, **kw: None,
})

_llm_settings_stub = _make_stub("llm_settings", {
    "load_settings": lambda *a, **kw: None,
})

_mysql_db_stub = _make_stub("mysql_db", {
    "init_db": lambda *a, **kw: None,
    "ensure_device_credential": lambda *a, **kw: None,
    "MySQLUnavailable": type("MySQLUnavailable", (Exception,), {}),
})

_assessment_queue_stub = _make_stub("assessment_queue", {
    "AssessmentQueue": type("AssessmentQueue", (), {
        "start": lambda self, *a: None,
        "stop": lambda self: None,
        "enqueue": lambda self, *a: type("Snapshot", (), {"queue_ahead": 0})(),
        "snapshot": lambda self, *a: None,
    }),
})

_assessment_export_stub = _make_stub("assessment_export", {
    "delete_assessment_export": lambda *a, **kw: None,
    "ensure_assessment_export": lambda *a, **kw: None,
    "export_filename": lambda *a, **kw: "export.pdf",
    "file_info": lambda *a, **kw: {},
})

_knowledge_admin_stub = _make_stub("knowledge_admin", {
    "KnowledgeUnavailable": type("KnowledgeUnavailable", (Exception,), {}),
    "coverage_payload": lambda *a, **kw: {},
    "entries_payload": lambda *a, **kw: {},
    "entry_payload": lambda *a, **kw: {},
    "sources_payload": lambda *a, **kw: {},
    "status_payload": lambda *a, **kw: {},
})

_admin_auth_stub = _make_stub("admin_auth", {
    "browser_origin_allowed": lambda *a, **kw: True,
    "issue_session_token": lambda user, token, ttl: token,
    "verify_session_token": lambda candidate, user, token: candidate == token,
})

_device_auth_stub = _make_stub("device_auth", {
    "DeviceCredential": type("DeviceCredential", (), {}),
    "DeviceTokenConfigError": type("DeviceTokenConfigError", (Exception,), {}),
    "authenticate_device_token": lambda *a, **kw: None,
    "credential_count": lambda *a, **kw: 0,
    "generate_device_token": lambda *a, **kw: "tok",
    "parse_named_tokens": lambda *a, **kw: {},
    "token_digest": lambda *a, **kw: "hash",
    "token_hint": lambda *a, **kw: "hint",
})

_eval_package_stub = _make_stub("eval_package", {
    "INSTITUTIONS": ("hospital", "device"),
    "locate_manifest_root": lambda path: path,
    "read_eval_package": lambda *a, **kw: None,
    "safe_extract_zip": lambda *a, **kw: None,
})

_device_patient_policy_stub = _make_stub("device_patient_policy", {
    "DevicePatientPolicyError": type("DevicePatientPolicyError", (Exception,), {}),
    "resolve_device_patient": lambda *a, **kw: None,
})

_inference_stub = _make_stub("inference", {
    "CHECKPOINTS": {},
    "SENTINEL": "SENTINEL",
    "AssessmentCancelled": type("AssessmentCancelled", (Exception,), {}),
    "ModelRegistry": type("ModelRegistry", (), {
        "models": {},
        "load_all": lambda self: None,
        "device": "cpu",
    }),
    "error_event": lambda *a, **kw: {},
    "run_pipeline": lambda *a, **kw: {},
})

_report_stub = _make_stub("report", {
    "REPORT_MODEL": type("REPORT_MODEL", (), {"load": lambda self: None})(),
    "llm_model_name": lambda: "stub",
    "llm_provider": lambda: "stub",
    "remote_url": lambda: "",
    "stream_report": lambda *a, **kw: "",
})

_session_events_stub = _make_stub("session_events", {
    "SessionEventStream": type("SessionEventStream", (), {}),
})

_schemas_stub = _make_stub("schemas", {})

from pydantic import BaseModel as _PydanticBase  # noqa: E402


class _FakeAssessSessionResponse(_PydanticBase):
    session_id: str = ""
    n_trials: int = 0


class _FakePatientInfo(_PydanticBase):
    patient_id: str = ""
    name: str = ""
    sex: str = "男"
    age: int | None = None
    diagnosis: str = ""
    disease_days: int | None = None
    paralysis_side: str = "左"


for _name, _cls in [
    ("AssessmentOverview", _PydanticBase),
    ("AssessmentResult", _PydanticBase),
    ("AssessSessionResponse", _FakeAssessSessionResponse),
    ("AuthLoginRequest", _PydanticBase),
    ("AuthLoginResponse", _PydanticBase),
    ("DeviceCredentialCreate", _PydanticBase),
    ("DeviceCredentialUpdate", _PydanticBase),
    ("DevicePatientRegistrationRequest", _PydanticBase),
    ("DevicePatientRegistrationResponse", _PydanticBase),
    ("EnrollmentRequest", _PydanticBase),
    ("LlmModelSettingsUpdate", _PydanticBase),
    ("LlmSettingsUpdate", _PydanticBase),
    ("MysqlAssessmentDetail", _PydanticBase),
    ("MysqlAssessmentList", _PydanticBase),
    ("PatientDetail", _PydanticBase),
    ("PatientInfo", _FakePatientInfo),
    ("PatientSummary", _PydanticBase),
    ("PatientUpdate", _PydanticBase),
    ("PredictionResult", _PydanticBase),
    ("StatsSummary", _PydanticBase),
]:
    setattr(_schemas_stub, _name, _cls)

_dotenv_stub = _make_stub("dotenv", {
    "load_dotenv": lambda *a, **kw: None,
})

_stubs = {
    "torch": _make_stub("torch"),
    "torch.nn": _make_stub("torch.nn"),
    "torch.nn.functional": _make_stub("torch.nn.functional"),
    "sentence_transformers": _make_stub("sentence_transformers"),
    "qdrant_client": _make_stub("qdrant_client"),
    "qdrant_client.models": _make_stub("qdrant_client.models"),
    "transformers": _make_stub("transformers"),
    "pandas": _make_stub("pandas"),
    "numpy": _make_stub("numpy"),
    "scipy": _make_stub("scipy"),
    "scipy.signal": _make_stub("scipy.signal"),
    "mne": _make_stub("mne"),
    "bjh_io": _bjh_io_stub,
    "bjh_io.bjh_loader": _bjh_loader_stub,
    "bjh_io.device_loader": _device_loader_stub,
    "alignment": _alignment_stub,
    "alignment.tri_strategies": _alignment_tri_stub,
    "alignment.wby_dtw": _alignment_wby_stub,
    "clinical_model": _clinical_model_stub,
    "task_config": _task_config_stub,
    "knowledge_admin": _knowledge_admin_stub,
    "admin_auth": _admin_auth_stub,
    "llm_settings": _llm_settings_stub,
    "mysql_db": _mysql_db_stub,
    "assessment_queue": _assessment_queue_stub,
    "assessment_export": _assessment_export_stub,
    "device_auth": _device_auth_stub,
    "device_patient_policy": _device_patient_policy_stub,
    "eval_package": _eval_package_stub,
    "inference": _inference_stub,
    "report": _report_stub,
    "schemas": _schemas_stub,
    "session_events": _session_events_stub,
    "dotenv": _dotenv_stub,
}

for mod_name, stub in _stubs.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = stub

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))
if str(ROOT / "Deeplearning") not in sys.path:
    sys.path.insert(0, str(ROOT / "Deeplearning"))

os.environ.setdefault("APP_AUTH_TOKEN", "test-token-for-rag-api")
os.environ.setdefault("APP_ADMIN_USER", "test")
os.environ.setdefault("APP_ADMIN_PASSWORD", "test")
os.environ.setdefault("ALLOW_LEGACY_ADMIN_BEARER", "true")
os.environ.setdefault("MYSQL_HOST", "127.0.0.1")
os.environ.setdefault("MYSQL_PORT", "3306")
os.environ.setdefault("RAG_GUIDELINE_TEST_ENABLED", "true")
os.environ.setdefault("RAG_SERVICE_URL", "http://127.0.0.1:8010")
os.environ.setdefault("RAG_GUIDELINE_TEST_SERVICE_URL", "http://127.0.0.1:8011")
os.environ.setdefault("RAG_GUIDELINE_TEST_COLLECTION", "rehab_knowledge_trial_v0_3")

from fastapi.testclient import TestClient  # noqa: E402

import backend.main as main_module  # noqa: E402
import rag_guideline_test_service as svc  # noqa: E402


AUTH_HEADERS = {"Authorization": "Bearer test-token-for-rag-api"}

EXPECTED_COLLECTION = "rehab_knowledge_trial_v0_3"


# ---------------------------------------------------------------------------
# Mock upstream RAG service responses (batch protocol, baseline health fields)
# ---------------------------------------------------------------------------

_MOCK_UPSTREAM_HEALTH_OK = {
    "status": "ok",
    "enabled": True,
    "loaded": True,
    "collection": EXPECTED_COLLECTION,
    "backend": "local",
    "allow_demo": True,
}

_MOCK_UPSTREAM_HEALTH_DISABLED = {
    "status": "disabled",
    "enabled": False,
    "loaded": False,
    "collection": EXPECTED_COLLECTION,
    "backend": "local",
    "allow_demo": True,
}

_MOCK_UPSTREAM_HEALTH_NO_DEMO = {
    "status": "ok",
    "enabled": True,
    "loaded": True,
    "collection": EXPECTED_COLLECTION,
    "backend": "local",
    "allow_demo": False,
}


def _make_retrieve_response(hits, *, collection=None, cached=False):
    """Build a batch-protocol retrieve response."""
    return {
        "schema_version": "rehab.rag.retrieve.v1",
        "collection": collection or EXPECTED_COLLECTION,
        "demo_evidence_included": True,
        "retrieval_ms": 100,
        "results": [
            {
                "key": "guideline_test",
                "query": "test",
                "hits": hits,
            },
        ],
        "cached": cached,
    }


_MOCK_UPSTREAM_RETRIEVE_OK = _make_retrieve_response([
    {
        "rank": 1,
        "score": 0.95,
        "knowledge_id": "KB-GUIDE-001",
        "chunk_id": "test-chunk-1",
        "title": "卒中康复训练原则",
        "text": "卒中后上肢康复训练应遵循循序渐进的原则。",
        "metadata": {
            "source_type": "系统评价与荟萃分析",
            "knowledge_type": "研究证据",
            "evidence_scope": "IMU 与 ICF 临床评估相关性",
            "research_type": "系统评价",
            "sample_size": "35 项研究、475 人",
            "applicable_scope": "群体研究证据检索",
            "limitations": ["不能用于患者级诊断"],
            "license": "CC BY-NC 4.0",
            "non_clinical_statement": "仅用于研究证据检索。",
            "research_only": True,
            "expert_verified": False,
            "references": [
                {
                    "source_id": "SRC-GUIDE-001",
                    "title": "中国卒中康复指南",
                    "year": "2024",
                    "doi": "https://doi.org/10.1234/test",
                    "page_locator": "第12页",
                },
            ],
        },
    },
    {
        "rank": 2,
        "score": 0.88,
        "knowledge_id": "KB-GUIDE-002",
        "chunk_id": "test-chunk-2",
        "title": "偏瘫肩关节管理",
        "text": "偏瘫患者肩关节半脱位的预防措施包括正确体位摆放。",
        "metadata": {
            "references": [
                {
                    "source_id": "SRC-GUIDE-001",
                    "title": "中国卒中康复指南",
                    "year": "2024",
                    "doi": "https://doi.org/10.1234/test",
                    "page_locator": "第15页",
                },
                {
                    "source_id": "SRC-GUIDE-002",
                    "title": "偏瘫康复实践手册",
                    "year": "2023",
                    "doi": "10.5678/practice",
                    "page_locator": "第8页",
                },
            ],
        },
    },
])

_MOCK_UPSTREAM_RETRIEVE_STRING_REFS = _make_retrieve_response([
    {
        "rank": 1,
        "score": 0.91,
        "knowledge_id": "KB-GUIDE-010",
        "chunk_id": "test-chunk-10",
        "title": "NICE Guidelines",
        "text": "Upper limb rehabilitation after stroke.",
        "metadata": {
            "references": [
                "[SRC-006] NICE NG236: Stroke rehabilitation in adults https://www.nice.org.uk/guidance/ng236",
            ],
        },
    },
])

_MOCK_UPSTREAM_MULTI_STRING_REFS = _make_retrieve_response([
    {
        "rank": 1,
        "score": 0.92,
        "knowledge_id": "KB-GUIDE-020",
        "chunk_id": "test-chunk-20",
        "title": "上肢功能训练",
        "text": "上肢功能训练应结合任务导向性训练。",
        "metadata": {
            "references": [
                "[SRC-010] AHA Stroke Guidelines 2024 https://doi.org/10.1161/STRROKEAHA.124",
                "[SRC-011] NICE NG236 https://www.nice.org.uk/guidance/ng236",
            ],
        },
    },
    {
        "rank": 2,
        "score": 0.85,
        "knowledge_id": "KB-GUIDE-021",
        "chunk_id": "test-chunk-21",
        "title": "肩关节管理",
        "text": "肩关节半脱位的预防包括正确体位摆放。",
        "metadata": {
            "references": [
                "[SRC-010] AHA Stroke Guidelines 2024 https://doi.org/10.1161/STRROKEAHA.124",
                "[SRC-012] 中国康复指南 https://rehab.example.cn/guide",
            ],
        },
    },
])


def _mock_httpx_client(mock_status=200, mock_json=None, side_effect=None):
    """Create a mock httpx.AsyncClient that returns specified responses."""
    mock_client = mock.AsyncMock()

    if side_effect:
        mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mock.AsyncMock(return_value=False)
        mock_client.get = mock.AsyncMock(side_effect=side_effect)
        mock_client.post = mock.AsyncMock(side_effect=side_effect)
        return mock_client

    mock_response = mock.MagicMock()
    mock_response.status_code = mock_status
    mock_response.json.return_value = mock_json or {}

    mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mock.AsyncMock(return_value=False)
    mock_client.get = mock.AsyncMock(return_value=mock_response)
    mock_client.post = mock.AsyncMock(return_value=mock_response)
    return mock_client


class RagGuidelineTestAPITests(unittest.TestCase):
    """Tests for /api/rag/guidelines/test/* endpoints."""

    def setUp(self):
        self.client = TestClient(main_module.app, raise_server_exceptions=False)

    # ------------------------------------------------------------------ #
    # Unauthenticated access
    # ------------------------------------------------------------------ #
    def test_status_requires_auth(self):
        resp = self.client.get("/api/rag/guidelines/test/status")
        self.assertEqual(resp.status_code, 401)

    def test_search_requires_auth(self):
        resp = self.client.post(
            "/api/rag/guidelines/test/search",
            json={"query": "test"},
        )
        self.assertEqual(resp.status_code, 401)

    # ------------------------------------------------------------------ #
    # Feature disabled
    # ------------------------------------------------------------------ #
    def test_status_returns_disabled_when_env_false(self):
        original = svc.RAG_GUIDELINE_TEST_ENABLED
        svc.RAG_GUIDELINE_TEST_ENABLED = False
        try:
            resp = self.client.get("/api/rag/guidelines/test/status", headers=AUTH_HEADERS)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertFalse(data["enabled"])
        finally:
            svc.RAG_GUIDELINE_TEST_ENABLED = original

    def test_search_disabled_returns_404(self):
        original = svc.RAG_GUIDELINE_TEST_ENABLED
        svc.RAG_GUIDELINE_TEST_ENABLED = False
        try:
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query"},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 404)
            self.assertIn("尚未启用", resp.json()["detail"])
        finally:
            svc.RAG_GUIDELINE_TEST_ENABLED = original

    # ------------------------------------------------------------------ #
    # Status does NOT call upstream
    # ------------------------------------------------------------------ #
    def test_status_does_not_call_upstream_when_disabled(self):
        original = svc.RAG_GUIDELINE_TEST_ENABLED
        svc.RAG_GUIDELINE_TEST_ENABLED = False
        try:
            with mock.patch("rag_guideline_test_service.httpx.AsyncClient") as mock_cls:
                resp = self.client.get("/api/rag/guidelines/test/status", headers=AUTH_HEADERS)
                self.assertEqual(resp.status_code, 200)
                mock_cls.assert_not_called()
                data = resp.json()
                self.assertFalse(data["enabled"])
        finally:
            svc.RAG_GUIDELINE_TEST_ENABLED = original

    # ------------------------------------------------------------------ #
    # Status validates loaded/enabled/allow_demo/collection
    # ------------------------------------------------------------------ #
    def test_status_calls_health_endpoint(self):
        original = svc.RAG_GUIDELINE_TEST_ENABLED
        svc.RAG_GUIDELINE_TEST_ENABLED = True
        try:
            mock_client = _mock_httpx_client(mock_status=200, mock_json=_MOCK_UPSTREAM_HEALTH_OK)
            with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
                resp = self.client.get("/api/rag/guidelines/test/status", headers=AUTH_HEADERS)
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["mode"], "test_only")
                self.assertEqual(data["allowed_rag_mode"], "test_only")
                self.assertTrue(data["enabled"])
                self.assertTrue(data["service_reachable"])
                self.assertTrue(data["allow_demo"])
                self.assertFalse(data["clinical_ready"])
                self.assertEqual(data["collection"], EXPECTED_COLLECTION)
                mock_client.get.assert_called_once_with(
                    f"{svc.RAG_GUIDELINE_TEST_SERVICE_URL}/health"
                )
        finally:
            svc.RAG_GUIDELINE_TEST_ENABLED = original

    def test_status_health_200_but_disabled(self):
        original = svc.RAG_GUIDELINE_TEST_ENABLED
        svc.RAG_GUIDELINE_TEST_ENABLED = True
        try:
            mock_client = _mock_httpx_client(mock_status=200, mock_json=_MOCK_UPSTREAM_HEALTH_DISABLED)
            with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
                resp = self.client.get("/api/rag/guidelines/test/status", headers=AUTH_HEADERS)
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertTrue(data["enabled"])
                self.assertFalse(data["service_reachable"])
                self.assertIn("未就绪", data["error"])
        finally:
            svc.RAG_GUIDELINE_TEST_ENABLED = original

    def test_status_health_status_not_ok_even_when_flags_true(self):
        original = svc.RAG_GUIDELINE_TEST_ENABLED
        svc.RAG_GUIDELINE_TEST_ENABLED = True
        try:
            bad_health = dict(_MOCK_UPSTREAM_HEALTH_OK, status="disabled")
            mock_client = _mock_httpx_client(mock_status=200, mock_json=bad_health)
            with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
                resp = self.client.get("/api/rag/guidelines/test/status", headers=AUTH_HEADERS)
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertFalse(data["service_reachable"])
                self.assertIn("未就绪", data["error"])
        finally:
            svc.RAG_GUIDELINE_TEST_ENABLED = original

    def test_status_health_200_but_allow_demo_false(self):
        original = svc.RAG_GUIDELINE_TEST_ENABLED
        svc.RAG_GUIDELINE_TEST_ENABLED = True
        try:
            mock_client = _mock_httpx_client(mock_status=200, mock_json=_MOCK_UPSTREAM_HEALTH_NO_DEMO)
            with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
                resp = self.client.get("/api/rag/guidelines/test/status", headers=AUTH_HEADERS)
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertTrue(data["enabled"])
                self.assertFalse(data["service_reachable"])
                self.assertFalse(data["allow_demo"])
                self.assertIn("演示", data["error"])
        finally:
            svc.RAG_GUIDELINE_TEST_ENABLED = original

    def test_status_health_collection_mismatch(self):
        original = svc.RAG_GUIDELINE_TEST_ENABLED
        svc.RAG_GUIDELINE_TEST_ENABLED = True
        try:
            bad_health = {
                "status": "ok",
                "enabled": True,
                "loaded": True,
                "collection": "wrong_collection",
                "backend": "local",
                "allow_demo": True,
            }
            mock_client = _mock_httpx_client(mock_status=200, mock_json=bad_health)
            with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
                resp = self.client.get("/api/rag/guidelines/test/status", headers=AUTH_HEADERS)
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertFalse(data["service_reachable"])
                self.assertIn("集合配置不匹配", data["error"])
        finally:
            svc.RAG_GUIDELINE_TEST_ENABLED = original

    def test_status_when_upstream_unreachable(self):
        import httpx as real_httpx
        original = svc.RAG_GUIDELINE_TEST_ENABLED
        svc.RAG_GUIDELINE_TEST_ENABLED = True
        try:
            mock_client = _mock_httpx_client(
                side_effect=real_httpx.ConnectError("connection refused")
            )
            with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
                resp = self.client.get("/api/rag/guidelines/test/status", headers=AUTH_HEADERS)
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertTrue(data["enabled"])
                self.assertFalse(data["service_reachable"])
                self.assertIn("不可达", data["error"])
        finally:
            svc.RAG_GUIDELINE_TEST_ENABLED = original

    # ------------------------------------------------------------------ #
    # Scope guard blocks before upstream call
    # ------------------------------------------------------------------ #
    def test_out_of_scope_blocked_before_upstream(self):
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient") as mock_cls:
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "EMG 肌电异常阈值是多少"},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["reason_code"], "numeric_reference_out_of_scope")
            self.assertEqual(data["results"], [])
            self.assertIn("正常范围", data["blocked_message"])
            mock_cls.assert_not_called()

    def test_blocked_query_does_not_access_upstream(self):
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient") as mock_cls:
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "EMG 阈值异常"},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 200)
            mock_cls.assert_not_called()

    # ------------------------------------------------------------------ #
    # top_k strict validation
    # ------------------------------------------------------------------ #
    def test_top_k_too_low(self):
        resp = self.client.post(
            "/api/rag/guidelines/test/search",
            json={"query": "test query", "top_k": 0},
            headers=AUTH_HEADERS,
        )
        self.assertEqual(resp.status_code, 422)

    def test_top_k_too_high(self):
        resp = self.client.post(
            "/api/rag/guidelines/test/search",
            json={"query": "test query", "top_k": 10},
            headers=AUTH_HEADERS,
        )
        self.assertEqual(resp.status_code, 422)

    def test_top_k_string_rejected(self):
        resp = self.client.post(
            "/api/rag/guidelines/test/search",
            json={"query": "test query", "top_k": "abc"},
            headers=AUTH_HEADERS,
        )
        self.assertEqual(resp.status_code, 422)

    def test_top_k_string_digit_rejected(self):
        resp = self.client.post(
            "/api/rag/guidelines/test/search",
            json={"query": "test query", "top_k": "3"},
            headers=AUTH_HEADERS,
        )
        self.assertEqual(resp.status_code, 422)

    def test_top_k_bool_rejected(self):
        resp = self.client.post(
            "/api/rag/guidelines/test/search",
            json={"query": "test query", "top_k": True},
            headers=AUTH_HEADERS,
        )
        self.assertEqual(resp.status_code, 422)

    def test_top_k_float_rejected(self):
        resp = self.client.post(
            "/api/rag/guidelines/test/search",
            json={"query": "test query", "top_k": 3.5},
            headers=AUTH_HEADERS,
        )
        self.assertEqual(resp.status_code, 422)

    # ------------------------------------------------------------------ #
    # Empty / too-long query validation
    # ------------------------------------------------------------------ #
    def test_empty_query_rejected(self):
        resp = self.client.post(
            "/api/rag/guidelines/test/search",
            json={"query": "   "},
            headers=AUTH_HEADERS,
        )
        self.assertEqual(resp.status_code, 422)

    def test_query_too_long_rejected(self):
        resp = self.client.post(
            "/api/rag/guidelines/test/search",
            json={"query": "x" * 2001},
            headers=AUTH_HEADERS,
        )
        self.assertEqual(resp.status_code, 422)

    # ------------------------------------------------------------------ #
    # Batch request body sent to upstream
    # ------------------------------------------------------------------ #
    def test_batch_request_body_sent_to_upstream(self):
        mock_client = _mock_httpx_client(
            mock_status=200, mock_json=_MOCK_UPSTREAM_RETRIEVE_OK
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "stroke rehabilitation guideline", "top_k": 2},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 200)
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            self.assertEqual(
                call_args[0][0],
                f"{svc.RAG_GUIDELINE_TEST_SERVICE_URL}/v1/retrieve",
            )
            self.assertNotEqual(
                svc.RAG_GUIDELINE_TEST_SERVICE_URL,
                os.environ["RAG_SERVICE_URL"],
            )
            body = call_args[1]["json"]
            self.assertIn("queries", body)
            self.assertEqual(len(body["queries"]), 1)
            self.assertEqual(body["queries"][0]["key"], "guideline_test")
            self.assertEqual(body["queries"][0]["text"], "stroke rehabilitation guideline")
            self.assertTrue(body["include_demo"])
            self.assertNotIn("collection", body, "caller must not specify collection")

    # ------------------------------------------------------------------ #
    # dataset uses upstream collection, not hardcoded
    # ------------------------------------------------------------------ #
    def test_dataset_uses_upstream_collection(self):
        mock_client = _mock_httpx_client(
            mock_status=200, mock_json=_MOCK_UPSTREAM_RETRIEVE_OK
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "stroke rehabilitation guideline", "top_k": 2},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["dataset"], EXPECTED_COLLECTION)
            self.assertNotEqual(data["dataset"], "rehab_guidelines_test_v0_1")

    def test_search_rejects_unexpected_collection(self):
        mock_client = _mock_httpx_client(
            mock_status=200,
            mock_json=_make_retrieve_response([], collection="production_collection"),
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "stroke rehabilitation guideline", "top_k": 2},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)
            self.assertIn("数据格式异常", resp.json()["detail"])

    def test_search_requires_demo_evidence_confirmation(self):
        upstream = _make_retrieve_response([])
        upstream["demo_evidence_included"] = False
        mock_client = _mock_httpx_client(mock_status=200, mock_json=upstream)
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "stroke rehabilitation guideline", "top_k": 2},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    # ------------------------------------------------------------------ #
    # Nested hits structure from batch response
    # ------------------------------------------------------------------ #
    def test_search_returns_correct_fields_with_citations(self):
        mock_client = _mock_httpx_client(
            mock_status=200, mock_json=_MOCK_UPSTREAM_RETRIEVE_OK
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "stroke rehabilitation guideline", "top_k": 2},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()

            for key in [
                "schema_version", "mode", "allowed_rag_mode", "test_report_banner", "query",
                "top_k", "dataset", "clinical_ready", "results", "cached",
                "elapsed_ms", "citations", "reason_code",
            ]:
                self.assertIn(key, data, f"missing key: {key}")

            self.assertEqual(data["mode"], "test_only")
            self.assertEqual(data["allowed_rag_mode"], "test_only")
            self.assertFalse(data["clinical_ready"])
            self.assertEqual(data["reason_code"], "in_scope")
            self.assertEqual(len(data["results"]), 2)

            self.assertEqual(len(data["citations"]), 2)
            self.assertEqual(data["citations"][0]["source_id"], "SRC-GUIDE-001")
            self.assertEqual(data["citations"][0]["index"], 1)
            self.assertEqual(data["citations"][1]["source_id"], "SRC-GUIDE-002")
            self.assertEqual(data["citations"][1]["index"], 2)

            hit1 = data["results"][0]
            self.assertEqual(hit1["chunk_id"], "test-chunk-1")
            self.assertEqual(len(hit1["references"]), 1)
            self.assertEqual(hit1["citation_indices"], [1])
            self.assertIn("chunk_id", hit1)
            self.assertIn("references", hit1)
            self.assertEqual(hit1["source_type"], "系统评价与荟萃分析")
            self.assertEqual(hit1["knowledge_type"], "研究证据")
            self.assertEqual(hit1["evidence_scope"], "IMU 与 ICF 临床评估相关性")
            self.assertEqual(hit1["research_type"], "系统评价")
            self.assertEqual(hit1["sample_size"], "35 项研究、475 人")
            self.assertEqual(hit1["applicable_scope"], "群体研究证据检索")
            self.assertEqual(hit1["limitations"], ["不能用于患者级诊断"])
            self.assertEqual(hit1["license"], "CC BY-NC 4.0")
            self.assertEqual(hit1["non_clinical_statement"], "仅用于研究证据检索。")
            self.assertTrue(hit1["research_only"])
            self.assertFalse(hit1["expert_verified"])

            hit2 = data["results"][1]
            self.assertEqual(hit2["chunk_id"], "test-chunk-2")
            self.assertEqual(len(hit2["references"]), 2)
            self.assertEqual(hit2["citation_indices"], [1, 2])
            self.assertEqual(hit2["source_type"], "")
            self.assertEqual(hit2["limitations"], [])
            self.assertFalse(hit2["research_only"])
            self.assertFalse(hit2["expert_verified"])

    # ------------------------------------------------------------------ #
    # String references normalized
    # ------------------------------------------------------------------ #
    def test_string_references_normalized(self):
        mock_client = _mock_httpx_client(
            mock_status=200, mock_json=_MOCK_UPSTREAM_RETRIEVE_STRING_REFS
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "stroke rehabilitation guideline", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(len(data["results"]), 1)
            hit = data["results"][0]
            self.assertEqual(len(hit["references"]), 1)
            ref = hit["references"][0]
            self.assertIsInstance(ref, dict, "reference must be a dict, not a string")
            self.assertEqual(ref["source_id"], "SRC-006")
            self.assertIn("NICE NG236", ref["title"])
            self.assertNotIn("[SRC-006]", ref["title"], "prefix should be stripped from title")
            self.assertNotIn("https://", ref["title"], "URL should be stripped from title")
            self.assertIn("raw_text", ref)
            self.assertIn("https://www.nice.org.uk/guidance/ng236", ref["raw_text"])
            self.assertIn("doi", ref, "URL should be extracted into doi field")
            self.assertEqual(ref["doi"], "https://www.nice.org.uk/guidance/ng236")

    # ------------------------------------------------------------------ #
    # Multiple string references, multiple hits independent numbering
    # ------------------------------------------------------------------ #
    def test_multi_string_refs_multi_hits_independent_numbering(self):
        mock_client = _mock_httpx_client(
            mock_status=200, mock_json=_MOCK_UPSTREAM_MULTI_STRING_REFS
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "stroke rehabilitation guideline", "top_k": 2},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(len(data["results"]), 2)
            self.assertEqual(len(data["citations"]), 3)

            hit1 = data["results"][0]
            self.assertEqual(len(hit1["references"]), 2)
            self.assertEqual(hit1["references"][0]["source_id"], "SRC-010")
            self.assertEqual(hit1["references"][1]["source_id"], "SRC-011")

            hit2 = data["results"][1]
            self.assertEqual(len(hit2["references"]), 2)
            self.assertEqual(hit2["references"][0]["source_id"], "SRC-010")
            self.assertEqual(hit2["references"][1]["source_id"], "SRC-012")

            citation_ids = [c["source_id"] for c in data["citations"]]
            self.assertEqual(citation_ids, ["SRC-010", "SRC-011", "SRC-012"])

            self.assertEqual(hit1["citation_index"], 1)
            self.assertEqual(hit2["citation_index"], 1)
            self.assertEqual(hit1["citation_indices"], [1, 2])
            self.assertEqual(hit2["citation_indices"], [1, 3])

    # ------------------------------------------------------------------ #
    # Upstream structural errors all return safe HTTP 503
    # ------------------------------------------------------------------ #
    def test_upstream_timeout_returns_503(self):
        import httpx as real_httpx
        mock_client = _mock_httpx_client(side_effect=real_httpx.TimeoutException("timeout"))
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)
            data = resp.json()
            self.assertIn("超时", data["detail"])
            self.assertNotIn("Traceback", str(data))
            self.assertNotIn("/root/", str(data))

    def test_upstream_connect_error_returns_503(self):
        import httpx as real_httpx
        mock_client = _mock_httpx_client(
            side_effect=real_httpx.ConnectError("connection refused")
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)
            data = resp.json()
            self.assertIn("不可用", data["detail"])

    def test_upstream_500_returns_503(self):
        mock_client = _mock_httpx_client(mock_status=500, mock_json={"detail": "Internal Server Error"})
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)
            data = resp.json()
            self.assertNotIn("Internal Server Error", data.get("detail", ""))

    def test_upstream_invalid_json_returns_503(self):
        mock_client = mock.AsyncMock()
        mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mock.AsyncMock(return_value=False)
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("No JSON")
        mock_client.post = mock.AsyncMock(return_value=mock_response)
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_root_is_list_returns_503(self):
        mock_client = _mock_httpx_client(mock_status=200, mock_json=[1, 2, 3])
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_missing_results_returns_503(self):
        mock_client = _mock_httpx_client(mock_status=200, mock_json={"results": "not_a_list"})
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_results_empty_list_returns_503(self):
        mock_client = _mock_httpx_client(mock_status=200, mock_json={"results": []})
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_results0_not_dict_returns_503(self):
        mock_client = _mock_httpx_client(
            mock_status=200, mock_json={"results": ["not_a_dict"]}
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_batch_key_wrong_returns_503(self):
        mock_client = _mock_httpx_client(
            mock_status=200,
            mock_json={"results": [{"key": "wrong_key", "hits": []}]},
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_missing_hits_returns_503(self):
        mock_client = _mock_httpx_client(
            mock_status=200,
            mock_json={"results": [{"key": "guideline_test", "hits": "not_a_list"}]},
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_hit_not_dict_returns_503(self):
        mock_client = _mock_httpx_client(
            mock_status=200,
            mock_json=_make_retrieve_response(["not_a_dict"]),
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_metadata_not_dict_returns_503(self):
        mock_client = _mock_httpx_client(
            mock_status=200,
            mock_json=_make_retrieve_response([{
                "rank": 1, "score": 0.5, "text": "t",
                "metadata": "not_a_dict",
            }]),
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_score_string_returns_503(self):
        mock_client = _mock_httpx_client(
            mock_status=200,
            mock_json=_make_retrieve_response([{
                "rank": 1, "score": "0.5", "text": "t",
            }]),
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_score_bool_returns_503(self):
        mock_client = _mock_httpx_client(
            mock_status=200,
            mock_json=_make_retrieve_response([{
                "rank": 1, "score": True, "text": "t",
            }]),
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_score_nan_returns_503(self):
        mock_client = _mock_httpx_client(
            mock_status=200,
            mock_json=_make_retrieve_response([{
                "rank": 1, "score": float("nan"), "text": "t",
            }]),
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_score_inf_returns_503(self):
        mock_client = _mock_httpx_client(
            mock_status=200,
            mock_json=_make_retrieve_response([{
                "rank": 1, "score": float("inf"), "text": "t",
            }]),
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_score_none_returns_503(self):
        mock_client = _mock_httpx_client(
            mock_status=200,
            mock_json=_make_retrieve_response([{
                "rank": 1, "score": None, "text": "t",
            }]),
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_score_list_returns_503(self):
        mock_client = _mock_httpx_client(
            mock_status=200,
            mock_json=_make_retrieve_response([{
                "rank": 1, "score": [0.5], "text": "t",
            }]),
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_invalid_reference_type_returns_503(self):
        mock_client = _mock_httpx_client(
            mock_status=200,
            mock_json=_make_retrieve_response([{
                "rank": 1,
                "score": 0.8,
                "text": "t",
                "metadata": {"references": [123]},
            }]),
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_missing_collection_returns_503(self):
        mock_client = _mock_httpx_client(
            mock_status=200,
            mock_json={"results": [{"key": "guideline_test", "hits": []}]},
        )
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)

    def test_upstream_generic_exception_returns_503(self):
        mock_client = _mock_httpx_client(side_effect=RuntimeError("something broke"))
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 503)
            data = resp.json()
            self.assertNotIn("something broke", data.get("detail", ""))
            self.assertNotIn("RuntimeError", data.get("detail", ""))

    # ------------------------------------------------------------------ #
    # Hit title priority
    # ------------------------------------------------------------------ #
    def test_hit_title_uses_hit_title_over_ref_title(self):
        mock_data = _make_retrieve_response([{
            "rank": 1, "score": 0.9, "text": "t",
            "title": "知识条目标题",
            "metadata": {
                "references": [
                    {"source_id": "SRC-001", "title": "参考文献标题", "year": "2024"},
                ],
            },
        }])
        mock_client = _mock_httpx_client(mock_status=200, mock_json=mock_data)
        with mock.patch("rag_guideline_test_service.httpx.AsyncClient", return_value=mock_client):
            resp = self.client.post(
                "/api/rag/guidelines/test/search",
                json={"query": "test query", "top_k": 1},
                headers=AUTH_HEADERS,
            )
            self.assertEqual(resp.status_code, 200)
            hit = resp.json()["results"][0]
            self.assertEqual(hit["title"], "知识条目标题")


if __name__ == "__main__":
    unittest.main()
