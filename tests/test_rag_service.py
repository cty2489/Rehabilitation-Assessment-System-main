from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from rag.config import RagSettings
from rag.vector_store import VectorHit
import rag.service as rag_service


def _settings(*, enabled: bool, allow_demo: bool = False) -> RagSettings:
    return RagSettings(
        enabled=enabled,
        allow_demo=allow_demo,
        backend="local",
        collection="test_collection",
        qdrant_path=Path("/tmp/not-opened-by-fake"),
        qdrant_url="http://127.0.0.1:6333",
        embedding_model="fake-model",
        device="cpu",
        top_k=3,
        max_sequence_length=128,
        batch_size=8,
    )


class _FakeEmbedder:
    instances = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.dimension = 2
        self.calls = []
        self.__class__.instances.append(self)

    def encode(self, texts):
        values = list(texts)
        self.calls.append(values)
        return [[1.0, 0.0] for _ in values]


class _FakeStore:
    instances = []

    def __init__(self, **_kwargs) -> None:
        self.closed = False
        self.__class__.instances.append(self)

    def search(self, _collection, _vector, _top_k):
        return [
            VectorHit(
                score=0.9,
                payload={
                    "knowledge_id": "KB-REVIEWED",
                    "chunk_id": "reviewed#1",
                    "title": "已审核条目",
                    "text": "用于同设备复测。",
                    "metadata": {"clinical_ready": True},
                },
            ),
            VectorHit(
                score=0.8,
                payload={
                    "knowledge_id": "KB-DEMO",
                    "chunk_id": "demo#1",
                    "title": "Demo 条目",
                    "text": "仅供实验。",
                    "metadata": {"clinical_ready": False},
                },
            ),
        ]

    def close(self) -> None:
        self.closed = True


@contextmanager
def _client(settings: RagSettings):
    _FakeEmbedder.instances.clear()
    _FakeStore.instances.clear()
    with mock.patch.object(
        rag_service, "SentenceTransformerEmbedder", _FakeEmbedder
    ), mock.patch.object(rag_service, "QdrantVectorStore", _FakeStore):
        app = rag_service.create_app(settings)
        with TestClient(app) as client:
            yield client


def test_disabled_service_reports_health_and_rejects_retrieval() -> None:
    with _client(_settings(enabled=False)) as client:
        health = client.get("/health")
        response = client.post(
            "/v1/retrieve",
            json={"queries": [{"key": "q", "text": "测试"}]},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "disabled"
    assert response.status_code == 503


def test_service_filters_demo_and_batches_query_embeddings() -> None:
    with _client(_settings(enabled=True)) as client:
        response = client.post(
            "/v1/retrieve",
            json={
                "queries": [
                    {"key": "scales", "text": "量表解释"},
                    {"key": "emg", "text": "肌电解释"},
                ],
                "top_k": 2,
                "include_demo": False,
            },
        )
        embedder = _FakeEmbedder.instances[0]
        store = _FakeStore.instances[0]

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "rehab.rag.retrieve.v1"
    assert body["retrieval_ms"] >= 0
    assert len(body["results"]) == 2
    assert [hit["knowledge_id"] for hit in body["results"][0]["hits"]] == ["KB-REVIEWED"]
    assert embedder.calls == [["量表解释", "肌电解释"]]
    assert store.closed


def test_demo_retrieval_requires_service_side_permission() -> None:
    with _client(_settings(enabled=True, allow_demo=False)) as client:
        denied = client.post(
            "/v1/retrieve",
            json={"queries": [{"key": "q", "text": "测试"}], "include_demo": True},
        )
    assert denied.status_code == 403

    with _client(_settings(enabled=True, allow_demo=True)) as client:
        allowed = client.post(
            "/v1/retrieve",
            json={
                "queries": [{"key": "q", "text": "测试"}],
                "top_k": 2,
                "include_demo": True,
            },
        )
    assert allowed.status_code == 200
    assert [hit["knowledge_id"] for hit in allowed.json()["results"][0]["hits"]] == [
        "KB-REVIEWED",
        "KB-DEMO",
    ]
