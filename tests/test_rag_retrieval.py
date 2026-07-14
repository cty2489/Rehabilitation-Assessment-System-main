from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rag.config import RagSettings
from rag.retrieval import build_index, load_chunks, retrieve
from rag.vector_store import VectorHit


class FakeEmbedder:
    dimension = 2

    def encode(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "emg" in lower or "肌电" in lower:
                vectors.append([1.0, 0.0])
            elif "eeg" in lower or "脑电" in lower:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([math.sqrt(0.5), math.sqrt(0.5)])
        return vectors


class FakeStore:
    def __init__(self):
        self.points = []

    def replace_collection(self, collection, *, dimension, points):
        self.collection = collection
        self.dimension = dimension
        self.points = list(points)
        return len(self.points)

    def search(self, collection, vector, top_k):
        scored = []
        for point in self.points:
            score = sum(a * b for a, b in zip(point.vector, vector))
            scored.append(VectorHit(score=score, payload=point.payload))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


def _chunks():
    return [
        {
            "schema_version": "rehab.knowledge.chunk.v1",
            "chunk_id": "emg#1",
            "knowledge_id": "KB-EMG",
            "entry_version": "0.1",
            "text": "EMG 肌电指标 RMS",
            "metadata": {"title": "肌电", "clinical_ready": False},
        },
        {
            "schema_version": "rehab.knowledge.chunk.v1",
            "chunk_id": "eeg#1",
            "knowledge_id": "KB-EEG",
            "entry_version": "0.1",
            "text": "EEG 脑电 mu节律",
            "metadata": {"title": "脑电", "clinical_ready": False},
        },
    ]


class RagRetrievalTests(unittest.TestCase):
    def test_settings_default_to_disabled_local_mode(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = RagSettings.from_env("/tmp/rehab")
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.backend, "local")
        self.assertEqual(settings.qdrant_path, Path("/tmp/rehab/knowledge_base/vector_store/qdrant_local"))

    def test_settings_accept_server_mode(self):
        with mock.patch.dict(
            os.environ,
            {"RAG_BACKEND": "server", "RAG_QDRANT_URL": "http://127.0.0.1:7333"},
            clear=True,
        ):
            settings = RagSettings.from_env("/tmp/rehab")
        self.assertEqual(settings.backend, "server")
        self.assertEqual(settings.qdrant_url, "http://127.0.0.1:7333")

    def test_demo_chunks_require_explicit_override(self):
        with self.assertRaisesRegex(ValueError, "--allow-demo"):
            build_index(
                _chunks(),
                embedder=FakeEmbedder(),
                store=FakeStore(),
                collection="demo",
            )

    def test_build_and_retrieve(self):
        store = FakeStore()
        summary = build_index(
            _chunks(),
            embedder=FakeEmbedder(),
            store=store,
            collection="demo",
            allow_demo=True,
        )
        results = retrieve(
            "肌电疲劳",
            embedder=FakeEmbedder(),
            store=store,
            collection="demo",
            top_k=2,
        )
        self.assertEqual(summary["indexed_chunks"], 2)
        self.assertEqual(summary["demo_chunks"], 2)
        self.assertEqual(results[0].knowledge_id, "KB-EMG")
        self.assertEqual(results[0].rank, 1)

    def test_load_chunks_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "chunks.jsonl"
            item = _chunks()[0]
            path.write_text(
                json.dumps(item, ensure_ascii=False) + "\n" + json.dumps(item, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate chunk_id"):
                load_chunks(path)

    def test_empty_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            retrieve(
                "  ",
                embedder=FakeEmbedder(),
                store=FakeStore(),
                collection="demo",
                top_k=1,
            )


if __name__ == "__main__":
    unittest.main()
