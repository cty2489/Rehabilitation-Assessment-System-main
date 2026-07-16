"""Environment-backed configuration for the standalone RAG experiment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RagSettings:
    enabled: bool
    allow_demo: bool
    backend: str
    collection: str
    qdrant_path: Path
    qdrant_url: str
    embedding_model: str
    device: str
    top_k: int
    max_sequence_length: int
    batch_size: int

    @classmethod
    def from_env(cls, project_root: str | Path | None = None) -> "RagSettings":
        root = Path(project_root or Path(__file__).resolve().parents[1])
        backend = os.getenv("RAG_BACKEND", "local").strip().lower()
        if backend not in {"local", "server"}:
            raise ValueError("RAG_BACKEND must be 'local' or 'server'")
        top_k = int(os.getenv("RAG_TOP_K", "5"))
        max_sequence_length = int(os.getenv("RAG_MAX_SEQUENCE_LENGTH", "1024"))
        batch_size = int(os.getenv("RAG_BATCH_SIZE", "8"))
        if min(top_k, max_sequence_length, batch_size) <= 0:
            raise ValueError("RAG_TOP_K, RAG_MAX_SEQUENCE_LENGTH and RAG_BATCH_SIZE must be positive")
        return cls(
            enabled=_env_bool("RAG_ENABLED", False),
            allow_demo=_env_bool("RAG_ALLOW_DEMO", False),
            backend=backend,
            collection=os.getenv("RAG_COLLECTION", "rehab_knowledge_demo_v0_1").strip(),
            qdrant_path=Path(
                os.getenv(
                    "RAG_QDRANT_PATH",
                    str(root / "knowledge_base/vector_store/qdrant_local"),
                )
            ),
            qdrant_url=os.getenv("RAG_QDRANT_URL", "http://127.0.0.1:6333").strip(),
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL",
                "/root/autodl-tmp/rag_models/BAAI/bge-m3",
            ).strip(),
            device=os.getenv("RAG_DEVICE", "cpu").strip(),
            top_k=top_k,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
        )


__all__ = ["RagSettings"]
