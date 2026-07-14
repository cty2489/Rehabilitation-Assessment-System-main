"""Dense embedding adapter used by the standalone retrieval experiment."""

from __future__ import annotations

from typing import Iterable, List


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        max_sequence_length: int = 1024,
        batch_size: int = 8,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; use the isolated requirements-rag.txt environment"
            ) from exc
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self._model = SentenceTransformer(
            model_name_or_path,
            device=device,
            trust_remote_code=False,
        )
        self._model.max_seq_length = max_sequence_length
        if hasattr(self._model, "get_embedding_dimension"):
            dimension = self._model.get_embedding_dimension()
        else:
            dimension = self._model.get_sentence_embedding_dimension()
        if not dimension:
            raise RuntimeError("embedding model did not report a vector dimension")
        self.dimension = int(dimension)

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        values = list(texts)
        if not values:
            return []
        vectors = self._model.encode(
            values,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()


__all__ = ["SentenceTransformerEmbedder"]
