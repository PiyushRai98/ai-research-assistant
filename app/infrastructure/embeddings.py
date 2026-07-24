"""Embedding model adapters.

Two implementations satisfy :class:`~app.domain.ports.EmbeddingModel`:

* :class:`SentenceTransformerEmbedding` — the production path, wrapping
  ``sentence-transformers`` (BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2). The
  model is loaded lazily on first use so importing this module is cheap.
* :class:`HashingEmbedding` — a deterministic, dependency-free fallback used
  for offline development, CI, and unit tests. It produces stable unit-norm
  vectors from token hashes; it is NOT semantically meaningful but lets the
  full pipeline (vector store, retrieval, RAG) run and be tested end-to-end.

``build_embedding_model`` selects the right adapter from configuration and
falls back to hashing if the heavy dependency is unavailable, so the app never
hard-crashes on a missing model.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from app.shared.exceptions import EmbeddingError
from app.shared.logging import get_logger

logger = get_logger("embeddings")

# bge models benefit from a query instruction prefix; applied only for queries.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _l2_normalize(vector: list[float]) -> list[float]:
    """Scale a vector to unit L2 norm (so dot product == cosine similarity)."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class HashingEmbedding:
    """Deterministic offline embedding based on hashed token buckets."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return _l2_normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class SentenceTransformerEmbedding:
    """Production embedding adapter over ``sentence-transformers``."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._is_bge = "bge" in model_name.lower()
        self._model = None  # lazy-loaded
        self._dimension = 0

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise EmbeddingError(
                "sentence-transformers is not installed; install the 'ml' extra."
            ) from exc
        try:
            logger.info("Loading embedding model {name}", name=self._model_name)
            self._model = SentenceTransformer(self._model_name, device=self._device)
            # The accessor was renamed across versions; support both names.
            get_dim = getattr(
                self._model,
                "get_embedding_dimension",
                self._model.get_sentence_embedding_dimension,
            )
            self._dimension = int(get_dim())
        except Exception as exc:
            raise EmbeddingError(
                "Failed to load embedding model.",
                details={"model": self._model_name, "error": str(exc)},
            ) from exc

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._ensure_model()
        try:
            vectors = self._model.encode(  # type: ignore[union-attr]
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return vectors.tolist()
        except Exception as exc:
            raise EmbeddingError(
                "Embedding of documents failed.", details={"error": str(exc)}
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        self._ensure_model()
        query = f"{_BGE_QUERY_PREFIX}{text}" if self._is_bge else text
        try:
            vector = self._model.encode(  # type: ignore[union-attr]
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]
            return vector.tolist()
        except Exception as exc:
            raise EmbeddingError("Embedding of query failed.", details={"error": str(exc)}) from exc


def build_embedding_model(
    *,
    model_name: str,
    device: str = "cpu",
    batch_size: int = 32,
    allow_fallback: bool = True,
) -> SentenceTransformerEmbedding | HashingEmbedding:
    """Construct the configured embedding model, degrading gracefully.

    When ``sentence-transformers`` is missing and ``allow_fallback`` is True,
    a :class:`HashingEmbedding` is returned so the system remains operational
    (e.g. in CI or offline demos). A warning is logged so the degradation is
    never silent.
    """
    try:
        import sentence_transformers  # noqa: F401  (probe availability)

        return SentenceTransformerEmbedding(model_name, device=device, batch_size=batch_size)
    except ImportError as exc:
        if not allow_fallback:
            raise EmbeddingError(
                "sentence-transformers is not installed and fallback is disabled."
            ) from exc
        logger.warning(
            "sentence-transformers unavailable; using deterministic hashing "
            "embeddings (non-semantic). Install the 'ml' extra for real search."
        )
        return HashingEmbedding()
