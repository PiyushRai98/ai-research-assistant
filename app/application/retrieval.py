"""Retrieval service: embeds a query and fetches relevant chunks.

Encapsulates the configured retrieval strategy (similarity or MMR), optional
per-document filtering, and a score threshold. It emits the structured
retrieval metric the brief requires and returns both the ranked chunks and the
elapsed time so callers can surface latency.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence

from app.domain.models import ScoredChunk
from app.domain.ports import EmbeddingModel, VectorStore
from app.shared.config import RetrievalStrategy, Settings
from app.shared.logging import get_logger, log_retrieval

logger = get_logger("retrieval")


class RetrievalService:
    """Coordinates embedding + vector search according to configuration."""

    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        settings: Settings,
    ) -> None:
        self._embeddings = embedding_model
        self._store = vector_store
        self._settings = settings

    def retrieve(
        self,
        query: str,
        *,
        document_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[ScoredChunk], float]:
        """Return (ranked_chunks, elapsed_ms) for a query string."""
        k = top_k or self._settings.retrieval_top_k
        started = time.perf_counter()

        query_vector = self._embeddings.embed_query(query)

        if self._settings.retrieval_strategy is RetrievalStrategy.MMR:
            results = self._store.search_mmr(
                query_vector,
                k=k,
                fetch_k=max(k * 4, k),
                lambda_mult=self._settings.retrieval_mmr_lambda,
                document_ids=document_ids,
            )
        else:
            results = self._store.search(query_vector, k=k, document_ids=document_ids)

        # Apply the configured minimum-score gate (0.0 disables it).
        threshold = self._settings.retrieval_score_threshold
        if threshold > 0.0:
            results = [r for r in results if r.score >= threshold]

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        log_retrieval(query_hash=query_hash, k=k, hits=len(results), duration_ms=elapsed_ms)
        return results, elapsed_ms
