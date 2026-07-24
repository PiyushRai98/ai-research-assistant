"""Vector store adapters implementing :class:`~app.domain.ports.VectorStore`.

The production path is :class:`FAISSVectorStore` (FAISS ``IndexFlatIP`` over
L2-normalised vectors, so inner product equals cosine similarity). A pure-NumPy
:class:`NumpyVectorStore` provides identical semantics without the FAISS
dependency for CI, tests, and constrained environments.

Both support the capabilities the brief requires: save, load, incremental
indexing, per-document deletion, metadata filtering, similarity search, and
MMR (maximal marginal relevance) search. Vectors and chunk metadata are
persisted side by side so an index can be rehydrated across restarts.
"""

from __future__ import annotations

import json
import pickle  # nosec B403 - trusted, app-generated index sidecar only
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from app.domain.models import Chunk, ScoredChunk
from app.domain.ports import VectorStore
from app.shared.exceptions import VectorStoreError
from app.shared.logging import get_logger

logger = get_logger("vectorstore")

_VECTORS_FILE = "vectors.npy"
_CHUNKS_FILE = "chunks.pkl"
_META_FILE = "index_meta.json"


def _as_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a sequence of vectors to a contiguous float32 matrix."""
    return np.asarray(vectors, dtype=np.float32).reshape(len(vectors), -1)


def _mmr_select(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    k: int,
    lambda_mult: float,
) -> list[int]:
    """Select up to ``k`` candidate indices via maximal marginal relevance.

    Balances relevance to the query against diversity among chosen results:
        score = lambda * sim(query, c) - (1 - lambda) * max sim(c, selected)
    Assumes rows are L2-normalised so dot products are cosine similarities.
    """
    if candidates.shape[0] == 0:
        return []
    relevance = candidates @ query  # cosine similarity to the query
    selected: list[int] = []
    remaining = list(range(candidates.shape[0]))

    # Seed with the single most relevant candidate.
    best = int(np.argmax(relevance))
    selected.append(best)
    remaining.remove(best)

    while remaining and len(selected) < k:
        selected_matrix = candidates[selected]
        best_idx = -1
        best_score = -np.inf
        for idx in remaining:
            redundancy = float(np.max(candidates[idx] @ selected_matrix.T))
            score = lambda_mult * float(relevance[idx]) - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


class NumpyVectorStore(VectorStore):
    """Dependency-light vector store using NumPy for exact search."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.mkdir(parents=True, exist_ok=True)
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    # -- write --------------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunks and vectors length mismatch")
        if not chunks:
            return
        matrix = _as_matrix(vectors)
        self._vectors = matrix if self._vectors is None else np.vstack([self._vectors, matrix])
        self._chunks.extend(chunks)

    def delete_document(self, document_id: str) -> int:
        if self._vectors is None or not self._chunks:
            return 0
        keep = [
            i for i, chunk in enumerate(self._chunks) if chunk.metadata.document_id != document_id
        ]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._vectors = self._vectors[keep] if keep else None
            self._chunks = [self._chunks[i] for i in keep]
        return removed

    # -- read ---------------------------------------------------------------
    def _candidate_indices(self, document_ids: Sequence[str] | None) -> list[int]:
        if document_ids is None:
            return list(range(len(self._chunks)))
        allowed = set(document_ids)
        return [i for i, chunk in enumerate(self._chunks) if chunk.metadata.document_id in allowed]

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        indices = self._candidate_indices(document_ids)
        if self._vectors is None or not indices:
            return []
        query = np.asarray(query_vector, dtype=np.float32).ravel()
        subset = self._vectors[indices]
        scores = subset @ query
        order = np.argsort(-scores)[:k]
        return [ScoredChunk(chunk=self._chunks[indices[i]], score=float(scores[i])) for i in order]

    def search_mmr(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        fetch_k: int,
        lambda_mult: float,
        document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        indices = self._candidate_indices(document_ids)
        if self._vectors is None or not indices:
            return []
        query = np.asarray(query_vector, dtype=np.float32).ravel()
        subset = self._vectors[indices]
        scores = subset @ query
        top = np.argsort(-scores)[:fetch_k]
        chosen = _mmr_select(query, subset[top], k=k, lambda_mult=lambda_mult)
        return [
            ScoredChunk(
                chunk=self._chunks[indices[top[i]]],
                score=float(scores[top[i]]),
            )
            for i in chosen
        ]

    def count(self) -> int:
        return len(self._chunks)

    # -- persistence --------------------------------------------------------
    def save(self) -> None:
        try:
            if self._vectors is not None:
                np.save(self._path / _VECTORS_FILE, self._vectors)
            with (self._path / _CHUNKS_FILE).open("wb") as handle:
                pickle.dump([c.model_dump() for c in self._chunks], handle)
            with (self._path / _META_FILE).open("w", encoding="utf-8") as handle:
                json.dump({"count": len(self._chunks), "backend": "numpy"}, handle)
        except OSError as exc:
            raise VectorStoreError(
                "Failed to persist vector store", details={"error": str(exc)}
            ) from exc

    def load(self) -> None:
        vectors_path = self._path / _VECTORS_FILE
        chunks_path = self._path / _CHUNKS_FILE
        if not chunks_path.exists():
            return
        try:
            self._vectors = np.load(vectors_path) if vectors_path.exists() else None
            with chunks_path.open("rb") as handle:
                self._chunks = [Chunk.model_validate(d) for d in pickle.load(handle)]  # nosec B301
        except (OSError, pickle.PickleError) as exc:
            raise VectorStoreError(
                "Failed to load vector store", details={"error": str(exc)}
            ) from exc


class FAISSVectorStore(VectorStore):
    """FAISS-backed store (IndexIDMap over IndexFlatIP) with cosine similarity."""

    def __init__(self, path: Path, *, dimension: int) -> None:
        self._path = path
        self._path.mkdir(parents=True, exist_ok=True)
        self._dimension = dimension
        self._faiss = self._import_faiss()
        self._index = self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(dimension))
        self._chunks: dict[int, Chunk] = {}
        self._next_id = 0

    @staticmethod
    def _import_faiss():  # noqa: ANN205
        try:
            import faiss

            return faiss
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise VectorStoreError("faiss-cpu is not installed; install the 'ml' extra.") from exc

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunks and vectors length mismatch")
        if not chunks:
            return
        matrix = _as_matrix(vectors)
        ids = np.arange(self._next_id, self._next_id + len(chunks), dtype=np.int64)
        self._index.add_with_ids(matrix, ids)
        for local_id, chunk in zip(ids.tolist(), chunks, strict=True):
            self._chunks[local_id] = chunk
        self._next_id += len(chunks)

    def _allowed_ids(self, document_ids: Sequence[str] | None) -> set[int] | None:
        if document_ids is None:
            return None
        allowed = set(document_ids)
        return {fid for fid, chunk in self._chunks.items() if chunk.metadata.document_id in allowed}

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        if self._index.ntotal == 0:
            return []
        allowed = self._allowed_ids(document_ids)
        query = _as_matrix([query_vector])
        # Over-fetch when filtering so we can drop out-of-scope hits.
        fetch = k if allowed is None else min(self._index.ntotal, max(k * 5, k))
        scores, ids = self._index.search(query, fetch)
        results: list[ScoredChunk] = []
        for score, fid in zip(scores[0].tolist(), ids[0].tolist(), strict=True):
            if fid == -1 or (allowed is not None and fid not in allowed):
                continue
            results.append(ScoredChunk(chunk=self._chunks[fid], score=float(score)))
            if len(results) >= k:
                break
        return results

    def search_mmr(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        fetch_k: int,
        lambda_mult: float,
        document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        if self._index.ntotal == 0:
            return []
        allowed = self._allowed_ids(document_ids)
        query_row = _as_matrix([query_vector])
        query = query_row[0]
        scores, ids = self._index.search(query_row, min(self._index.ntotal, fetch_k))
        valid = [
            fid for fid in ids[0].tolist() if fid != -1 and (allowed is None or fid in allowed)
        ]
        if not valid:
            return []
        candidate_vectors = np.vstack([self._index.reconstruct(int(fid)) for fid in valid])
        chosen = _mmr_select(query, candidate_vectors, k=k, lambda_mult=lambda_mult)
        return [
            ScoredChunk(
                chunk=self._chunks[valid[i]],
                score=float(candidate_vectors[i] @ query),
            )
            for i in chosen
        ]

    def delete_document(self, document_id: str) -> int:
        to_remove = [
            fid for fid, chunk in self._chunks.items() if chunk.metadata.document_id == document_id
        ]
        if not to_remove:
            return 0
        self._index.remove_ids(np.asarray(to_remove, dtype=np.int64))
        for fid in to_remove:
            del self._chunks[fid]
        return len(to_remove)

    def count(self) -> int:
        return int(self._index.ntotal)

    def save(self) -> None:
        try:
            self._faiss.write_index(self._index, str(self._path / "index.faiss"))
            with (self._path / _CHUNKS_FILE).open("wb") as handle:
                pickle.dump(
                    {
                        "next_id": self._next_id,
                        "chunks": {k: v.model_dump() for k, v in self._chunks.items()},
                    },
                    handle,
                )
            with (self._path / _META_FILE).open("w", encoding="utf-8") as handle:
                json.dump(
                    {"count": self.count(), "backend": "faiss", "dim": self._dimension},
                    handle,
                )
        except OSError as exc:
            raise VectorStoreError(
                "Failed to persist FAISS index", details={"error": str(exc)}
            ) from exc

    def load(self) -> None:
        index_path = self._path / "index.faiss"
        chunks_path = self._path / _CHUNKS_FILE
        if not index_path.exists() or not chunks_path.exists():
            return
        try:
            self._index = self._faiss.read_index(str(index_path))
            with chunks_path.open("rb") as handle:
                payload = pickle.load(handle)  # nosec B301
            self._next_id = payload["next_id"]
            self._chunks = {int(k): Chunk.model_validate(v) for k, v in payload["chunks"].items()}
        except (OSError, pickle.PickleError, KeyError) as exc:
            raise VectorStoreError(
                "Failed to load FAISS index", details={"error": str(exc)}
            ) from exc


def build_vector_store(
    *,
    path: Path,
    dimension: int,
    prefer_faiss: bool = True,
) -> VectorStore:
    """Return a FAISS store when available, else the NumPy fallback."""
    if prefer_faiss:
        try:
            import faiss  # noqa: F401

            store: VectorStore = FAISSVectorStore(path, dimension=dimension)
            store.load()
            return store
        except (ImportError, VectorStoreError):
            logger.warning("FAISS unavailable; using NumPy vector store fallback.")
    store = NumpyVectorStore(path)
    store.load()
    return store
