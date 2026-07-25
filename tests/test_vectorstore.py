"""Unit tests for the NumPy vector store and MMR selection."""

from __future__ import annotations

import numpy as np
from app.domain.models import Chunk, ChunkMetadata
from app.infrastructure.embeddings import HashingEmbedding
from app.infrastructure.vectorstore import NumpyVectorStore, _mmr_select


def _make_chunks(texts: list[str], doc_id: str = "d1") -> list[Chunk]:
    return [
        Chunk(
            text=text,
            metadata=ChunkMetadata(
                document_id=doc_id,
                document_name="doc.pdf",
                page_number=1,
                chunk_index=i,
            ),
        )
        for i, text in enumerate(texts)
    ]


def test_add_search_and_count(tmp_path) -> None:
    store = NumpyVectorStore(tmp_path)
    emb = HashingEmbedding(128)
    texts = ["attention transformers", "recurrent networks", "convolution images"]
    chunks = _make_chunks(texts)
    store.add(chunks, emb.embed_documents(texts))

    assert store.count() == 3
    results = store.search(emb.embed_query("transformer attention"), k=2)
    assert len(results) == 2
    assert results[0].chunk.text == "attention transformers"


def test_metadata_filtering(tmp_path) -> None:
    store = NumpyVectorStore(tmp_path)
    emb = HashingEmbedding(128)
    store.add(_make_chunks(["alpha"], "d1"), emb.embed_documents(["alpha"]))
    store.add(_make_chunks(["beta"], "d2"), emb.embed_documents(["beta"]))

    results = store.search(emb.embed_query("alpha"), k=5, document_ids=["d2"])
    assert all(r.chunk.metadata.document_id == "d2" for r in results)


def test_delete_document(tmp_path) -> None:
    store = NumpyVectorStore(tmp_path)
    emb = HashingEmbedding(128)
    store.add(_make_chunks(["a", "b"], "d1"), emb.embed_documents(["a", "b"]))
    store.add(_make_chunks(["c"], "d2"), emb.embed_documents(["c"]))

    removed = store.delete_document("d1")
    assert removed == 2
    assert store.count() == 1


def test_persistence_round_trip(tmp_path) -> None:
    emb = HashingEmbedding(128)
    store = NumpyVectorStore(tmp_path)
    store.add(_make_chunks(["persist me"]), emb.embed_documents(["persist me"]))
    store.save()

    reloaded = NumpyVectorStore(tmp_path)
    reloaded.load()
    assert reloaded.count() == 1
    assert reloaded._chunks[0].text == "persist me"


def test_numpy_store_degrades_on_foreign_backend_pickle(tmp_path) -> None:
    """A chunks.pkl saved by the FAISS backend must not crash the NumPy store.

    Both backends persist to the same filename with different payload shapes
    (FAISS: dict, NumPy: list). Switching backends between runs (e.g. FAISS
    becomes unavailable) must degrade to an empty, rebuildable index rather
    than raising — per the "never crash" requirement.
    """
    import pickle

    (tmp_path / "chunks.pkl").write_bytes(
        pickle.dumps({"next_id": 3, "chunks": {"0": {"foo": "bar"}}})
    )
    store = NumpyVectorStore(tmp_path)
    store.load()  # must not raise
    assert store.count() == 0


def test_mmr_prefers_diversity() -> None:
    # Two near-identical relevant vectors + one distinct; MMR should not pick
    # both duplicates before the distinct one.
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    candidates = np.array([[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.6, 0.8, 0.0]], dtype=np.float32)
    # lambda < 0.5 weights diversity above pure relevance, so the distinct
    # candidate (2) is preferred over the near-duplicate (1).
    chosen = _mmr_select(query, candidates, k=2, lambda_mult=0.3)
    assert 0 in chosen
    assert 2 in chosen  # the diverse candidate is chosen over the near-duplicate
