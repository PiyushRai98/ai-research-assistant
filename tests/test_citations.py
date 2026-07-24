"""Unit tests for the citation engine (no fabricated citations)."""

from __future__ import annotations

from app.application.citations import (
    extract_citations,
    infer_citations,
    resolve_citations,
)
from app.domain.models import Chunk, ChunkMetadata, ScoredChunk


def _scored(texts: list[str]) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                text=text,
                metadata=ChunkMetadata(
                    document_id="d1",
                    document_name="paper.pdf",
                    page_number=i + 1,
                    chunk_index=i,
                ),
            ),
            score=1.0 - i * 0.1,
        )
        for i, text in enumerate(texts)
    ]


def test_extract_maps_markers_to_sources() -> None:
    chunks = _scored(["attention is all you need", "recurrent models"])
    citations = extract_citations("Transformers rely on attention [1].", chunks)
    assert len(citations) == 1
    assert citations[0].marker == 1
    assert citations[0].page_number == 1
    assert "attention" in citations[0].quote


def test_out_of_range_markers_are_ignored() -> None:
    chunks = _scored(["only one chunk"])
    citations = extract_citations("Claim [1] and fake [9].", chunks)
    assert [c.marker for c in citations] == [1]


def test_duplicate_markers_collapse() -> None:
    chunks = _scored(["a", "b"])
    citations = extract_citations("See [1], again [1], and [2].", chunks)
    assert [c.marker for c in citations] == [1, 2]


def test_infer_citations_by_overlap_when_no_markers() -> None:
    chunks = _scored(["softmax attention mechanism", "gradient descent optimizer"])
    citations = infer_citations("The attention mechanism uses softmax.", chunks)
    assert citations
    assert citations[0].document_name == "paper.pdf"


def test_resolve_prefers_explicit_markers() -> None:
    chunks = _scored(["explicit source text", "other text"])
    citations = resolve_citations("Grounded claim [2].", chunks)
    assert citations[0].marker == 2


def test_no_context_yields_no_citations() -> None:
    assert resolve_citations("Anything [1].", []) == []
