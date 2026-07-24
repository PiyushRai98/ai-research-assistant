"""Unit tests for the recursive, citation-aware chunker."""

from __future__ import annotations

import pytest
from app.domain.models import PageContent
from app.infrastructure.chunking import RecursiveChunker


def test_chunker_preserves_page_numbers() -> None:
    chunker = RecursiveChunker(chunk_size=120, chunk_overlap=20)
    pages = [
        PageContent(page_number=1, text="Alpha sentence. " * 20),
        PageContent(page_number=2, text="Beta sentence. " * 20),
    ]
    chunks = chunker.split(pages, document_id="d1", document_name="doc.pdf")

    assert chunks, "expected at least one chunk"
    assert {c.metadata.page_number for c in chunks} == {1, 2}
    # Chunk indices are contiguous and ordered.
    assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_size_respected_within_tolerance() -> None:
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    pages = [PageContent(page_number=1, text="word " * 200)]
    chunks = chunker.split(pages, document_id="d1", document_name="doc.pdf")
    # Overlap can push a chunk slightly over; allow the overlap as tolerance.
    assert all(len(c.text) <= 100 + 10 for c in chunks)


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError, match="overlap"):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)


def test_empty_pages_produce_no_chunks() -> None:
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.split(
        [PageContent(page_number=1, text="   ")],
        document_id="d1",
        document_name="doc.pdf",
    )
    assert chunks == []
