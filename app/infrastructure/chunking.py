"""Recursive, citation-aware text chunking.

Design decision: chunking is implemented natively rather than delegating to
``langchain-text-splitters``. Reasons:

1. Citations require every chunk to retain an exact page number. Splitting
   *per page* and then recursively within the page guarantees that mapping,
   which a document-level splitter would blur across page boundaries.
2. Keeping this logic dependency-free makes it fully unit-testable in CI
   without installing the ML extra.

The algorithm mirrors LangChain's ``RecursiveCharacterTextSplitter``: it tries
progressively finer separators (paragraph -> line -> sentence -> word -> char)
so chunks break on natural boundaries, then applies a sliding overlap.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.models import Chunk, ChunkMetadata, PageContent

# Separators tried in order, coarsest first. Empty string is the final
# fallback (hard character split) and must remain last.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


class RecursiveChunker:
    """Concrete :class:`~app.domain.ports.Chunker`.

    Args:
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.
    """

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._size = chunk_size
        self._overlap = chunk_overlap

    def split(
        self,
        pages: Sequence[PageContent],
        *,
        document_id: str,
        document_name: str,
    ) -> list[Chunk]:
        """Split every page into overlapping chunks tagged with its page number."""
        chunks: list[Chunk] = []
        running_index = 0
        for page in pages:
            for piece in self._split_text(page.text):
                cleaned = piece.strip()
                if not cleaned:
                    continue
                chunks.append(
                    Chunk(
                        text=cleaned,
                        metadata=ChunkMetadata(
                            document_id=document_id,
                            document_name=document_name,
                            page_number=page.page_number,
                            chunk_index=running_index,
                        ),
                    )
                )
                running_index += 1
        return chunks

    # -- internal recursive splitter ---------------------------------------
    def _split_text(self, text: str) -> list[str]:
        """Return size-bounded, overlapping fragments of ``text``."""
        if not text:
            return []
        fragments = self._recursive_split(text, list(_SEPARATORS))
        return self._merge_with_overlap(fragments)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Break text on the coarsest separator that yields sub-size pieces."""
        if len(text) <= self._size:
            return [text]

        separator = separators[0]
        remaining = separators[1:] or [""]

        if separator == "":
            # Hard split on character count as the ultimate fallback.
            return [text[i : i + self._size] for i in range(0, len(text), self._size)]

        splits = text.split(separator)
        results: list[str] = []
        for split in splits:
            piece = split + separator if separator else split
            if len(piece) <= self._size:
                results.append(piece)
            else:
                results.extend(self._recursive_split(piece, remaining))
        return results

    def _merge_with_overlap(self, fragments: list[str]) -> list[str]:
        """Greedily merge small fragments up to ``chunk_size`` with overlap."""
        merged: list[str] = []
        current = ""
        for fragment in fragments:
            if not current:
                current = fragment
            elif len(current) + len(fragment) <= self._size:
                current += fragment
            else:
                merged.append(current)
                # Carry the tail of the previous chunk forward as overlap.
                tail = current[-self._overlap :] if self._overlap else ""
                current = tail + fragment
        if current:
            merged.append(current)
        return merged
