"""Citation engine.

Turns retrieved context + a generated answer into verifiable citations. Every
citation is derived from a real retrieved chunk (document, page, chunk id, and
a verbatim quote) — the engine never fabricates a source. It links the ``[n]``
markers the model emits back to the exact chunk that produced passage ``n``,
and can also infer citations by lexical overlap when a model omits markers.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.domain.models import Citation, ScoredChunk

_MARKER_RE = re.compile(r"\[(\d{1,3})\]")
# Roughly one clause; enough to let a reader locate the passage in the source.
_QUOTE_MAX_CHARS = 240


def _make_quote(text: str) -> str:
    """Produce a short, clean verbatim quote from a chunk."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _QUOTE_MAX_CHARS:
        return collapsed
    return collapsed[:_QUOTE_MAX_CHARS].rsplit(" ", 1)[0] + "…"


def _citation_from_chunk(marker: int, scored: ScoredChunk) -> Citation:
    meta = scored.chunk.metadata
    return Citation(
        marker=marker,
        document_id=meta.document_id,
        document_name=meta.document_name,
        page_number=meta.page_number,
        chunk_id=scored.chunk.id,
        quote=_make_quote(scored.chunk.text),
        score=scored.score,
    )


def extract_citations(
    answer_text: str,
    context_chunks: Sequence[ScoredChunk],
) -> list[Citation]:
    """Map ``[n]`` markers in the answer to their source chunks.

    Markers are 1-based and index into ``context_chunks`` in the same order the
    context block was rendered. Only markers that resolve to a real chunk are
    kept, guaranteeing there are no fake citations. Duplicate markers collapse
    to a single citation.
    """
    referenced = sorted(
        {
            int(match)
            for match in _MARKER_RE.findall(answer_text)
            if 1 <= int(match) <= len(context_chunks)
        }
    )
    return [_citation_from_chunk(marker, context_chunks[marker - 1]) for marker in referenced]


def infer_citations(
    answer_text: str,
    context_chunks: Sequence[ScoredChunk],
    *,
    max_citations: int = 3,
) -> list[Citation]:
    """Fallback: infer citations by lexical overlap when no markers exist.

    Used only when a model returns an answer without ``[n]`` markers. It still
    only ever cites real retrieved chunks, preserving the no-fabrication rule.
    """
    answer_terms = {w.lower() for w in re.findall(r"\w+", answer_text) if len(w) > 3}
    if not answer_terms:
        return []
    scored: list[tuple[int, int]] = []
    for index, chunk in enumerate(context_chunks):
        chunk_terms = {w.lower() for w in re.findall(r"\w+", chunk.chunk.text)}
        overlap = len(answer_terms & chunk_terms)
        if overlap:
            scored.append((overlap, index))
    scored.sort(reverse=True)
    return [
        _citation_from_chunk(marker, context_chunks[index])
        for marker, (_, index) in enumerate(scored[:max_citations], start=1)
    ]


def resolve_citations(
    answer_text: str,
    context_chunks: Sequence[ScoredChunk],
) -> list[Citation]:
    """Prefer explicit markers; fall back to inference when none are present."""
    if not context_chunks:
        return []
    explicit = extract_citations(answer_text, context_chunks)
    if explicit:
        return explicit
    return infer_citations(answer_text, context_chunks)
