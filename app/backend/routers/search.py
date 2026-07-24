"""Semantic / global / document-scoped search endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.backend.dependencies import OwnerDep, RetrievalServiceDep
from app.backend.schemas import SearchHit, SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse, summary="Semantic search over documents")
def search(
    request: SearchRequest,
    service: RetrievalServiceDep,
    owner: OwnerDep,  # noqa: ARG001
) -> SearchResponse:
    """Run semantic retrieval globally or scoped to specific documents."""
    results, elapsed_ms = service.retrieve(
        request.query,
        document_ids=request.document_ids,
        top_k=request.top_k,
    )
    hits = [
        SearchHit(
            text=r.chunk.text,
            score=r.score,
            document_id=r.chunk.metadata.document_id,
            document_name=r.chunk.metadata.document_name,
            page_number=r.chunk.metadata.page_number,
        )
        for r in results
    ]
    return SearchResponse(hits=hits, elapsed_ms=elapsed_ms)
