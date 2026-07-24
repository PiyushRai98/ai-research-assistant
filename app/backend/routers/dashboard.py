"""Dashboard statistics endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.backend.dependencies import ChatServiceDep, DocumentServiceDep, OwnerDep
from app.backend.schemas import DashboardResponse

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardResponse, summary="Aggregate usage statistics")
def dashboard(
    documents: DocumentServiceDep,
    chats: ChatServiceDep,
    owner: OwnerDep,
) -> DashboardResponse:
    """Return document, storage, embedding, and chat statistics."""
    stats = documents.statistics(owner=owner)
    return DashboardResponse(
        document_count=stats["document_count"],
        indexed_count=stats["indexed_count"],
        failed_count=stats["failed_count"],
        total_chunks=stats["total_chunks"],
        storage_bytes=stats["storage_bytes"],
        avg_processing_ms=stats["avg_processing_ms"],
        chat_count=chats.count(owner=owner),
    )
