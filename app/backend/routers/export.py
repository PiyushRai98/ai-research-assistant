"""Export endpoints: conversation history and research artefacts."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import Response

from app.application.export import (
    conversation_to_markdown,
    markdown_to_pdf_bytes,
)
from app.backend.dependencies import ChatServiceDep, OwnerDep

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/chats/{session_id}", summary="Export a conversation")
def export_chat(
    session_id: str,
    service: ChatServiceDep,
    owner: OwnerDep,  # noqa: ARG001
    fmt: str = Query(default="markdown", description="markdown | pdf"),
) -> Response:
    """Export a full conversation (with citations) as Markdown or PDF."""
    session = service.get_session(session_id)
    markdown = conversation_to_markdown(session.messages, title=session.title)
    safe_title = session.title.replace(" ", "_")[:40] or "conversation"

    if fmt.lower() == "pdf":
        pdf = markdown_to_pdf_bytes(markdown, title=session.title)
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}.pdf"'},
        )
    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{safe_title}.md"'},
    )
