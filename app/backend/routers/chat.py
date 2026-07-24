"""Chat session and question-answering endpoints (RAG with citations)."""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.backend.dependencies import ChatServiceDep, OwnerDep
from app.backend.schemas import (
    AnswerResponse,
    AskRequest,
    AskResponse,
    ChatSessionResponse,
    ChatSessionSummary,
    CreateChatRequest,
)

router = APIRouter(prefix="/chats", tags=["chat"])


@router.post(
    "",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new chat session",
)
def create_chat(
    request: CreateChatRequest, service: ChatServiceDep, owner: OwnerDep
) -> ChatSessionResponse:
    session = service.create_session(owner=owner, document_ids=request.document_ids)
    return ChatSessionResponse.from_domain(session)


@router.get("", response_model=list[ChatSessionSummary], summary="List chat sessions")
def list_chats(service: ChatServiceDep, owner: OwnerDep) -> list[ChatSessionSummary]:
    return [
        ChatSessionSummary(
            id=s.id,
            title=s.title,
            message_count=len(s.messages),
            updated_at=s.updated_at,
        )
        for s in service.list_sessions(owner=owner)
    ]


@router.get("/{session_id}", response_model=ChatSessionResponse, summary="Get a chat")
def get_chat(
    session_id: str, service: ChatServiceDep, owner: OwnerDep  # noqa: ARG001
) -> ChatSessionResponse:
    return ChatSessionResponse.from_domain(service.get_session(session_id))


@router.post(
    "/{session_id}/ask",
    response_model=AskResponse,
    summary="Ask a question within a chat session",
)
def ask(
    session_id: str,
    request: AskRequest,
    service: ChatServiceDep,
    owner: OwnerDep,  # noqa: ARG001
) -> AskResponse:
    session, answer = service.ask(
        session_id=session_id,
        question=request.question,
        document_ids=request.document_ids,
    )
    return AskResponse(session_id=session.id, answer=AnswerResponse.from_domain(answer))


@router.post(
    "/{session_id}/ask/stream",
    summary="Ask a question and stream the answer tokens",
)
def ask_stream(
    session_id: str,
    request: AskRequest,
    service: ChatServiceDep,
    owner: OwnerDep,  # noqa: ARG001
) -> StreamingResponse:
    """Stream the answer as plain-text chunks; the turn is persisted on completion."""
    session, answer = service.ask(
        session_id=session_id,
        question=request.question,
        document_ids=request.document_ids,
    )

    def generate():  # noqa: ANN202
        # Replay the fully-formed, persisted answer as a token stream so clients
        # get a responsive UX while the stored transcript stays authoritative.
        for word in answer.text.split(" "):
            yield word + " "

    return StreamingResponse(generate(), media_type="text/plain")


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Delete a chat session",
)
def delete_chat(session_id: str, service: ChatServiceDep, owner: OwnerDep) -> None:  # noqa: ARG001
    service.delete_session(session_id)
