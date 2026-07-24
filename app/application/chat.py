"""Chat use case: persistent, citation-grounded conversations.

Wraps :class:`RAGService` with conversation persistence. Each user turn is
answered against the session's documents using the recent history as context,
and both the question and grounded answer are stored so conversations survive
restarts and feed the dashboard's "number of chats" metric.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.application.rag import RAGService
from app.domain.models import Answer, ChatMessage, ChatSession, MessageRole
from app.domain.ports import ChatRepository
from app.shared.exceptions import NotFoundError
from app.shared.logging import get_logger

logger = get_logger("chat")


class ChatService:
    """Manages chat sessions and answers questions via RAG."""

    def __init__(self, *, repository: ChatRepository, rag_service: RAGService) -> None:
        self._repo = repository
        self._rag = rag_service

    def create_session(
        self,
        *,
        owner: str = "guest",
        document_ids: Sequence[str] | None = None,
    ) -> ChatSession:
        """Start a new conversation, optionally scoped to specific documents."""
        session = ChatSession(owner=owner, document_ids=list(document_ids or []))
        return self._repo.create(session)

    def get_session(self, session_id: str) -> ChatSession:
        session = self._repo.get(session_id)
        if session is None:
            raise NotFoundError(f"Chat session '{session_id}' was not found.")
        return session

    def list_sessions(self, *, owner: str | None = None) -> list[ChatSession]:
        return self._repo.list(owner=owner)

    def delete_session(self, session_id: str) -> None:
        self.get_session(session_id)
        self._repo.delete(session_id)

    def _history_pairs(self, session: ChatSession) -> list[tuple[str, str]]:
        """Extract (question, answer) pairs from stored messages for context."""
        pairs: list[tuple[str, str]] = []
        pending_question: str | None = None
        for message in session.messages:
            if message.role is MessageRole.USER:
                pending_question = message.content
            elif pending_question is not None:
                pairs.append((pending_question, message.content))
                pending_question = None
        return pairs

    def ask(
        self,
        *,
        session_id: str,
        question: str,
        document_ids: Sequence[str] | None = None,
    ) -> tuple[ChatSession, Answer]:
        """Answer a question within a session and persist both turns."""
        session = self.get_session(session_id)
        scope = list(document_ids) if document_ids is not None else session.document_ids

        answer = self._rag.answer(
            question,
            document_ids=scope or None,
            history=self._history_pairs(session),
        )

        session = session.with_message(ChatMessage(role=MessageRole.USER, content=question))
        session = session.with_message(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=answer.text,
                citations=answer.citations,
            )
        )
        # Persist the updated document scope alongside the messages.
        session = session.model_copy(update={"document_ids": scope})
        self._repo.save(session)
        return session, answer

    def count(self, *, owner: str | None = None) -> int:
        return self._repo.count(owner=owner)
