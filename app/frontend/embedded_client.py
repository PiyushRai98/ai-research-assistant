"""In-process client used when no remote backend API is reachable.

Streamlit Community Cloud runs exactly one process, so the FastAPI backend has
nowhere to live alongside the UI. Rather than making the app unusable on that
platform, this client builds the same application services
(:mod:`app.backend.container`) directly in the Streamlit process and exposes
the identical interface as :class:`~app.frontend.api_client.APIClient`. Views
call this interchangeably with the HTTP client, so no view code needs to know
which mode is active.

The same domain rules, citation guarantees, and offline-graceful adapters
(hashing embeddings, echo LLM, NumPy vector store) apply either way — only the
transport differs. When a real backend URL is configured and reachable, that
one is preferred and this module is never used.
"""

from __future__ import annotations

from typing import Any

from app import __version__
from app.application.ai_features import format_citation
from app.application.export import conversation_to_markdown, markdown_to_pdf_bytes
from app.backend.container import Container, build_container
from app.backend.schemas import (
    AnswerResponse,
    ChatSessionResponse,
    CitationFormatResponse,
    DashboardResponse,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
)
from app.frontend.api_client import APIError
from app.shared.exceptions import ResearchAssistantError

_OWNER = "guest"


def _wrap(fn, *args, **kwargs):  # noqa: ANN001, ANN202
    """Call an application-layer method, translating domain errors to APIError."""
    try:
        return fn(*args, **kwargs)
    except ResearchAssistantError as exc:
        raise APIError(exc.message, code=exc.code) from exc


class EmbeddedClient:
    """Talks directly to the in-process application services (no HTTP)."""

    def __init__(self) -> None:
        self._container: Container = build_container()

    def set_token(self, token: str | None) -> None:  # noqa: ARG002
        """No-op: auth tokens are irrelevant for the embedded, single-user mode."""

    # -- health & dashboard -------------------------------------------------
    def health(self) -> dict[str, Any]:
        settings = self._container.settings
        return HealthResponse(
            status="ok",
            version=__version__,
            environment=settings.app_env.value,
            embedding_model=settings.embedding_model,
            llm=settings.llm_provider.value,
        ).model_dump()

    def dashboard(self) -> dict[str, Any]:
        stats = _wrap(self._container.document_service.statistics, owner=_OWNER)
        return DashboardResponse(
            document_count=stats["document_count"],
            indexed_count=stats["indexed_count"],
            failed_count=stats["failed_count"],
            total_chunks=stats["total_chunks"],
            storage_bytes=stats["storage_bytes"],
            avg_processing_ms=stats["avg_processing_ms"],
            chat_count=self._container.chat_service.count(owner=_OWNER),
        ).model_dump()

    # -- documents ------------------------------------------------------------
    def upload_document(self, *, filename: str, data: bytes) -> dict[str, Any]:
        document = _wrap(
            self._container.document_service.upload,
            data=data,
            filename=filename,
            content_type="application/pdf",
            owner=_OWNER,
        )
        return DocumentResponse.from_domain(document).model_dump()

    def list_documents(self) -> dict[str, Any]:
        documents = _wrap(self._container.document_service.list, owner=_OWNER)
        return DocumentListResponse(
            documents=[DocumentResponse.from_domain(d) for d in documents],
            total=len(documents),
        ).model_dump()

    def delete_document(self, document_id: str) -> None:
        _wrap(self._container.document_service.delete, document_id)

    def citation(self, document_id: str, *, style: str) -> dict[str, Any]:
        document = _wrap(self._container.document_service.get, document_id)
        text = _wrap(format_citation, document, style=style)
        return CitationFormatResponse(
            document_id=document_id, style=style.lower(), citation=text
        ).model_dump()

    # -- chat -----------------------------------------------------------------
    def create_chat(self, *, document_ids: list[str]) -> dict[str, Any]:
        session = _wrap(
            self._container.chat_service.create_session,
            owner=_OWNER,
            document_ids=document_ids,
        )
        return ChatSessionResponse.from_domain(session).model_dump()

    def list_chats(self) -> list[dict[str, Any]]:
        sessions = _wrap(self._container.chat_service.list_sessions, owner=_OWNER)
        return [
            {
                "id": s.id,
                "title": s.title,
                "message_count": len(s.messages),
                "updated_at": s.updated_at,
            }
            for s in sessions
        ]

    def get_chat(self, session_id: str) -> dict[str, Any]:
        session = _wrap(self._container.chat_service.get_session, session_id)
        return ChatSessionResponse.from_domain(session).model_dump()

    def ask(
        self, session_id: str, *, question: str, document_ids: list[str] | None
    ) -> dict[str, Any]:
        session, answer = _wrap(
            self._container.chat_service.ask,
            session_id=session_id,
            question=question,
            document_ids=document_ids,
        )
        return {
            "session_id": session.id,
            "answer": AnswerResponse.from_domain(answer).model_dump(),
        }

    def delete_chat(self, session_id: str) -> None:
        _wrap(self._container.chat_service.delete_session, session_id)

    # -- search -----------------------------------------------------------------
    def search(
        self, *, query: str, document_ids: list[str] | None, top_k: int | None
    ) -> dict[str, Any]:
        results, elapsed_ms = _wrap(
            self._container.retrieval_service.retrieve,
            query,
            document_ids=document_ids,
            top_k=top_k,
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
        return SearchResponse(hits=hits, elapsed_ms=elapsed_ms).model_dump()

    # -- AI features --------------------------------------------------------
    def summarize(self, document_id: str) -> dict[str, Any]:
        answer = _wrap(self._container.ai_feature_service.summarize, document_id=document_id)
        return AnswerResponse.from_domain(answer).model_dump()

    def methodology(self, document_id: str) -> dict[str, Any]:
        answer = _wrap(
            self._container.ai_feature_service.extract_methodology, document_id=document_id
        )
        return AnswerResponse.from_domain(answer).model_dump()

    def limitations(self, document_id: str) -> dict[str, Any]:
        answer = _wrap(
            self._container.ai_feature_service.extract_limitations, document_id=document_id
        )
        return AnswerResponse.from_domain(answer).model_dump()

    def future_work(self, document_id: str) -> dict[str, Any]:
        answer = _wrap(
            self._container.ai_feature_service.extract_future_work, document_id=document_id
        )
        return AnswerResponse.from_domain(answer).model_dump()

    def explain(self, *, document_id: str, concept: str) -> dict[str, Any]:
        answer = _wrap(
            self._container.ai_feature_service.explain_concept,
            concept=concept,
            document_id=document_id,
        )
        return AnswerResponse.from_domain(answer).model_dump()

    def quiz(self, *, document_id: str, num_questions: int) -> dict[str, Any]:
        answer = _wrap(
            self._container.ai_feature_service.generate_quiz,
            document_id=document_id,
            num_questions=num_questions,
        )
        return AnswerResponse.from_domain(answer).model_dump()

    def flashcards(self, *, document_id: str, num_cards: int) -> dict[str, Any]:
        answer = _wrap(
            self._container.ai_feature_service.generate_flashcards,
            document_id=document_id,
            num_cards=num_cards,
        )
        return AnswerResponse.from_domain(answer).model_dump()

    def compare(self, *, document_ids: list[str], aspect: str | None) -> dict[str, Any]:
        answer = _wrap(
            self._container.ai_feature_service.compare_papers,
            document_ids=document_ids,
            aspect=aspect,
        )
        return AnswerResponse.from_domain(answer).model_dump()

    def literature_review(self, *, document_ids: list[str], topic: str | None) -> dict[str, Any]:
        answer = _wrap(
            self._container.ai_feature_service.literature_review,
            document_ids=document_ids,
            topic=topic,
        )
        return AnswerResponse.from_domain(answer).model_dump()

    # -- export ---------------------------------------------------------------
    def export_chat(self, session_id: str, *, fmt: str) -> bytes:
        session = _wrap(self._container.chat_service.get_session, session_id)
        markdown = conversation_to_markdown(session.messages, title=session.title)
        if fmt.lower() == "pdf":
            return _wrap(markdown_to_pdf_bytes, markdown, title=session.title)
        return markdown.encode("utf-8")

    # -- document file (used by the Viewer page) ------------------------------
    def document_file_bytes(self, document_id: str) -> bytes:
        """Return the raw stored PDF bytes for a document (embedded-mode only)."""
        path = _wrap(self._container.document_service.file_path, document_id)
        return path.read_bytes()
