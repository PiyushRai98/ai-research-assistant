"""Pydantic request/response models for the HTTP API.

These DTOs form the API contract and are intentionally separate from domain
entities: the API can evolve independently, and we control precisely which
fields are exposed. Mapper helpers convert domain objects to responses.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.models import (
    Answer,
    ChatSession,
    Citation,
    Document,
)


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
class DocumentMetadataResponse(BaseModel):
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    page_count: int = 0


class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    size_bytes: int
    chunk_count: int
    processing_ms: float
    error_message: str | None = None
    metadata: DocumentMetadataResponse
    created_at: datetime

    @classmethod
    def from_domain(cls, document: Document) -> DocumentResponse:
        return cls(
            id=document.id,
            filename=document.filename,
            status=document.status.value,
            size_bytes=document.size_bytes,
            chunk_count=document.chunk_count,
            processing_ms=document.processing_ms,
            error_message=document.error_message,
            metadata=DocumentMetadataResponse(**document.metadata.model_dump()),
            created_at=document.created_at,
        )


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


# --------------------------------------------------------------------------
# Citations & answers
# --------------------------------------------------------------------------
class CitationResponse(BaseModel):
    marker: int
    document_id: str
    document_name: str
    page_number: int
    chunk_id: str
    quote: str
    score: float

    @classmethod
    def from_domain(cls, citation: Citation) -> CitationResponse:
        return cls(**citation.model_dump())


class AnswerResponse(BaseModel):
    text: str
    citations: list[CitationResponse]
    context_found: bool
    llm_ms: float
    retrieval_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @classmethod
    def from_domain(cls, answer: Answer) -> AnswerResponse:
        return cls(
            text=answer.text,
            citations=[CitationResponse.from_domain(c) for c in answer.citations],
            context_found=answer.context_found,
            llm_ms=answer.llm_ms,
            retrieval_ms=answer.retrieval_ms,
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
        )


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
class CreateChatRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] | None = None


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    citations: list[CitationResponse]
    created_at: datetime


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    document_ids: list[str]
    messages: list[ChatMessageResponse]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, session: ChatSession) -> ChatSessionResponse:
        return cls(
            id=session.id,
            title=session.title,
            document_ids=session.document_ids,
            messages=[
                ChatMessageResponse(
                    id=m.id,
                    role=m.role.value,
                    content=m.content,
                    citations=[CitationResponse.from_domain(c) for c in m.citations],
                    created_at=m.created_at,
                )
                for m in session.messages
            ],
            created_at=session.created_at,
            updated_at=session.updated_at,
        )


class ChatSessionSummary(BaseModel):
    id: str
    title: str
    message_count: int
    updated_at: datetime


class AskResponse(BaseModel):
    session_id: str
    answer: AnswerResponse


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class SearchHit(BaseModel):
    text: str
    score: float
    document_id: str
    document_name: str
    page_number: int


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    elapsed_ms: float


# --------------------------------------------------------------------------
# AI features
# --------------------------------------------------------------------------
class ExplainRequest(BaseModel):
    document_id: str
    concept: str = Field(min_length=1, max_length=500)


class QuizRequest(BaseModel):
    document_id: str
    num_questions: int = Field(default=5, ge=1, le=20)


class FlashcardRequest(BaseModel):
    document_id: str
    num_cards: int = Field(default=8, ge=1, le=30)


class CompareRequest(BaseModel):
    document_ids: list[str] = Field(min_length=2)
    aspect: str | None = None


class LiteratureReviewRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1)
    topic: str | None = None


class CitationFormatResponse(BaseModel):
    document_id: str
    style: str
    citation: str


# --------------------------------------------------------------------------
# Dashboard & health
# --------------------------------------------------------------------------
class DashboardResponse(BaseModel):
    document_count: int
    indexed_count: int
    failed_count: int
    total_chunks: int
    storage_bytes: int
    avg_processing_ms: float
    chat_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    embedding_model: str
    llm: str


class ErrorResponse(BaseModel):
    code: str
    message: str
