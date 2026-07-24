"""Core domain entities and value objects.

These immutable, validated models are the shared vocabulary of the entire
application. They carry no persistence or framework concerns; adapters in the
infrastructure layer translate to/from these types.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp (avoids naive-datetime bugs)."""
    return datetime.now(UTC)


def new_id() -> str:
    """Generate a short, URL-safe unique identifier."""
    return uuid.uuid4().hex


class DocumentStatus(StrEnum):
    """Lifecycle of an uploaded document as it is processed for retrieval."""

    PENDING = "pending"  # stored, not yet processed
    PROCESSING = "processing"  # extraction/chunking/embedding in progress
    INDEXED = "indexed"  # searchable in the vector store
    FAILED = "failed"  # processing failed (see error_message)


class PageContent(BaseModel):
    """Text extracted from a single PDF page, retaining its 1-based number."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1, description="1-based page index within the document.")
    text: str = Field(description="Raw extracted text for the page.")


class DocumentMetadata(BaseModel):
    """Bibliographic metadata extracted from a PDF (best-effort)."""

    model_config = ConfigDict(frozen=True)

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    page_count: int = Field(default=0, ge=0)


class Document(BaseModel):
    """A user-uploaded research paper tracked through its processing lifecycle."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_id)
    filename: str
    content_hash: str = Field(description="SHA-256 of raw bytes; enables dedup.")
    size_bytes: int = Field(ge=0)
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    chunk_count: int = Field(default=0, ge=0)
    processing_ms: float = Field(default=0.0, ge=0.0)
    error_message: str | None = None
    owner: str = Field(default="guest", description="Username or 'guest'.")
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def with_status(
        self,
        status: DocumentStatus,
        *,
        error_message: str | None = None,
    ) -> Document:
        """Return a copy transitioned to a new status (entities are immutable)."""
        return self.model_copy(
            update={
                "status": status,
                "error_message": error_message,
                "updated_at": _utcnow(),
            }
        )


class ChunkMetadata(BaseModel):
    """Provenance attached to every chunk, enabling precise citations."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    document_name: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0, description="Ordinal position within the document.")


class Chunk(BaseModel):
    """A retrievable unit of text plus the metadata needed to cite it."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_id)
    text: str
    metadata: ChunkMetadata


class ScoredChunk(BaseModel):
    """A chunk returned by retrieval together with its relevance score."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(description="Higher is more relevant (0..1 normalised).")


class Citation(BaseModel):
    """A verifiable reference backing part of an answer.

    Every field required by the brief is present: document, page, chunk, and a
    verbatim quote drawn from the retrieved context (never fabricated).
    """

    model_config = ConfigDict(frozen=True)

    marker: int = Field(ge=1, description="1-based citation marker, e.g. [1].")
    document_id: str
    document_name: str
    page_number: int = Field(ge=1)
    chunk_id: str
    quote: str = Field(description="Verbatim excerpt from the cited chunk.")
    score: float = Field(default=0.0)


class MessageRole(StrEnum):
    """Author of a chat message."""

    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single turn in a conversation."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_id)
    role: MessageRole
    content: str
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class ChatSession(BaseModel):
    """A persisted conversation, scoped to an owner and optional documents."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_id)
    title: str = "New conversation"
    owner: str = "guest"
    document_ids: list[str] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def with_message(self, message: ChatMessage) -> ChatSession:
        """Return a copy with an appended message and bumped timestamp."""
        title = self.title
        if title == "New conversation" and message.role is MessageRole.USER:
            title = message.content[:60]
        return self.model_copy(
            update={
                "messages": [*self.messages, message],
                "title": title,
                "updated_at": _utcnow(),
            }
        )


class Answer(BaseModel):
    """The RAG pipeline's response: generated text plus grounding citations."""

    model_config = ConfigDict(frozen=True)

    text: str
    citations: list[Citation] = Field(default_factory=list)
    context_found: bool = Field(
        default=True,
        description="False when retrieval produced insufficient context.",
    )
    llm_ms: float = Field(default=0.0, ge=0.0)
    retrieval_ms: float = Field(default=0.0, ge=0.0)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
