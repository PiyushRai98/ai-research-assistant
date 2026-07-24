"""Ports: abstract interfaces the application depends on.

Following the Dependency Inversion Principle, use cases in the application
layer depend on these Protocols/ABCs rather than concrete adapters. This keeps
business logic decoupled from FAISS, PyMuPDF, sentence-transformers, SQLite,
and any particular LLM, and makes every collaborator trivially mockable.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol, runtime_checkable

from app.domain.models import (
    Answer,
    ChatSession,
    Chunk,
    Document,
    DocumentStatus,
    PageContent,
    ScoredChunk,
)


@runtime_checkable
class PDFParser(Protocol):
    """Extracts page-level text and metadata from raw PDF bytes."""

    def extract_pages(self, data: bytes) -> list[PageContent]:
        """Return ordered page contents. Raises ``PDFProcessingError`` on failure."""

    def extract_metadata(self, data: bytes) -> dict[str, str | int | None]:
        """Return best-effort bibliographic metadata."""


@runtime_checkable
class Chunker(Protocol):
    """Splits page content into overlapping, citation-aware chunks."""

    def split(
        self,
        pages: Sequence[PageContent],
        *,
        document_id: str,
        document_name: str,
    ) -> list[Chunk]:
        """Produce chunks that preserve page references and ordering."""


@runtime_checkable
class EmbeddingModel(Protocol):
    """Turns text into dense vectors for semantic search."""

    @property
    def dimension(self) -> int:
        """Dimensionality of the produced embedding vectors."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents (optimised path)."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""


class VectorStore(abc.ABC):
    """Persistent similarity index with metadata filtering and deletion."""

    @abc.abstractmethod
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Incrementally index chunks alongside their vectors."""

    @abc.abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        """Return the top-k most similar chunks, optionally filtered by document."""

    @abc.abstractmethod
    def search_mmr(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        fetch_k: int,
        lambda_mult: float,
        document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        """Maximal-marginal-relevance search balancing relevance and diversity."""

    @abc.abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Remove every chunk belonging to a document. Returns count removed."""

    @abc.abstractmethod
    def count(self) -> int:
        """Total number of indexed chunks."""

    @abc.abstractmethod
    def save(self) -> None:
        """Persist the index and metadata to disk."""

    @abc.abstractmethod
    def load(self) -> None:
        """Load a previously persisted index from disk (no-op if absent)."""


class DocumentRepository(abc.ABC):
    """Persistence port for document metadata (SQLite adapter in infra)."""

    @abc.abstractmethod
    def add(self, document: Document) -> Document:
        """Persist a new document record."""

    @abc.abstractmethod
    def update(self, document: Document) -> Document:
        """Persist changes to an existing document."""

    @abc.abstractmethod
    def get(self, document_id: str) -> Document | None:
        """Fetch a document by id, or None if absent."""

    @abc.abstractmethod
    def get_by_hash(self, content_hash: str, owner: str) -> Document | None:
        """Fetch a document by content hash for duplicate detection."""

    @abc.abstractmethod
    def list(self, *, owner: str | None = None) -> list[Document]:
        """List documents, optionally scoped to an owner."""

    @abc.abstractmethod
    def delete(self, document_id: str) -> bool:
        """Delete a document record. Returns True if a row was removed."""

    @abc.abstractmethod
    def set_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        error_message: str | None = None,
    ) -> None:
        """Update just the lifecycle status of a document."""


class ChatRepository(abc.ABC):
    """Persistence port for chat sessions and their messages."""

    @abc.abstractmethod
    def create(self, session: ChatSession) -> ChatSession:
        """Persist a new chat session."""

    @abc.abstractmethod
    def save(self, session: ChatSession) -> ChatSession:
        """Upsert a chat session (including its messages)."""

    @abc.abstractmethod
    def get(self, session_id: str) -> ChatSession | None:
        """Fetch a chat session by id, or None if absent."""

    @abc.abstractmethod
    def list(self, *, owner: str | None = None) -> list[ChatSession]:
        """List chat sessions, optionally scoped to an owner."""

    @abc.abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a chat session. Returns True if a row was removed."""

    @abc.abstractmethod
    def count(self, *, owner: str | None = None) -> int:
        """Count chat sessions, optionally scoped to an owner."""


@runtime_checkable
class LLMClient(Protocol):
    """A chat-completion language model with optional streaming."""

    @property
    def name(self) -> str:
        """Human-readable provider/model identifier for logging."""

    def complete(self, *, system: str, prompt: str) -> tuple[str, int | None, int | None]:
        """Return (text, prompt_tokens, completion_tokens) for a single call."""

    def stream(self, *, system: str, prompt: str) -> Iterator[str]:
        """Yield incremental text tokens as they are generated."""


class RAGEngine(Protocol):
    """Orchestrates retrieval + generation + citation extraction."""

    def answer(
        self,
        question: str,
        *,
        document_ids: Sequence[str] | None = None,
        history: Iterable[tuple[str, str]] | None = None,
    ) -> Answer:
        """Produce a grounded answer with citations for a user question."""
