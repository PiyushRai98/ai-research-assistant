"""Domain-agnostic exception hierarchy.

A single base class (:class:`ResearchAssistantError`) lets the API layer map
internal failures onto user-friendly HTTP responses without leaking stack
traces, while preserving a machine-readable ``code`` and an optional
``details`` payload for structured logging.
"""

from __future__ import annotations

from typing import Any


class ResearchAssistantError(Exception):
    """Base class for all application errors.

    Attributes:
        message: Human-readable, user-safe description.
        code: Stable machine-readable identifier (used by the API/frontend).
        status_code: Suggested HTTP status for API responses.
        details: Optional structured context for logging (never shown raw).
    """

    code: str = "internal_error"
    status_code: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}


class ValidationError(ResearchAssistantError):
    """Input failed validation (bad file, wrong MIME, oversized upload)."""

    code = "validation_error"
    status_code = 422


class NotFoundError(ResearchAssistantError):
    """A requested resource (document, chat, chunk) does not exist."""

    code = "not_found"
    status_code = 404


class DuplicateError(ResearchAssistantError):
    """The resource already exists (duplicate document upload)."""

    code = "duplicate"
    status_code = 409


class PDFProcessingError(ResearchAssistantError):
    """A PDF could not be parsed or was malformed/corrupt."""

    code = "pdf_processing_error"
    status_code = 422


class EmbeddingError(ResearchAssistantError):
    """The embedding backend failed to produce vectors."""

    code = "embedding_error"
    status_code = 502


class VectorStoreError(ResearchAssistantError):
    """The vector store could not be read, written, or queried."""

    code = "vector_store_error"
    status_code = 500


class RetrievalError(ResearchAssistantError):
    """Retrieval failed to return usable context."""

    code = "retrieval_error"
    status_code = 500


class LLMError(ResearchAssistantError):
    """The language model backend failed or timed out."""

    code = "llm_error"
    status_code = 502


class SecurityError(ResearchAssistantError):
    """A security constraint was violated (e.g. path traversal, injection)."""

    code = "security_error"
    status_code = 400


class AuthenticationError(ResearchAssistantError):
    """Authentication is required or failed."""

    code = "authentication_error"
    status_code = 401
