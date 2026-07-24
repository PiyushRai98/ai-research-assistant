"""Document ingestion use case.

Coordinates the full upload-to-indexed lifecycle:

    validate -> dedup -> persist file -> parse -> chunk -> embed -> index ->
    update metadata

Each collaborator is injected as a port, so this service is unit-testable with
fakes and free of any specific PDF/embedding/vector library. Failures mark the
document ``FAILED`` (with a message) rather than raising past the caller,
supporting the "never crash, degrade gracefully" requirement.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.domain.models import Document, DocumentMetadata, DocumentStatus
from app.domain.ports import (
    Chunker,
    DocumentRepository,
    EmbeddingModel,
    PDFParser,
    VectorStore,
)
from app.shared.config import Settings
from app.shared.exceptions import DuplicateError, NotFoundError, ResearchAssistantError
from app.shared.logging import get_logger, log_embedding, log_error, log_upload
from app.shared.security import sanitize_filename, sha256_hex, validate_pdf_upload

logger = get_logger("documents")


class DocumentService:
    """Ingests, processes, lists, and deletes research documents."""

    def __init__(
        self,
        *,
        repository: DocumentRepository,
        parser: PDFParser,
        chunker: Chunker,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        settings: Settings,
    ) -> None:
        self._repo = repository
        self._parser = parser
        self._chunker = chunker
        self._embeddings = embedding_model
        self._store = vector_store
        self._settings = settings

    # -- ingestion ----------------------------------------------------------
    def upload(
        self,
        *,
        data: bytes,
        filename: str,
        content_type: str | None = None,
        owner: str = "guest",
    ) -> Document:
        """Validate, deduplicate, persist, and process an uploaded PDF."""
        validate_pdf_upload(
            data,
            max_bytes=self._settings.upload_max_bytes,
            declared_content_type=content_type,
        )
        safe_name = sanitize_filename(filename)
        content_hash = sha256_hex(data)

        existing = self._repo.get_by_hash(content_hash, owner)
        if existing is not None:
            raise DuplicateError(
                f"'{safe_name}' has already been uploaded.",
                details={"document_id": existing.id},
            )

        document = Document(
            filename=safe_name,
            content_hash=content_hash,
            size_bytes=len(data),
            status=DocumentStatus.PENDING,
            owner=owner,
        )
        self._persist_file(document.id, data)
        self._repo.add(document)
        log_upload(document_id=document.id, filename=safe_name, size_bytes=len(data))

        return self._process(document, data)

    def _persist_file(self, document_id: str, data: bytes) -> None:
        """Store the raw PDF on disk under a hash-safe filename."""
        upload_dir = self._settings.storage_upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / f"{document_id}.pdf").write_bytes(data)

    def _process(self, document: Document, data: bytes) -> Document:
        """Parse, chunk, embed, and index a stored document."""
        self._repo.set_status(document.id, DocumentStatus.PROCESSING)
        started = time.perf_counter()
        try:
            raw_meta = self._parser.extract_metadata(data)
            pages = self._parser.extract_pages(data)
            chunks = self._chunker.split(
                pages, document_id=document.id, document_name=document.filename
            )
            if not chunks:
                raise ResearchAssistantError("No text chunks produced from document.")

            embed_started = time.perf_counter()
            vectors = self._embeddings.embed_documents([c.text for c in chunks])
            embed_ms = (time.perf_counter() - embed_started) * 1000.0
            log_embedding(
                document_id=document.id,
                chunk_count=len(chunks),
                duration_ms=embed_ms,
            )

            self._store.add(chunks, vectors)
            self._store.save()

            processing_ms = (time.perf_counter() - started) * 1000.0
            updated = document.model_copy(
                update={
                    "status": DocumentStatus.INDEXED,
                    "metadata": DocumentMetadata.model_validate(raw_meta),
                    "chunk_count": len(chunks),
                    "processing_ms": processing_ms,
                    "error_message": None,
                }
            )
            self._repo.update(updated)
            return updated
        except Exception as exc:  # degrade gracefully, record the failure
            log_error(exc, document_id=document.id, stage="processing")
            self._repo.set_status(document.id, DocumentStatus.FAILED, error_message=str(exc))
            failed = self._repo.get(document.id)
            if failed is None:  # pragma: no cover - defensive
                raise
            return failed

    # -- queries ------------------------------------------------------------
    def get(self, document_id: str) -> Document:
        document = self._repo.get(document_id)
        if document is None:
            raise NotFoundError(f"Document '{document_id}' was not found.")
        return document

    def list(self, *, owner: str | None = None) -> list[Document]:
        return self._repo.list(owner=owner)

    def delete(self, document_id: str) -> None:
        """Remove a document's record, vectors, and stored file."""
        self.get(document_id)  # raises NotFoundError if absent
        removed = self._store.delete_document(document_id)
        if removed:
            self._store.save()
        self._repo.delete(document_id)
        stored = self._settings.storage_upload_dir / f"{document_id}.pdf"
        stored.unlink(missing_ok=True)
        logger.info(
            "Deleted document {id} ({chunks} chunks)",
            id=document_id,
            chunks=removed,
        )

    def file_path(self, document_id: str) -> Path:
        """Return the on-disk path of a stored PDF, ensuring it exists."""
        self.get(document_id)
        path = self._settings.storage_upload_dir / f"{document_id}.pdf"
        if not path.exists():
            raise NotFoundError("The stored PDF file is missing.")
        return path

    # -- statistics (dashboard) --------------------------------------------
    def statistics(self, *, owner: str | None = None) -> dict[str, object]:
        """Aggregate document stats for the dashboard."""
        documents = self.list(owner=owner)
        total_bytes = sum(d.size_bytes for d in documents)
        indexed = [d for d in documents if d.status is DocumentStatus.INDEXED]
        avg_ms = sum(d.processing_ms for d in indexed) / len(indexed) if indexed else 0.0
        return {
            "document_count": len(documents),
            "indexed_count": len(indexed),
            "failed_count": sum(1 for d in documents if d.status is DocumentStatus.FAILED),
            "total_chunks": self._store.count(),
            "storage_bytes": total_bytes,
            "avg_processing_ms": round(avg_ms, 2),
        }
