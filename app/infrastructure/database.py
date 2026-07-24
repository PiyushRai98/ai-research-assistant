"""SQLite-backed implementation of :class:`DocumentRepository`.

SQLite is chosen per the brief for lightweight, zero-config metadata storage.
The adapter is thread-safe for the single-writer FastAPI use case: each call
opens a short-lived connection (``check_same_thread=False`` is unnecessary),
and writes are serialised by SQLite's file lock. JSON is used for the nested
metadata value object so the schema stays flat and migration-friendly.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from app.domain.models import Document, DocumentMetadata, DocumentStatus
from app.domain.ports import DocumentRepository
from app.shared.exceptions import VectorStoreError
from app.shared.logging import get_logger

logger = get_logger("database")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id             TEXT PRIMARY KEY,
    filename       TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL,
    status         TEXT NOT NULL,
    metadata_json  TEXT NOT NULL,
    chunk_count    INTEGER NOT NULL DEFAULT 0,
    processing_ms  REAL NOT NULL DEFAULT 0,
    error_message  TEXT,
    owner          TEXT NOT NULL DEFAULT 'guest',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash_owner
    ON documents(content_hash, owner);
"""


class SQLiteDocumentRepository(DocumentRepository):
    """Persist document metadata in a local SQLite database file."""

    def __init__(self, database_path: Path) -> None:
        self._path = database_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # A re-entrant lock guards multi-statement operations across threads.
        self._lock = threading.RLock()
        self._initialise()

    # -- connection helpers -------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _initialise(self) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.executescript(_SCHEMA)
        except sqlite3.Error as exc:  # pragma: no cover - startup failure
            raise VectorStoreError(
                "Failed to initialise metadata database",
                details={"error": str(exc)},
            ) from exc

    # -- (de)serialisation --------------------------------------------------
    @staticmethod
    def _to_row(document: Document) -> dict[str, object]:
        return {
            "id": document.id,
            "filename": document.filename,
            "content_hash": document.content_hash,
            "size_bytes": document.size_bytes,
            "status": document.status.value,
            "metadata_json": document.metadata.model_dump_json(),
            "chunk_count": document.chunk_count,
            "processing_ms": document.processing_ms,
            "error_message": document.error_message,
            "owner": document.owner,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
        }

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            filename=row["filename"],
            content_hash=row["content_hash"],
            size_bytes=row["size_bytes"],
            status=DocumentStatus(row["status"]),
            metadata=DocumentMetadata.model_validate_json(row["metadata_json"]),
            chunk_count=row["chunk_count"],
            processing_ms=row["processing_ms"],
            error_message=row["error_message"],
            owner=row["owner"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # -- DocumentRepository -------------------------------------------------
    def add(self, document: Document) -> Document:
        row = self._to_row(document)
        columns = ", ".join(row.keys())
        placeholders = ", ".join(f":{key}" for key in row)
        with self._lock, self._connect() as conn:
            conn.execute(f"INSERT INTO documents ({columns}) VALUES ({placeholders})", row)
        logger.debug("Inserted document {id}", id=document.id)
        return document

    def update(self, document: Document) -> Document:
        row = self._to_row(document)
        assignments = ", ".join(f"{key} = :{key}" for key in row if key != "id")
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE documents SET {assignments} WHERE id = :id", row)
        return document

    def get(self, document_id: str) -> Document | None:
        with self._lock, self._connect() as conn:
            cur = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
            row = cur.fetchone()
        return self._from_row(row) if row else None

    def get_by_hash(self, content_hash: str, owner: str) -> Document | None:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ? AND owner = ?",
                (content_hash, owner),
            )
            row = cur.fetchone()
        return self._from_row(row) if row else None

    def list(self, *, owner: str | None = None) -> list[Document]:
        query = "SELECT * FROM documents"
        params: tuple[str, ...] = ()
        if owner is not None:
            query += " WHERE owner = ?"
            params = (owner,)
        query += " ORDER BY created_at DESC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._from_row(row) for row in rows]

    def delete(self, document_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            deleted = cur.rowcount > 0
        if deleted:
            logger.debug("Deleted document {id}", id=document_id)
        return deleted

    def set_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        error_message: str | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, error_message = ?, updated_at = ? "
                "WHERE id = ?",
                (status.value, error_message, datetime.now().isoformat(), document_id),
            )
