"""SQLite-backed implementation of :class:`ChatRepository`.

Chat sessions and their messages persist in the same SQLite database file used
for document metadata. Messages (including citations) are stored as a JSON blob
on the session row — conversations are always read and written whole, so a
denormalised design is simpler and faster than a joined message table for this
access pattern.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from app.domain.models import ChatMessage, ChatSession
from app.domain.ports import ChatRepository
from app.shared.exceptions import VectorStoreError
from app.shared.logging import get_logger

logger = get_logger("chat_store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    owner         TEXT NOT NULL DEFAULT 'guest',
    document_ids  TEXT NOT NULL DEFAULT '[]',
    messages_json TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_owner ON chat_sessions(owner);
"""


class SQLiteChatRepository(ChatRepository):
    """Persist chat sessions in a local SQLite database file."""

    def __init__(self, database_path: Path) -> None:
        self._path = database_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _initialise(self) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.executescript(_SCHEMA)
        except sqlite3.Error as exc:  # pragma: no cover - startup failure
            raise VectorStoreError(
                "Failed to initialise chat database", details={"error": str(exc)}
            ) from exc

    @staticmethod
    def _to_row(session: ChatSession) -> dict[str, object]:
        return {
            "id": session.id,
            "title": session.title,
            "owner": session.owner,
            "document_ids": json.dumps(session.document_ids),
            "messages_json": json.dumps([m.model_dump(mode="json") for m in session.messages]),
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ChatSession:
        return ChatSession(
            id=row["id"],
            title=row["title"],
            owner=row["owner"],
            document_ids=json.loads(row["document_ids"]),
            messages=[ChatMessage.model_validate(m) for m in json.loads(row["messages_json"])],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def create(self, session: ChatSession) -> ChatSession:
        row = self._to_row(session)
        columns = ", ".join(row.keys())
        placeholders = ", ".join(f":{key}" for key in row)
        with self._lock, self._connect() as conn:
            conn.execute(f"INSERT INTO chat_sessions ({columns}) VALUES ({placeholders})", row)
        return session

    def save(self, session: ChatSession) -> ChatSession:
        row = self._to_row(session)
        columns = ", ".join(row.keys())
        placeholders = ", ".join(f":{key}" for key in row)
        # UPSERT keeps create/save idempotent for the API layer.
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT INTO chat_sessions ({columns}) VALUES ({placeholders}) "
                "ON CONFLICT(id) DO UPDATE SET "
                "title=excluded.title, document_ids=excluded.document_ids, "
                "messages_json=excluded.messages_json, updated_at=excluded.updated_at",
                row,
            )
        return session

    def get(self, session_id: str) -> ChatSession | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        return self._from_row(row) if row else None

    def list(self, *, owner: str | None = None) -> list[ChatSession]:
        query = "SELECT * FROM chat_sessions"
        params: tuple[str, ...] = ()
        if owner is not None:
            query += " WHERE owner = ?"
            params = (owner,)
        query += " ORDER BY updated_at DESC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._from_row(row) for row in rows]

    def delete(self, session_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0

    def count(self, *, owner: str | None = None) -> int:
        query = "SELECT COUNT(*) AS n FROM chat_sessions"
        params: tuple[str, ...] = ()
        if owner is not None:
            query += " WHERE owner = ?"
            params = (owner,)
        with self._lock, self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return int(row["n"])
