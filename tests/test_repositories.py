"""Unit tests for the SQLite document and chat repositories."""

from __future__ import annotations

from pathlib import Path

from app.domain.models import (
    ChatMessage,
    ChatSession,
    Document,
    DocumentStatus,
    MessageRole,
)
from app.infrastructure.chat_store import SQLiteChatRepository
from app.infrastructure.database import SQLiteDocumentRepository


def _doc(**kwargs) -> Document:
    defaults = {
        "filename": "a.pdf",
        "content_hash": "h1",
        "size_bytes": 10,
        "owner": "guest",
    }
    defaults.update(kwargs)
    return Document(**defaults)


def test_document_crud(tmp_path: Path) -> None:
    repo = SQLiteDocumentRepository(tmp_path / "meta.db")
    doc = _doc()
    repo.add(doc)

    fetched = repo.get(doc.id)
    assert fetched is not None and fetched.filename == "a.pdf"

    repo.set_status(doc.id, DocumentStatus.INDEXED)
    assert repo.get(doc.id).status is DocumentStatus.INDEXED

    assert len(repo.list(owner="guest")) == 1
    assert repo.delete(doc.id) is True
    assert repo.get(doc.id) is None


def test_document_dedup_by_hash(tmp_path: Path) -> None:
    repo = SQLiteDocumentRepository(tmp_path / "meta.db")
    repo.add(_doc(content_hash="dup"))
    assert repo.get_by_hash("dup", "guest") is not None
    assert repo.get_by_hash("missing", "guest") is None


def test_chat_persistence(tmp_path: Path) -> None:
    repo = SQLiteChatRepository(tmp_path / "meta.db")
    session = ChatSession(owner="guest", document_ids=["d1"])
    repo.create(session)

    session = session.with_message(ChatMessage(role=MessageRole.USER, content="Hello?"))
    repo.save(session)

    fetched = repo.get(session.id)
    assert fetched is not None
    assert fetched.messages[0].content == "Hello?"
    assert fetched.title == "Hello?"  # title derived from first user message
    assert repo.count(owner="guest") == 1

    assert repo.delete(session.id) is True
    assert repo.count(owner="guest") == 0
