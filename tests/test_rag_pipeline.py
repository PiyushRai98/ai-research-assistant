"""Integration tests for the RAG pipeline using the offline container."""

from __future__ import annotations

from app.backend.container import Container


def _ingest(container: Container, sample_pdf_bytes: bytes):  # noqa: ANN202
    return container.document_service.upload(
        data=sample_pdf_bytes, filename="paper.pdf", content_type="application/pdf"
    )


def test_upload_indexes_document(container: Container, sample_pdf_bytes: bytes) -> None:
    document = _ingest(container, sample_pdf_bytes)
    assert document.status.value == "indexed"
    assert document.chunk_count > 0
    assert container.vector_store.count() == document.chunk_count


def test_duplicate_upload_detected(container: Container, sample_pdf_bytes: bytes) -> None:
    from app.shared.exceptions import DuplicateError

    _ingest(container, sample_pdf_bytes)
    try:
        _ingest(container, sample_pdf_bytes)
    except DuplicateError as exc:
        assert "already" in exc.message.lower()
    else:  # pragma: no cover
        raise AssertionError("expected DuplicateError")


def test_rag_answer_has_citations(container: Container, sample_pdf_bytes: bytes) -> None:
    document = _ingest(container, sample_pdf_bytes)
    answer = container.rag_service.answer(
        "What do transformers use for long-range dependencies?",
        document_ids=[document.id],
    )
    assert answer.context_found
    assert answer.citations
    # Every citation must resolve to the real ingested document (no fabrication).
    assert all(c.document_id == document.id for c in answer.citations)
    assert all(c.page_number >= 1 for c in answer.citations)


def test_rag_reports_not_found_when_no_documents(container: Container) -> None:
    answer = container.rag_service.answer("Anything at all?")
    assert answer.context_found is False
    assert answer.citations == []


def test_chat_persists_turns(container: Container, sample_pdf_bytes: bytes) -> None:
    document = _ingest(container, sample_pdf_bytes)
    session = container.chat_service.create_session(document_ids=[document.id])
    updated, answer = container.chat_service.ask(
        session_id=session.id, question="What is attention?"
    )
    assert len(updated.messages) == 2
    assert answer.text
    assert container.chat_service.count() == 1


def test_delete_removes_vectors(container: Container, sample_pdf_bytes: bytes) -> None:
    document = _ingest(container, sample_pdf_bytes)
    assert container.vector_store.count() > 0
    container.document_service.delete(document.id)
    assert container.vector_store.count() == 0
