"""Tests for the in-process EmbeddedClient used when no remote API is reachable.

Verifies the embedded client satisfies the same :class:`ResearchClient`
protocol as :class:`APIClient` and produces correct, grounded results end to
end — upload, dashboard, chat, search, citation formatting, export, and the
PDF-viewer byte fetch — entirely in-process (no HTTP, no separate backend).
"""

from __future__ import annotations

import pytest
from app.backend.container import Container
from app.frontend.api_client import ResearchClient
from app.frontend.embedded_client import EmbeddedClient


@pytest.fixture
def embedded(container: Container, monkeypatch: pytest.MonkeyPatch) -> EmbeddedClient:
    """An EmbeddedClient wired to the fast, offline test container."""
    client = EmbeddedClient.__new__(EmbeddedClient)  # bypass build_container()
    client._container = container  # noqa: SLF001
    return client


def test_embedded_client_satisfies_protocol(embedded: EmbeddedClient) -> None:
    assert isinstance(embedded, ResearchClient)


def test_health_reports_configured_models(embedded: EmbeddedClient) -> None:
    health = embedded.health()
    assert health["status"] == "ok"
    assert "embedding_model" in health
    assert "llm" in health


def test_full_workflow_end_to_end(embedded: EmbeddedClient, sample_pdf_bytes: bytes) -> None:
    upload = embedded.upload_document(filename="paper.pdf", data=sample_pdf_bytes)
    assert upload["status"] == "indexed"
    document_id = upload["id"]

    listing = embedded.list_documents()
    assert listing["total"] == 1

    dashboard = embedded.dashboard()
    assert dashboard["document_count"] == 1
    assert dashboard["indexed_count"] == 1

    session = embedded.create_chat(document_ids=[document_id])
    result = embedded.ask(
        session["id"],
        question="What do transformers use for long-range dependencies?",
        document_ids=[document_id],
    )
    answer = result["answer"]
    assert answer["context_found"] is True
    assert answer["citations"]

    search_result = embedded.search(query="attention", document_ids=None, top_k=3)
    assert search_result["hits"]

    citation = embedded.citation(document_id, style="apa")
    assert citation["style"] == "apa"

    markdown = embedded.export_chat(session["id"], fmt="markdown")
    assert markdown.startswith(b"#")

    pdf_bytes = embedded.document_file_bytes(document_id)
    assert pdf_bytes.startswith(b"%PDF-")

    embedded.delete_document(document_id)
    assert embedded.list_documents()["total"] == 0


def test_not_found_raises_api_error(embedded: EmbeddedClient) -> None:
    from app.frontend.api_client import APIError

    with pytest.raises(APIError):
        embedded.citation("does-not-exist", style="apa")
