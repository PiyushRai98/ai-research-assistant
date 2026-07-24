"""End-to-end API tests via FastAPI TestClient (offline container)."""

from __future__ import annotations


def test_health(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "llm" in body


def test_upload_and_list_documents(client, sample_pdf_bytes: bytes) -> None:
    upload = client.post(
        "/api/documents",
        files={"file": ("paper.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert upload.status_code == 201, upload.text
    document = upload.json()
    assert document["status"] == "indexed"

    listing = client.get("/api/documents")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1


def test_reject_non_pdf_upload(client) -> None:
    response = client.post(
        "/api/documents",
        files={"file": ("evil.pdf", b"not a pdf at all", "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_chat_flow_returns_citations(client, sample_pdf_bytes: bytes) -> None:
    document = client.post(
        "/api/documents",
        files={"file": ("paper.pdf", sample_pdf_bytes, "application/pdf")},
    ).json()

    created = client.post("/api/chats", json={"document_ids": [document["id"]]})
    assert created.status_code == 201
    session_id = created.json()["id"]

    answer = client.post(
        f"/api/chats/{session_id}/ask",
        json={"question": "What do transformers use?", "document_ids": [document["id"]]},
    )
    assert answer.status_code == 200
    payload = answer.json()["answer"]
    assert payload["context_found"] is True
    assert payload["citations"]


def test_search_endpoint(client, sample_pdf_bytes: bytes) -> None:
    client.post(
        "/api/documents",
        files={"file": ("paper.pdf", sample_pdf_bytes, "application/pdf")},
    )
    response = client.post("/api/search", json={"query": "attention", "top_k": 3})
    assert response.status_code == 200
    assert "hits" in response.json()


def test_dashboard_counts(client, sample_pdf_bytes: bytes) -> None:
    client.post(
        "/api/documents",
        files={"file": ("paper.pdf", sample_pdf_bytes, "application/pdf")},
    )
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    body = response.json()
    assert body["document_count"] == 1
    assert body["indexed_count"] == 1


def test_citation_formatting_endpoint(client, sample_pdf_bytes: bytes) -> None:
    document = client.post(
        "/api/documents",
        files={"file": ("paper.pdf", sample_pdf_bytes, "application/pdf")},
    ).json()
    response = client.get(f"/api/ai/{document['id']}/citation", params={"style": "apa"})
    assert response.status_code == 200
    assert response.json()["style"] == "apa"


def test_summary_endpoint(client, sample_pdf_bytes: bytes) -> None:
    document = client.post(
        "/api/documents",
        files={"file": ("paper.pdf", sample_pdf_bytes, "application/pdf")},
    ).json()
    response = client.post(f"/api/ai/{document['id']}/summary")
    assert response.status_code == 200
    assert "text" in response.json()


def test_export_markdown(client, sample_pdf_bytes: bytes) -> None:
    document = client.post(
        "/api/documents",
        files={"file": ("paper.pdf", sample_pdf_bytes, "application/pdf")},
    ).json()
    session_id = client.post("/api/chats", json={"document_ids": [document["id"]]}).json()["id"]
    client.post(
        f"/api/chats/{session_id}/ask",
        json={"question": "What is attention?", "document_ids": [document["id"]]},
    )
    response = client.get(f"/api/export/chats/{session_id}", params={"fmt": "markdown"})
    assert response.status_code == 200
    assert b"#" in response.content


def test_not_found_document(client) -> None:
    response = client.get("/api/documents/does-not-exist")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
