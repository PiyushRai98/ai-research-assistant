"""Thin HTTP client wrapping the FastAPI backend.

Keeps all network concerns out of the Streamlit views (Clean Architecture:
the presentation layer talks to the API, never to the domain/services
directly). Errors are normalised into :class:`APIError` so the UI can show
friendly messages consistently.
"""

from __future__ import annotations

from typing import Any

import httpx


class APIError(Exception):
    """Raised when the backend returns an error or is unreachable."""

    def __init__(self, message: str, *, code: str = "error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class APIClient:
    """Synchronous client for the research assistant API."""

    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self._base = base_url.rstrip("/") + "/api"
        self._timeout = timeout
        self._token: str | None = None

    def set_token(self, token: str | None) -> None:
        self._token = token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self._base}{path}"
        try:
            response = httpx.request(
                method, url, timeout=self._timeout, headers=self._headers(), **kwargs
            )
        except httpx.HTTPError as exc:
            raise APIError(
                "Could not reach the backend API. Is the server running?",
                code="connection_error",
            ) from exc
        if response.status_code >= 400:
            try:
                payload = response.json()
                message = payload.get("message", response.text)
                code = payload.get("code", "error")
            except (ValueError, AttributeError):
                message, code = response.text, "error"
            raise APIError(message, code=code)
        return response

    # -- health & dashboard -------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health").json()

    def dashboard(self) -> dict[str, Any]:
        return self._request("GET", "/dashboard").json()

    # -- documents ----------------------------------------------------------
    def upload_document(self, *, filename: str, data: bytes) -> dict[str, Any]:
        files = {"file": (filename, data, "application/pdf")}
        return self._request("POST", "/documents", files=files).json()

    def list_documents(self) -> dict[str, Any]:
        return self._request("GET", "/documents").json()

    def delete_document(self, document_id: str) -> None:
        self._request("DELETE", f"/documents/{document_id}")

    def citation(self, document_id: str, *, style: str) -> dict[str, Any]:
        return self._request("GET", f"/ai/{document_id}/citation", params={"style": style}).json()

    # -- chat ---------------------------------------------------------------
    def create_chat(self, *, document_ids: list[str]) -> dict[str, Any]:
        return self._request("POST", "/chats", json={"document_ids": document_ids}).json()

    def list_chats(self) -> list[dict[str, Any]]:
        return self._request("GET", "/chats").json()

    def get_chat(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/chats/{session_id}").json()

    def ask(
        self, session_id: str, *, question: str, document_ids: list[str] | None
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/chats/{session_id}/ask",
            json={"question": question, "document_ids": document_ids},
        ).json()

    def delete_chat(self, session_id: str) -> None:
        self._request("DELETE", f"/chats/{session_id}")

    # -- search -------------------------------------------------------------
    def search(
        self, *, query: str, document_ids: list[str] | None, top_k: int | None
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/search",
            json={"query": query, "document_ids": document_ids, "top_k": top_k},
        ).json()

    # -- AI features --------------------------------------------------------
    def summarize(self, document_id: str) -> dict[str, Any]:
        return self._request("POST", f"/ai/{document_id}/summary").json()

    def methodology(self, document_id: str) -> dict[str, Any]:
        return self._request("POST", f"/ai/{document_id}/methodology").json()

    def limitations(self, document_id: str) -> dict[str, Any]:
        return self._request("POST", f"/ai/{document_id}/limitations").json()

    def future_work(self, document_id: str) -> dict[str, Any]:
        return self._request("POST", f"/ai/{document_id}/future-work").json()

    def explain(self, *, document_id: str, concept: str) -> dict[str, Any]:
        return self._request(
            "POST", "/ai/explain", json={"document_id": document_id, "concept": concept}
        ).json()

    def quiz(self, *, document_id: str, num_questions: int) -> dict[str, Any]:
        return self._request(
            "POST",
            "/ai/quiz",
            json={"document_id": document_id, "num_questions": num_questions},
        ).json()

    def flashcards(self, *, document_id: str, num_cards: int) -> dict[str, Any]:
        return self._request(
            "POST",
            "/ai/flashcards",
            json={"document_id": document_id, "num_cards": num_cards},
        ).json()

    def compare(self, *, document_ids: list[str], aspect: str | None) -> dict[str, Any]:
        return self._request(
            "POST",
            "/ai/compare",
            json={"document_ids": document_ids, "aspect": aspect},
        ).json()

    def literature_review(self, *, document_ids: list[str], topic: str | None) -> dict[str, Any]:
        return self._request(
            "POST",
            "/ai/literature-review",
            json={"document_ids": document_ids, "topic": topic},
        ).json()

    # -- export -------------------------------------------------------------
    def export_chat(self, session_id: str, *, fmt: str) -> bytes:
        return self._request("GET", f"/export/chats/{session_id}", params={"fmt": fmt}).content
