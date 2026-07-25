"""Document viewer: preview a PDF and jump to cited pages."""

from __future__ import annotations

import base64

import streamlit as st

from app.frontend import components as ui
from app.frontend.api_client import APIError, ResearchClient


def _pdf_iframe(data: bytes, *, page: int) -> str:
    """Return an HTML iframe embedding the PDF, opened at ``page``."""
    b64 = base64.b64encode(data).decode("ascii")
    # #page=N is honoured by browser PDF viewers to jump to a cited page.
    return (
        f'<iframe src="data:application/pdf;base64,{b64}#page={page}" '
        f'width="100%" height="720" style="border:1px solid #e6e6e6;'
        f'border-radius:8px;"></iframe>'
    )


def render(client: ResearchClient) -> None:
    """Render the PDF viewer with page navigation."""
    ui.display_title("Document viewer", size="display-lg")

    try:
        docs = client.list_documents()["documents"]
    except APIError as exc:
        st.error(exc.message)
        return

    if not docs:
        ui.empty_state(
            variant="pink",
            title="Nothing to preview",
            message="Upload a document to read it here and jump straight to cited pages.",
        )
        return

    labels = {d["id"]: (d["metadata"].get("title") or d["filename"]) for d in docs}
    doc_id = st.selectbox("Document", options=list(labels.keys()), format_func=lambda i: labels[i])
    page = st.number_input("Jump to page", min_value=1, value=1, step=1)

    try:
        data = client.document_file_bytes(doc_id)
    except APIError as exc:
        st.error(f"Could not load the PDF: {exc.message}")
        return

    st.markdown(_pdf_iframe(data, page=int(page)), unsafe_allow_html=True)
