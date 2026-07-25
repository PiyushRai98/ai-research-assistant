"""Documents view: multi-file upload with progress, listing, and deletion."""

from __future__ import annotations

import streamlit as st

from app.frontend import components as ui
from app.frontend.api_client import APIError, ResearchClient

_STATUS_LABEL = {
    "indexed": "● Indexed",
    "processing": "◐ Processing",
    "pending": "○ Pending",
    "failed": "✕ Failed",
}


def render(client: ResearchClient) -> None:
    """Render the document library and uploader."""
    ui.display_title("Documents", size="display-lg")
    ui.body(
        "Upload research papers as PDF. Each is validated, chunked, embedded, "
        "and indexed for retrieval.",
        size="body-lg",
    )

    uploaded = st.file_uploader(
        "Drag and drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Multiple files supported. Duplicates are detected automatically.",
    )

    if uploaded and st.button("Process uploads", type="primary"):
        progress = st.progress(0.0, text="Starting…")
        total = len(uploaded)
        for index, file in enumerate(uploaded, start=1):
            progress.progress((index - 1) / total, text=f"Processing {file.name} ({index}/{total})")
            try:
                data = file.getvalue()
                result = client.upload_document(filename=file.name, data=data)
                if result["status"] == "indexed":
                    st.success(f"Indexed {result['filename']} · {result['chunk_count']} chunks")
                else:
                    st.warning(
                        f"{result['filename']}: {result.get('error_message') or result['status']}"
                    )
            except APIError as exc:
                if exc.code == "duplicate":
                    st.info(f"{file.name} is already in your library.")
                else:
                    st.error(f"{file.name}: {exc.message}")
        progress.progress(1.0, text="Done")

    st.markdown("<hr style='border:none;border-top:1px solid #f1f1f1;'>", unsafe_allow_html=True)
    ui.eyebrow("Library")

    try:
        payload = client.list_documents()
    except APIError as exc:
        st.error(exc.message)
        return

    documents = payload["documents"]
    if not documents:
        ui.empty_state(
            variant="cream",
            title="No documents yet",
            message="Your uploaded papers will appear here, ready to query.",
        )
        return

    for doc in documents:
        cols = st.columns([6, 2, 2])
        with cols[0]:
            meta = doc["metadata"]
            title = meta.get("title") or doc["filename"]
            ui.card(
                title,
                f"{meta.get('page_count', 0)} pages · {doc['chunk_count']} chunks",
                caption=_STATUS_LABEL.get(doc["status"], doc["status"]),
            )
        with cols[1]:
            st.write("")
            if st.button(
                "Cite", key=f"cite_{doc['id']}", type="secondary", use_container_width=True
            ):
                st.session_state[f"show_cite_{doc['id']}"] = True
        with cols[2]:
            st.write("")
            if st.button("Delete", key=f"del_{doc['id']}", use_container_width=True):
                try:
                    client.delete_document(doc["id"])
                    st.rerun()
                except APIError as exc:
                    st.error(exc.message)

        if st.session_state.get(f"show_cite_{doc['id']}"):
            style = st.selectbox(
                "Citation style",
                ["apa", "ieee", "mla", "bibtex"],
                key=f"style_{doc['id']}",
            )
            try:
                citation = client.citation(doc["id"], style=style)
                st.code(citation["citation"], language="text")
            except APIError as exc:
                st.error(exc.message)
