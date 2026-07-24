"""Dashboard view: documents, statistics, storage, embeddings, chats."""

from __future__ import annotations

import streamlit as st

from app.frontend import components as ui
from app.frontend.api_client import APIClient, APIError


def _human_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def render(client: APIClient) -> None:
    """Render the overview dashboard."""
    ui.display_title("Your research, at a glance", size="display-lg")

    try:
        stats = client.dashboard()
    except APIError as exc:
        st.error(exc.message)
        ui.empty_state(
            variant="lime",
            title="Start the backend to see your stats",
            message="Once the API is running, upload a paper and this dashboard "
            "fills with live document, storage, and chat metrics.",
        )
        return

    row1 = st.columns(3)
    with row1[0]:
        ui.metric_tile("Documents", str(stats["document_count"]))
    with row1[1]:
        ui.metric_tile("Indexed", str(stats["indexed_count"]))
    with row1[2]:
        ui.metric_tile("Conversations", str(stats["chat_count"]))

    st.write("")
    row2 = st.columns(3)
    with row2[0]:
        ui.metric_tile("Total chunks", str(stats["total_chunks"]))
    with row2[1]:
        ui.metric_tile("Storage used", _human_bytes(stats["storage_bytes"]))
    with row2[2]:
        ui.metric_tile("Avg processing", f"{stats['avg_processing_ms']:.0f} ms")

    if stats["failed_count"]:
        st.warning(f"{stats['failed_count']} document(s) failed to process.")

    if stats["document_count"] == 0:
        ui.color_block(
            variant="lime",
            eyebrow_text="Get started",
            title="Upload your first paper",
            body_text="Head to Documents, drop in a PDF, and start asking grounded, "
            "cited questions in seconds.",
        )
    else:
        ui.color_block(
            variant="navy",
            eyebrow_text="Ready",
            title="Ask anything, get cited answers",
            body_text="Every response is grounded in your documents and backed by "
            "page-level citations you can verify.",
        )
