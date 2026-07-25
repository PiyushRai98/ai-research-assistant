"""Export view: download conversation history as Markdown or PDF."""

from __future__ import annotations

import streamlit as st

from app.frontend import components as ui
from app.frontend.api_client import APIError, ResearchClient


def render(client: ResearchClient) -> None:
    """Render the export interface for conversations."""
    ui.display_title("Export", size="display-lg")
    ui.body(
        "Download your conversations, with citations preserved, as Markdown or PDF.",
        size="body-lg",
    )

    try:
        sessions = client.list_chats()
    except APIError as exc:
        st.error(exc.message)
        return

    if not sessions:
        ui.empty_state(
            variant="lime",
            title="No conversations to export",
            message="Start a chat, then return here to export the transcript and " "its citations.",
        )
        return

    labels = {s["id"]: f"{s['title']} ({s['message_count']} messages)" for s in sessions}
    session_id = st.selectbox("Conversation", list(labels.keys()), format_func=lambda i: labels[i])

    cols = st.columns(2)
    with cols[0]:
        if st.button("Prepare Markdown", type="secondary", use_container_width=True):
            try:
                data = client.export_chat(session_id, fmt="markdown")
                st.download_button(
                    "Download .md",
                    data=data,
                    file_name="conversation.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            except APIError as exc:
                st.error(exc.message)
    with cols[1]:
        if st.button("Prepare PDF", type="primary", use_container_width=True):
            try:
                data = client.export_chat(session_id, fmt="pdf")
                st.download_button(
                    "Download .pdf",
                    data=data,
                    file_name="conversation.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except APIError as exc:
                st.error(exc.message)
