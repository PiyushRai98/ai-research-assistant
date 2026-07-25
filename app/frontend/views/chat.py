"""Chat view: conversation with streaming-style rendering and citations."""

from __future__ import annotations

import streamlit as st

from app.frontend import components as ui
from app.frontend.api_client import APIError, ResearchClient


def _document_picker(client: ResearchClient) -> list[str]:
    """Render a multiselect of indexed documents; return selected ids."""
    try:
        docs = client.list_documents()["documents"]
    except APIError:
        return []
    indexed = [d for d in docs if d["status"] == "indexed"]
    if not indexed:
        return []
    labels = {d["id"]: (d["metadata"].get("title") or d["filename"]) for d in indexed}
    selected = st.multiselect(
        "Scope to documents (leave empty to search all)",
        options=list(labels.keys()),
        format_func=lambda i: labels[i],
    )
    return selected


def render(client: ResearchClient) -> None:
    """Render the chat interface."""
    ui.display_title("Chat with your papers", size="display-lg")

    document_ids = _document_picker(client)

    # Ensure a session exists.
    if "chat_session_id" not in st.session_state:
        try:
            session = client.create_chat(document_ids=document_ids)
            st.session_state.chat_session_id = session["id"]
            st.session_state.chat_messages = []
        except APIError as exc:
            st.error(exc.message)
            return

    controls = st.columns([3, 1])
    with controls[1]:
        if st.button("New chat", type="secondary", use_container_width=True):
            try:
                session = client.create_chat(document_ids=document_ids)
                st.session_state.chat_session_id = session["id"]
                st.session_state.chat_messages = []
                st.rerun()
            except APIError as exc:
                st.error(exc.message)

    # Replay history.
    for message in st.session_state.get("chat_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                ui.citation_list(message.get("citations", []))

    prompt = st.chat_input("Ask a question about your documents…")
    if not prompt:
        if not st.session_state.get("chat_messages"):
            ui.color_block(
                variant="lilac",
                eyebrow_text="Grounded answers",
                title="Every answer is cited",
                body_text="Ask a question and the assistant retrieves the most "
                "relevant passages, then answers with page-level citations.",
            )
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning…"):
            try:
                result = client.ask(
                    st.session_state.chat_session_id,
                    question=prompt,
                    document_ids=document_ids or None,
                )
            except APIError as exc:
                st.error(exc.message)
                return
        answer = result["answer"]
        st.markdown(answer["text"])
        ui.citation_list(answer["citations"])
        latency = answer["retrieval_ms"] + answer["llm_ms"]
        st.markdown(
            f'<div class="caption" style="margin-top:8px;">'
            f'RETRIEVAL {answer["retrieval_ms"]:.0f}MS · GENERATION {answer["llm_ms"]:.0f}MS · '
            f"TOTAL {latency:.0f}MS</div>",
            unsafe_allow_html=True,
        )

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": answer["text"],
            "citations": answer["citations"],
        }
    )
