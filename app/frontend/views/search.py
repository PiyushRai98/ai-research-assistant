"""Search view: semantic search across all or selected documents."""

from __future__ import annotations

import streamlit as st

from app.frontend import components as ui
from app.frontend.api_client import APIError, ResearchClient


def render(client: ResearchClient) -> None:
    """Render the semantic search interface."""
    ui.display_title("Search", size="display-lg")
    ui.body(
        "Semantic search finds passages by meaning, not just keywords, across "
        "your entire library.",
        size="body-lg",
    )

    query = st.text_input("Search your documents", placeholder="e.g. attention mechanism")
    top_k = st.slider("Results", min_value=1, max_value=20, value=5)

    if not query:
        ui.empty_state(
            variant="mint",
            title="Search by meaning",
            message="Type a concept or question to retrieve the most relevant "
            "passages, ranked by semantic similarity.",
        )
        return

    try:
        payload = client.search(query=query, document_ids=None, top_k=top_k)
    except APIError as exc:
        st.error(exc.message)
        return

    hits = payload["hits"]
    st.markdown(
        f'<div class="caption">{len(hits)} RESULTS · {payload["elapsed_ms"]:.0f}MS</div>',
        unsafe_allow_html=True,
    )
    if not hits:
        st.info("No matching passages found.")
        return

    for hit in hits:
        ui.card(
            f"{hit['document_name']} · page {hit['page_number']}",
            hit["text"][:400] + ("…" if len(hit["text"]) > 400 else ""),
            caption=f"SCORE {hit['score']:.3f}",
        )
        st.write("")
