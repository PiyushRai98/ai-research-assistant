"""Streamlit entrypoint for the AI Research Assistant.

Bootstraps the DESIGN.md theme, resolves the API client, renders the top-nav /
sidebar navigation, and routes to the selected view. Run with:

    streamlit run app/frontend/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# When launched via `streamlit run app/frontend/app.py`, Streamlit puts this
# file's directory on sys.path rather than the project root, so the `app`
# package would not be importable. Ensure the project root is on sys.path
# before importing any first-party modules.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app.frontend import components as ui  # noqa: E402
from app.frontend.api_client import APIClient, APIError, ResearchClient  # noqa: E402
from app.frontend.theme import build_css  # noqa: E402
from app.frontend.views import (  # noqa: E402
    ai_tools,
    chat,
    dashboard,
    documents,
    export,
    search,
    viewer,
)

# Navigation registry: label -> (eyebrow tag, render function).
PAGES = {
    "Dashboard": ("Overview", dashboard.render),
    "Documents": ("Library", documents.render),
    "Chat": ("Ask", chat.render),
    "Search": ("Find", search.render),
    "Viewer": ("Read", viewer.render),
    "AI Tools": ("Synthesize", ai_tools.render),
    "Export": ("Share", export.render),
}


@st.cache_resource
def get_client() -> tuple[ResearchClient, bool]:
    """Resolve the client to use, preferring a remote backend when reachable.

    Returns ``(client, embedded)``. When ``API_BASE_URL`` points at a live
    FastAPI backend, that HTTP client is used (multi-user, scalable). When no
    backend responds — notably on Streamlit Community Cloud, which can only
    run this single process — the frontend transparently falls back to an
    in-process :class:`EmbeddedClient` that runs the same application services
    directly, so the app remains fully functional with zero separate hosting.
    """
    base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
    remote = APIClient(base_url)
    try:
        remote.health()
        return remote, False
    except APIError:
        from app.frontend.embedded_client import EmbeddedClient

        return EmbeddedClient(), True


def _render_nav() -> str:
    """Render the sidebar navigation and return the selected page label."""
    with st.sidebar:
        st.markdown('<div class="eyebrow">AI Research</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="display-lg" style="margin:8px 0 24px;">Assistant</div>',
            unsafe_allow_html=True,
        )
        if "page" not in st.session_state:
            st.session_state.page = "Dashboard"
        for label in PAGES:
            button_type = "primary" if st.session_state.page == label else "secondary"
            if st.button(label, key=f"nav_{label}", use_container_width=True, type=button_type):
                st.session_state.page = label
        st.markdown(
            '<div class="caption" style="margin-top:32px;">GUEST MODE</div>',
            unsafe_allow_html=True,
        )
    return st.session_state.page


def main() -> None:
    """Application entrypoint."""
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="◼",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(build_css(), unsafe_allow_html=True)

    client, embedded = get_client()

    # Connectivity banner (marquee ribbon per DESIGN.md).
    try:
        health = client.health()
        items = [
            "AI RESEARCH ASSISTANT",
            f"LLM · {health['llm'].upper()}",
            f"EMBEDDINGS · {health['embedding_model'].split('/')[-1].upper()}",
            "RAG WITH CITATIONS",
        ]
        if embedded:
            items.insert(1, "EMBEDDED MODE — RUNNING LOCALLY IN THIS PROCESS")
        ui.marquee(items)
    except APIError:
        ui.marquee(["BACKEND UNAVAILABLE — RELOAD TO RETRY"])

    page = _render_nav()
    eyebrow_tag, render = PAGES[page]

    ui.eyebrow(eyebrow_tag)
    render(client)
    ui.footer()


if __name__ == "__main__":
    main()
