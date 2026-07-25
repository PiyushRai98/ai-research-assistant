"""AI Tools view: summaries, extraction, quiz, flashcards, comparison, review."""

from __future__ import annotations

import streamlit as st

from app.frontend import components as ui
from app.frontend.api_client import APIError, ResearchClient


def _indexed_docs(client: ResearchClient) -> dict[str, str]:
    docs = client.list_documents()["documents"]
    return {
        d["id"]: (d["metadata"].get("title") or d["filename"])
        for d in docs
        if d["status"] == "indexed"
    }


def _show_answer(result: dict) -> None:
    st.markdown(result["text"])
    ui.citation_list(result.get("citations", []))


def render(client: ResearchClient) -> None:
    """Render the AI research tools."""
    ui.display_title("AI research tools", size="display-lg")

    try:
        labels = _indexed_docs(client)
    except APIError as exc:
        st.error(exc.message)
        return

    if not labels:
        ui.empty_state(
            variant="coral",
            title="No indexed documents",
            message="Upload and index a paper to unlock summaries, quizzes, "
            "comparisons, and literature reviews.",
        )
        return

    tabs = st.tabs(["Summary", "Extract", "Explain", "Quiz", "Flashcards", "Compare", "Review"])

    def pick(key: str) -> str:
        return st.selectbox(
            "Document", list(labels.keys()), format_func=lambda i: labels[i], key=key
        )

    with tabs[0]:
        doc = pick("sum_doc")
        if st.button("Summarize", type="primary", key="sum_btn"):
            with st.spinner("Summarizing…"):
                try:
                    _show_answer(client.summarize(doc))
                except APIError as exc:
                    st.error(exc.message)

    with tabs[1]:
        doc = pick("ext_doc")
        kind = st.radio(
            "What to extract",
            ["Methodology", "Limitations", "Future work"],
            horizontal=True,
            key="ext_kind",
        )
        if st.button("Extract", type="primary", key="ext_btn"):
            fn = {
                "Methodology": client.methodology,
                "Limitations": client.limitations,
                "Future work": client.future_work,
            }[kind]
            with st.spinner("Extracting…"):
                try:
                    _show_answer(fn(doc))
                except APIError as exc:
                    st.error(exc.message)

    with tabs[2]:
        doc = pick("exp_doc")
        concept = st.text_input("Concept to explain", key="exp_concept")
        if st.button("Explain", type="primary", key="exp_btn") and concept:
            with st.spinner("Explaining…"):
                try:
                    _show_answer(client.explain(document_id=doc, concept=concept))
                except APIError as exc:
                    st.error(exc.message)

    with tabs[3]:
        doc = pick("quiz_doc")
        n = st.slider("Questions", 1, 20, 5, key="quiz_n")
        if st.button("Generate quiz", type="primary", key="quiz_btn"):
            with st.spinner("Building quiz…"):
                try:
                    _show_answer(client.quiz(document_id=doc, num_questions=n))
                except APIError as exc:
                    st.error(exc.message)

    with tabs[4]:
        doc = pick("fc_doc")
        n = st.slider("Cards", 1, 30, 8, key="fc_n")
        if st.button("Generate flashcards", type="primary", key="fc_btn"):
            with st.spinner("Building flashcards…"):
                try:
                    _show_answer(client.flashcards(document_id=doc, num_cards=n))
                except APIError as exc:
                    st.error(exc.message)

    with tabs[5]:
        chosen = st.multiselect(
            "Papers to compare (2+)",
            list(labels.keys()),
            format_func=lambda i: labels[i],
            key="cmp_docs",
        )
        aspect = st.text_input("Focus aspect (optional)", key="cmp_aspect")
        if st.button("Compare", type="primary", key="cmp_btn"):
            if len(chosen) < 2:
                st.warning("Select at least two papers.")
            else:
                with st.spinner("Comparing…"):
                    try:
                        _show_answer(client.compare(document_ids=chosen, aspect=aspect or None))
                    except APIError as exc:
                        st.error(exc.message)

    with tabs[6]:
        chosen = st.multiselect(
            "Papers to synthesize",
            list(labels.keys()),
            format_func=lambda i: labels[i],
            key="lit_docs",
        )
        topic = st.text_input("Topic (optional)", key="lit_topic")
        if st.button("Draft literature review", type="primary", key="lit_btn"):
            if not chosen:
                st.warning("Select at least one paper.")
            else:
                with st.spinner("Synthesizing…"):
                    try:
                        _show_answer(
                            client.literature_review(document_ids=chosen, topic=topic or None)
                        )
                    except APIError as exc:
                        st.error(exc.message)
