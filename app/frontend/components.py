"""Reusable UI primitives implementing DESIGN.md components.

Each helper renders one documented component (eyebrow label, color-block
section, template card, metric tile, citation chip, marquee strip) using the
tokens in ``theme.py``. Views compose these rather than hand-writing HTML, so
the design system stays consistent everywhere.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

import streamlit as st


def eyebrow(text: str) -> None:
    """Render a figmaMono uppercase section eyebrow (taxonomy label)."""
    st.markdown(f'<div class="eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


def display_title(text: str, *, size: str = "display-lg") -> None:
    """Render an oversized editorial headline (display-xl / display-lg)."""
    st.markdown(f'<div class="{size}">{html.escape(text)}</div>', unsafe_allow_html=True)


def body(text: str, *, size: str = "body") -> None:
    """Render body copy at a documented weight/size role."""
    st.markdown(f'<div class="{size}">{html.escape(text)}</div>', unsafe_allow_html=True)


def color_block(
    *,
    variant: str,
    eyebrow_text: str,
    title: str,
    body_text: str,
) -> None:
    """Render a signature color-block section (poster-style panel).

    ``variant`` is one of: lime, lilac, cream, mint, pink, coral, navy.
    """
    st.markdown(
        f"""
        <div class="color-block block-{variant}">
          <div class="eyebrow">{html.escape(eyebrow_text)}</div>
          <div class="headline" style="margin-top:12px;">{html.escape(title)}</div>
          <div class="body-lg" style="margin-top:16px;max-width:60%;">
            {html.escape(body_text)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def marquee(items: Sequence[str]) -> None:
    """Render the thin black marquee ribbon under the nav."""
    text = "  •  ".join(html.escape(item) for item in items)
    st.markdown(f'<div class="marquee-strip">{text}</div>', unsafe_allow_html=True)


def metric_tile(label: str, value: str) -> None:
    """Render a dashboard metric as a soft-surface tile."""
    st.markdown(
        f"""
        <div class="metric-tile">
          <div class="caption">{html.escape(label)}</div>
          <div class="metric-value" style="margin-top:8px;">{html.escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body_text: str, *, caption: str | None = None) -> None:
    """Render a hairline-bordered white card (elevation level 1)."""
    caption_html = (
        f'<div class="caption" style="margin-bottom:8px;">{html.escape(caption)}</div>'
        if caption
        else ""
    )
    st.markdown(
        f"""
        <div class="rc-card">
          {caption_html}
          <div class="card-title">{html.escape(title)}</div>
          <div class="body-sm" style="margin-top:8px;">{html.escape(body_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def citation_list(citations: list[dict]) -> None:
    """Render answer citations as mono chips followed by quote lines."""
    if not citations:
        return
    st.markdown('<div class="caption">Sources</div>', unsafe_allow_html=True)
    for citation in citations:
        chip = (
            f'<span class="citation-chip">[{citation["marker"]}] '
            f'{html.escape(citation["document_name"])} · p.{citation["page_number"]}</span>'
        )
        quote = html.escape(citation["quote"])
        st.markdown(
            f'{chip}<div class="body-sm" style="margin:4px 0 12px;">“{quote}”</div>',
            unsafe_allow_html=True,
        )


def footer() -> None:
    """Render the editorial footer wordmark."""
    st.markdown(
        """
        <div class="rc-footer">
          <span class="display-lg">Research</span>
          <div class="caption" style="margin-top:16px;">
            AI RESEARCH ASSISTANT · RETRIEVAL AUGMENTED · CITED ANSWERS
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(*, variant: str, title: str, message: str) -> None:
    """Render an empty-state prompt inside a color block (DESIGN.md states)."""
    color_block(
        variant=variant,
        eyebrow_text="Nothing here yet",
        title=title,
        body_text=message,
    )
