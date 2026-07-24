"""Unit tests for citation formatting and export rendering."""

from __future__ import annotations

import pytest
from app.application.ai_features import (
    SUPPORTED_CITATION_STYLES,
    format_citation,
)
from app.application.export import (
    conversation_to_markdown,
    markdown_to_pdf_bytes,
    summary_to_markdown,
)
from app.domain.models import (
    ChatMessage,
    Citation,
    Document,
    DocumentMetadata,
    MessageRole,
)


def _document() -> Document:
    return Document(
        filename="attention.pdf",
        content_hash="h",
        size_bytes=1,
        metadata=DocumentMetadata(title="Attention Is All You Need", author="Vaswani"),
    )


@pytest.mark.parametrize("style", SUPPORTED_CITATION_STYLES)
def test_citation_formats_include_title(style: str) -> None:
    text = format_citation(_document(), style=style)
    assert "Attention Is All You Need" in text
    assert "Vaswani" in text


def test_bibtex_has_entry_shape() -> None:
    text = format_citation(_document(), style="bibtex")
    assert text.startswith("@article{")
    assert "title" in text


def test_unsupported_style_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        format_citation(_document(), style="chicago")


def test_conversation_markdown_includes_citations() -> None:
    citation = Citation(
        marker=1,
        document_id="d1",
        document_name="p.pdf",
        page_number=3,
        chunk_id="c1",
        quote="a grounded quote",
    )
    messages = [
        ChatMessage(role=MessageRole.USER, content="What is X?"),
        ChatMessage(role=MessageRole.ASSISTANT, content="X is Y [1].", citations=[citation]),
    ]
    md = conversation_to_markdown(messages, title="Session")
    assert "# Session" in md
    assert "a grounded quote" in md
    assert "p.pdf" in md


def test_summary_markdown_and_pdf_bytes() -> None:
    md = summary_to_markdown(title="Summary", body="Some body.", citations=[])
    assert "# Summary" in md
    pdf = markdown_to_pdf_bytes(md, title="Summary")
    assert pdf.startswith(b"%PDF-")
