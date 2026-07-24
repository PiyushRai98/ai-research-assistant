"""Export utilities: conversation history and research artefacts.

Produces Markdown (always available) and PDF (via PyMuPDF, already a core
dependency — no extra library needed). Exports include citations so exported
research remains verifiable.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain.models import ChatMessage, Citation, MessageRole
from app.shared.exceptions import ResearchAssistantError
from app.shared.logging import get_logger

logger = get_logger("export")


def _format_citations(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = ["", "**Sources:**"]
    for citation in citations:
        lines.append(
            f"- [{citation.marker}] {citation.document_name}, p.{citation.page_number} "
            f"— “{citation.quote}”"
        )
    return "\n".join(lines)


def conversation_to_markdown(
    messages: list[ChatMessage],
    *,
    title: str = "Research Conversation",
) -> str:
    """Render a full conversation (with citations) as a Markdown document."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts = [f"# {title}", f"_Exported {stamp}_", ""]
    for message in messages:
        speaker = "You" if message.role is MessageRole.USER else "Assistant"
        parts.append(f"## {speaker}")
        parts.append(message.content)
        if message.role is MessageRole.ASSISTANT and message.citations:
            parts.append(_format_citations(message.citations))
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def summary_to_markdown(
    *,
    title: str,
    body: str,
    citations: list[Citation],
) -> str:
    """Render a single research artefact (e.g. summary) as Markdown."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts = [f"# {title}", f"_Generated {stamp}_", "", body]
    citation_block = _format_citations(citations)
    if citation_block:
        parts.append(citation_block)
    return "\n".join(parts).strip() + "\n"


def markdown_to_pdf_bytes(markdown_text: str, *, title: str = "Export") -> bytes:
    """Render Markdown text into a simple, readable PDF using PyMuPDF.

    This is intentionally a lightweight text renderer (headings, bullets, and
    paragraphs) rather than a full Markdown engine — it keeps exports dependency
    free while producing a clean, printable document.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ResearchAssistantError(
            "PDF export requires PyMuPDF, which is not installed."
        ) from exc

    document = fitz.open()
    page = document.new_page()
    margin = 56
    width = page.rect.width - 2 * margin
    cursor_y = margin
    line_height = 16

    def new_page_if_needed() -> None:
        nonlocal page, cursor_y
        if cursor_y > page.rect.height - margin:
            page = document.new_page()
            cursor_y = margin

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        font_size = 11
        text = line
        if line.startswith("# "):
            font_size, text = 20, line[2:]
        elif line.startswith("## "):
            font_size, text = 15, line[3:]
        elif line.startswith("- "):
            text = f"•  {line[2:]}"

        if not text:
            cursor_y += line_height
            new_page_if_needed()
            continue

        # Naive word-wrap to the page width.
        words = text.split(" ")
        current = ""
        char_budget = max(int(width / (font_size * 0.5)), 10)
        for word in words:
            if len(current) + len(word) + 1 > char_budget:
                page.insert_text((margin, cursor_y), current, fontsize=font_size)
                cursor_y += font_size + 4
                new_page_if_needed()
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            page.insert_text((margin, cursor_y), current, fontsize=font_size)
            cursor_y += font_size + 6
            new_page_if_needed()

    document.set_metadata({"title": title})
    data = document.tobytes()
    document.close()
    return data
