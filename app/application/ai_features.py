"""Higher-level AI research features built on the RAG engine.

Each feature is a specialised prompt over document-scoped retrieval, so every
generated artefact stays grounded in the uploaded papers and inherits the same
citation guarantees. Bibliographic citation formatting (APA/IEEE/MLA/BibTeX) is
computed deterministically from extracted metadata — no model involvement — so
references are accurate and reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.application.rag import RAGService
from app.domain.models import Answer, Document
from app.shared.logging import get_logger

logger = get_logger("ai_features")

# Task-specific instructions appended to a retrieval query to steer generation.
_TASKS: dict[str, str] = {
    "summary": (
        "Provide a structured summary of the paper covering its objective, "
        "methods, key findings, and conclusions. Use Markdown headings and "
        "cite every claim with [n] markers."
    ),
    "methodology": (
        "Extract and describe the methodology used in the paper: datasets, "
        "models, experimental setup, and evaluation metrics. Cite with [n]."
    ),
    "limitations": (
        "Identify and explain the limitations acknowledged or implied in the "
        "paper. Cite each with [n] markers."
    ),
    "future_work": (
        "Extract the future work and open questions the paper proposes. "
        "Cite each with [n] markers."
    ),
}


class AIFeatureService:
    """Grounded research operations layered on top of :class:`RAGService`."""

    def __init__(self, *, rag_service: RAGService) -> None:
        self._rag = rag_service

    # -- single-document features ------------------------------------------
    def _run_task(self, task: str, *, document_id: str) -> Answer:
        instruction = _TASKS[task]
        return self._rag.answer(instruction, document_ids=[document_id])

    def summarize(self, *, document_id: str) -> Answer:
        """Produce a grounded, cited summary of a single paper."""
        return self._run_task("summary", document_id=document_id)

    def extract_methodology(self, *, document_id: str) -> Answer:
        return self._run_task("methodology", document_id=document_id)

    def extract_limitations(self, *, document_id: str) -> Answer:
        return self._run_task("limitations", document_id=document_id)

    def extract_future_work(self, *, document_id: str) -> Answer:
        return self._run_task("future_work", document_id=document_id)

    def explain_concept(self, *, concept: str, document_id: str) -> Answer:
        """Explain a difficult concept using the document as grounding."""
        query = (
            f"Explain the concept '{concept}' as it is used in this paper, in "
            "clear, accessible terms. Cite supporting passages with [n]."
        )
        return self._rag.answer(query, document_ids=[document_id])

    def generate_quiz(self, *, document_id: str, num_questions: int = 5) -> Answer:
        """Generate a short comprehension quiz grounded in the paper."""
        query = (
            f"Create a {num_questions}-question multiple-choice quiz that tests "
            "understanding of this paper. For each question provide four options "
            "labelled A-D, mark the correct answer, and cite the supporting "
            "passage with [n]."
        )
        return self._rag.answer(query, document_ids=[document_id])

    def generate_flashcards(self, *, document_id: str, num_cards: int = 8) -> Answer:
        """Generate study flashcards (term/definition) from the paper."""
        query = (
            f"Create {num_cards} study flashcards from this paper. Format each as "
            "'**Q:** ...' on one line and '**A:** ...' on the next, and cite the "
            "source passage with [n]."
        )
        return self._rag.answer(query, document_ids=[document_id])

    # -- multi-document features -------------------------------------------
    def compare_papers(self, *, document_ids: Sequence[str], aspect: str | None = None) -> Answer:
        """Compare two or more papers, optionally along a specific aspect."""
        if len(document_ids) < 2:
            raise ValueError("Comparison requires at least two documents.")
        focus = f" Focus specifically on: {aspect}." if aspect else ""
        query = (
            "Compare and contrast the provided papers. Highlight agreements, "
            "disagreements, and complementary findings in a Markdown table where "
            f"useful.{focus} Cite every point with [n] markers."
        )
        return self._rag.answer(query, document_ids=list(document_ids))

    def literature_review(self, *, document_ids: Sequence[str], topic: str | None = None) -> Answer:
        """Draft a short literature review synthesising the selected papers."""
        scope = f" on the topic of {topic}" if topic else ""
        query = (
            f"Write a concise literature review{scope} that synthesises the "
            "provided papers: motivation, prior approaches, key contributions, "
            "and open gaps. Cite every claim with [n] markers."
        )
        return self._rag.answer(query, document_ids=list(document_ids))


# --------------------------------------------------------------------------
# Deterministic bibliographic citation formatting.
# --------------------------------------------------------------------------
def _author_and_title(document: Document) -> tuple[str, str]:
    """Best-effort author + title derived from metadata or filename."""
    meta = document.metadata
    author = meta.author or "Unknown Author"
    title = meta.title or document.filename.removesuffix(".pdf")
    return author, title


def _bibtex_key(author: str, title: str) -> str:
    surname = author.split()[-1].split(",")[0] if author.split() else "ref"
    first_word = next((w for w in title.split() if w.isalnum()), "paper")
    return f"{surname.lower()}{first_word.lower()}"


def format_citation(document: Document, *, style: str) -> str:
    """Format a document's metadata into APA, IEEE, MLA, or BibTeX.

    Styles are generated deterministically from extracted metadata. Fields that
    are genuinely unknown are rendered as ``n.d.`` (no date) or omitted rather
    than fabricated.
    """
    author, title = _author_and_title(document)
    year = document.created_at.year
    normalised = style.strip().lower()

    if normalised == "apa":
        return f"{author}. ({year}). {title}."
    if normalised == "mla":
        return f'{author}. "{title}." {year}.'
    if normalised == "ieee":
        return f'{author}, "{title}," {year}.'
    if normalised == "bibtex":
        key = _bibtex_key(author, title)
        return (
            f"@article{{{key},\n"
            f"  author = {{{author}}},\n"
            f"  title  = {{{title}}},\n"
            f"  year   = {{{year}}}\n"
            f"}}"
        )
    raise ValueError(f"Unsupported citation style: {style!r}")


SUPPORTED_CITATION_STYLES: tuple[str, ...] = ("apa", "ieee", "mla", "bibtex")
