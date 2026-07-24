"""Unit tests for the task-aware offline writer."""

from __future__ import annotations

from app.application.prompts import build_rag_prompt
from app.domain.models import Chunk, ChunkMetadata, ScoredChunk
from app.infrastructure import offline_writer


def _ctx(texts: list[tuple[str, str, int]]) -> list[ScoredChunk]:
    """Build scored chunks from (text, document_name, page) tuples."""
    scored = []
    for i, (text, doc, page) in enumerate(texts):
        scored.append(
            ScoredChunk(
                chunk=Chunk(
                    text=text,
                    metadata=ChunkMetadata(
                        document_id="d1",
                        document_name=doc,
                        page_number=page,
                        chunk_index=i,
                    ),
                ),
                score=1.0 - i * 0.1,
            )
        )
    return scored


_PASSAGES = _ctx(
    [
        (
            "Transformers use self-attention to model long-range dependencies. "
            "The methodology trains on large corpora using the Adam optimizer.",
            "transformers.pdf",
            1,
        ),
        (
            "A limitation of transformers is quadratic memory cost. Future work "
            "explores sparse attention to scale to longer documents.",
            "transformers.pdf",
            2,
        ),
    ]
)


def _compose(question: str) -> str:
    return offline_writer.compose(build_rag_prompt(question, _PASSAGES))


def test_parse_prompt_extracts_entries() -> None:
    entries, question = offline_writer.parse_prompt(build_rag_prompt("What is X?", _PASSAGES))
    assert len(entries) == 2
    assert entries[0].marker == 1
    assert entries[0].document == "transformers.pdf"
    assert entries[0].page == 1
    assert question == "What is X?"


def test_summary_is_bulleted_and_cited() -> None:
    out = _compose("Provide a structured summary of the paper.")
    assert "### Summary" in out
    assert "- " in out
    assert "[1]" in out or "[2]" in out


def test_quiz_has_questions_and_answers() -> None:
    out = _compose("Create a 2-question multiple-choice quiz.")
    assert "quiz" in out.lower()
    assert "**Q1.**" in out
    assert "[blank]" in out  # fill-in-the-blank placeholder
    assert "Answer key" in out
    assert "<details>" not in out  # no raw HTML leaks into the UI


def test_flashcards_have_cards() -> None:
    out = _compose("Create 3 study flashcards from this paper.")
    assert "Flashcards" in out
    assert "**Card 1" in out


def test_methodology_extraction() -> None:
    out = _compose("Extract and describe the methodology used in the paper.")
    assert "Methodology" in out
    assert "[1]" in out


def test_limitations_extraction() -> None:
    out = _compose("Identify and explain the limitations of the paper.")
    assert "Limitations" in out


def test_future_work_extraction() -> None:
    out = _compose("Extract the future work the paper proposes.")
    assert "Future work" in out


def test_explain_targets_concept() -> None:
    out = _compose("Explain the concept 'self-attention' as used in this paper.")
    assert "self-attention" in out.lower()
    assert "[1]" in out


def test_compare_groups_by_document() -> None:
    passages = _ctx(
        [
            ("Transformers use attention over all tokens.", "a.pdf", 1),
            ("Recurrent networks process tokens sequentially.", "b.pdf", 1),
        ]
    )
    out = offline_writer.compose(
        build_rag_prompt("Compare and contrast the provided papers.", passages)
    )
    assert "a.pdf" in out and "b.pdf" in out
    assert "### Comparison" in out


def test_generic_question_returns_grounded_sentence() -> None:
    out = _compose("What do transformers use for dependencies?")
    assert "attention" in out.lower()
    assert "[1]" in out


def test_empty_context_reports_not_found() -> None:
    out = offline_writer.compose(build_rag_prompt("Anything?", []))
    assert "could not be found" in out.lower()
