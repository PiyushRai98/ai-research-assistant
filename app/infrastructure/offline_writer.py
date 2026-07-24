"""Task-aware, deterministic offline text generation.

When no live LLM endpoint is reachable, the :class:`EchoClient` uses these
functions to produce genuinely useful, well-structured output for each research
task instead of a generic extractive blob. Everything is derived strictly from
the retrieved context and preserves the ``[n]`` citation markers, so answers
stay grounded and verifiable. A real LLM, when configured, bypasses this module
entirely.

The public entry point is :func:`compose`, which inspects the task instruction
(the "question") and dispatches to the matching writer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Small stop-word set used when picking a "salient" term to blank out / define.
_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "as",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "their",
        "there",
        "here",
        "from",
        "at",
        "into",
        "over",
        "under",
        "between",
        "such",
        "which",
        "while",
        "using",
        "used",
        "use",
        "can",
        "may",
        "will",
        "not",
        "no",
        "than",
        "then",
        "also",
        "more",
        "most",
        "each",
        "other",
        "some",
        "any",
        "all",
        "both",
        "via",
        "within",
        "across",
        "per",
        "due",
        "one",
        "two",
        "three",
    ]
)

_NOT_FOUND = "The answer could not be found in the provided documents."
_CONTEXT_RE = re.compile(r"CONTEXT:\n(.*?)\n\nQUESTION:", re.DOTALL)
_QUESTION_RE = re.compile(r"QUESTION:\s*(.*?)\s*$", re.DOTALL)
_EQUATION_REF_RE = re.compile(r"\(\d+\)")
_ENTRY_HEAD_RE = re.compile(r"^\[(\d+)\]\s*(?:\(Document:\s*(.*?),\s*page\s*(\d+)\))?", re.DOTALL)


@dataclass(frozen=True)
class ContextEntry:
    """One numbered passage parsed from the composed RAG prompt."""

    marker: int
    document: str
    page: int | None
    text: str


# --------------------------------------------------------------------------
# Prompt parsing
# --------------------------------------------------------------------------
def parse_prompt(prompt: str) -> tuple[list[ContextEntry], str]:
    """Extract the numbered context entries and the task/question text."""
    context_match = _CONTEXT_RE.search(prompt)
    question_match = _QUESTION_RE.search(prompt)
    question = question_match.group(1).strip() if question_match else ""
    if not context_match:
        return [], question

    entries: list[ContextEntry] = []
    for block in context_match.group(1).strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        head = _ENTRY_HEAD_RE.match(block)
        if not head:
            continue
        marker = int(head.group(1))
        document = (head.group(2) or "the document").strip()
        page = int(head.group(3)) if head.group(3) else None
        # Body = everything after the first line (the "[n] (Document ...)" header).
        body = block.split("\n", 1)[1].strip() if "\n" in block else ""
        entries.append(ContextEntry(marker=marker, document=document, page=page, text=body))
    return entries, question


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def _sentences(text: str) -> list[str]:
    """Split text into clean, meaningful sentences."""
    # Normalise whitespace, then split on sentence terminators.
    normalised = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", normalised)
    return [p.strip() for p in parts if len(p.strip()) > 25]


# Markers of bibliography / boilerplate lines that make poor prose to quote.
_REFERENCE_MARKERS = (
    "issn",
    "isbn",
    "vol.",
    "vol ",
    "pp.",
    "springer",
    "et al",
    "no.",
    "www.",
    "http",
    "doi",
    "©",
    "proceedings",
    "journal of",
)


def _is_quality(sentence: str) -> bool:
    """Heuristically reject reference lists, equations, and symbol soup.

    Real papers contain bibliographies, numbered equations, and math symbols
    that extract as low-value text. This filter keeps sentences that read like
    prose: enough alphabetic words, not dominated by digits/symbols, and not
    matching common citation/reference patterns.
    """
    words = re.findall(r"[A-Za-z]{2,}", sentence)
    if len(words) < 6:
        return False
    if re.match(r"^\s*\[\d+\]", sentence):  # starts with a "[n]" reference
        return False
    if "=" in sentence:  # an equation, not prose
        return False
    if len(_EQUATION_REF_RE.findall(sentence)) >= 2:  # "(1) ... (2) ..." numbering
        return False
    low = sentence.lower()
    if any(marker in low for marker in _REFERENCE_MARKERS):
        return False
    length = len(sentence)
    digit_ratio = sum(ch.isdigit() for ch in sentence) / length
    if digit_ratio > 0.12:  # equation- or citation-heavy
        return False
    # Symbols beyond ordinary punctuation (math glyphs, unrenderable boxes).
    allowed = set(" .,;:'\"-()%/&")
    symbol_ratio = sum(1 for ch in sentence if not (ch.isalnum() or ch in allowed)) / length
    return symbol_ratio <= 0.08


def _quality_pairs(entries: list[ContextEntry]) -> list[tuple[str, int]]:
    """Prose-quality (sentence, marker) pairs, falling back to all if none pass."""
    pairs = _collect_sentences(entries)
    filtered = [(s, m) for s, m in pairs if _is_quality(s)]
    return filtered or pairs


def _salient_term(sentence: str) -> str | None:
    """Pick the most 'content-bearing' word in a sentence to blank/define.

    Skips the sentence-initial word (awkward to blank) and adverbs ending in
    ``-ly`` (weak quiz answers), preferring capitalised or longer nouns.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z-]{3,}", sentence)
    if len(tokens) <= 1:
        return None
    candidates = [
        word
        for word in tokens[1:]  # never blank the first word
        if word.lower() not in _STOPWORDS and not word.lower().endswith("ly")
    ]
    if not candidates:
        return None
    # Prefer capitalised (proper/technical) terms, else the longest word.
    capitalised = [w for w in candidates if w[0].isupper()]
    pool = capitalised or candidates
    return max(pool, key=len)


def _collect_sentences(entries: list[ContextEntry]) -> list[tuple[str, int]]:
    """Flatten entries into (sentence, marker) pairs, de-duplicated."""
    seen: set[str] = set()
    pairs: list[tuple[str, int]] = []
    for entry in entries:
        for sentence in _sentences(entry.text):
            key = sentence.lower()
            if key not in seen:
                seen.add(key)
                pairs.append((sentence, entry.marker))
    return pairs


def _rank_by_overlap(pairs: list[tuple[str, int]], query: str) -> list[tuple[str, int]]:
    """Order sentences by term overlap with the query (most relevant first)."""
    terms = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 3}
    if not terms:
        return pairs
    return sorted(
        pairs,
        key=lambda pair: sum(t in pair[0].lower() for t in terms),
        reverse=True,
    )


# --------------------------------------------------------------------------
# Task writers
# --------------------------------------------------------------------------
def write_summary(entries: list[ContextEntry], *, heading: str = "Summary") -> str:
    """Produce a concise, bulleted summary with citation markers."""
    pairs = _quality_pairs(entries)
    if not pairs:
        return _NOT_FOUND
    bullets = [f"- {sentence} [{marker}]" for sentence, marker in pairs[:6]]
    return f"### {heading}\n\n" + "\n".join(bullets)


def write_extract(entries: list[ContextEntry], *, kind: str, keywords: list[str]) -> str:
    """Extract sentences relevant to a facet (methodology/limitations/future)."""
    pairs = _quality_pairs(entries)
    matched = [(s, m) for s, m in pairs if any(k in s.lower() for k in keywords)]
    chosen = matched or pairs[:3]
    if not chosen:
        return _NOT_FOUND
    bullets = [f"- {sentence} [{marker}]" for sentence, marker in chosen[:6]]
    return f"### {kind}\n\n" + "\n".join(bullets)


def write_explain(entries: list[ContextEntry], *, concept: str) -> str:
    """Explain a concept using the passages that mention it."""
    pairs = _rank_by_overlap(_quality_pairs(entries), concept)
    if not pairs:
        return _NOT_FOUND
    relevant = [(s, m) for s, m in pairs if concept.lower() in s.lower()] or pairs
    body = " ".join(f"{s} [{m}]" for s, m in relevant[:3])
    return f"**{concept.strip().capitalize()}** — {body}"


def write_quiz(entries: list[ContextEntry], *, num_questions: int) -> str:
    """Generate grounded short-answer (fill-in-the-blank) questions.

    Offline generation cannot invent plausible multiple-choice distractors, so
    it produces honest fill-in-the-blank self-check questions. Output is pure
    Markdown (questions first, answer key last) so it renders safely in the UI.
    """
    pairs = _quality_pairs(entries)
    questions: list[str] = []
    answers: list[str] = []
    for sentence, marker in pairs:
        term = _salient_term(sentence)
        if not term:
            continue
        index = len(questions) + 1
        blanked = re.sub(rf"\b{re.escape(term)}\b", "**[blank]**", sentence, count=1)
        questions.append(f"**Q{index}.** {blanked} [{marker}]")
        answers.append(f"{index}. {term}")
        if len(questions) >= num_questions:
            break
    if not questions:
        return _NOT_FOUND
    body = "\n\n".join(questions)
    answer_key = "  ·  ".join(answers)
    return f"### Self-check quiz\n\n{body}\n\n**Answer key:** {answer_key}"


def write_flashcards(entries: list[ContextEntry], *, num_cards: int) -> str:
    """Generate term/definition flashcards grounded in the passages."""
    pairs = _quality_pairs(entries)
    cards: list[str] = []
    for sentence, marker in pairs:
        term = _salient_term(sentence)
        if not term:
            continue
        cards.append(f"**Card {len(cards) + 1} — {term}**  \n" f"{sentence} [{marker}]")
        if len(cards) >= num_cards:
            break
    if not cards:
        return _NOT_FOUND
    return "### Flashcards\n\n" + "\n\n".join(cards)


def write_compare(entries: list[ContextEntry]) -> str:
    """Contrast passages grouped by their source document.

    Every source document is always represented: quality-filtered sentences are
    preferred, falling back to that document's raw sentences if the filter left
    it empty (so a paper is never silently dropped from the comparison).
    """
    raw: dict[str, list[tuple[str, int]]] = {}
    quality: dict[str, list[tuple[str, int]]] = {}
    for entry in entries:
        for sentence in _sentences(entry.text):
            raw.setdefault(entry.document, []).append((sentence, entry.marker))
            if _is_quality(sentence):
                quality.setdefault(entry.document, []).append((sentence, entry.marker))
    if not raw:
        return _NOT_FOUND
    sections = ["### Comparison", ""]
    for document, raw_pairs in raw.items():
        pairs = quality.get(document) or raw_pairs
        sections.append(f"**{document}**")
        sections.extend(f"- {s} [{m}]" for s, m in pairs[:3])
        sections.append("")
    return "\n".join(sections).strip()


def write_generic(entries: list[ContextEntry], query: str) -> str:
    """Answer a free-form question with the most relevant grounded sentences."""
    pairs = _rank_by_overlap(_quality_pairs(entries), query)
    if not pairs:
        return _NOT_FOUND
    body = " ".join(f"{s} [{m}]" for s, m in pairs[:3])
    return body


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
def compose(prompt: str) -> str:
    """Parse the RAG prompt and produce task-appropriate grounded output."""
    entries, question = parse_prompt(prompt)
    if not entries:
        return _NOT_FOUND

    q = question.lower()

    # Order matters: check the most specific tasks first.
    if "flashcard" in q:
        num = _first_int(q, default=8)
        return write_flashcards(entries, num_cards=num)
    if "quiz" in q:
        num = _first_int(q, default=5)
        return write_quiz(entries, num_questions=num)
    if "methodology" in q:
        return write_extract(
            entries,
            kind="Methodology",
            keywords=["method", "dataset", "train", "evaluat", "experiment", "model"],
        )
    if "limitation" in q:
        return write_extract(
            entries,
            kind="Limitations",
            keywords=["limitation", "however", "cannot", "struggl", "difficult", "cost"],
        )
    if "future work" in q or "future research" in q:
        return write_extract(
            entries,
            kind="Future work",
            keywords=["future", "next", "further", "could", "extend", "explore"],
        )
    if "literature review" in q:
        return write_summary(entries, heading="Literature review")
    if "compare" in q or "contrast" in q:
        return write_compare(entries)
    if "summary" in q or "summarize" in q or "summarise" in q:
        return write_summary(entries)
    if q.startswith("explain") or "explain the concept" in q:
        concept = _extract_quoted(question) or question
        return write_explain(entries, concept=concept)

    return write_generic(entries, question)


def _first_int(text: str, *, default: int) -> int:
    match = re.search(r"\b(\d{1,2})\b", text)
    return int(match.group(1)) if match else default


def _extract_quoted(text: str) -> str | None:
    match = re.search(r"'([^']+)'", text)
    return match.group(1) if match else None
