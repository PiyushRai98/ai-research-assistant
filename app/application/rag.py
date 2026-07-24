"""Retrieval-Augmented Generation engine.

Implements the pipeline from the brief:

    question -> retriever -> context -> LLM -> answer -> citations

Guarantees:
* When retrieval yields no usable context, the engine short-circuits and
  returns a clear "answer could not be found" response with ``context_found``
  set to False — it never calls the model to hallucinate.
* Every returned answer carries citations resolved from real retrieved chunks.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence

from app.application.citations import resolve_citations
from app.application.prompts import (
    RAG_SYSTEM_PROMPT,
    build_history_block,
    build_rag_prompt,
)
from app.application.retrieval import RetrievalService
from app.domain.models import Answer, ScoredChunk
from app.domain.ports import LLMClient
from app.shared.logging import get_logger, log_llm

logger = get_logger("rag")

_NOT_FOUND = "The answer could not be found in the provided documents."


class RAGService:
    """Orchestrates retrieval, generation, and citation extraction."""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        llm_client: LLMClient,
    ) -> None:
        self._retrieval = retrieval_service
        self._llm = llm_client

    def _prepare(
        self,
        question: str,
        document_ids: Sequence[str] | None,
        history: Iterable[tuple[str, str]] | None,
    ) -> tuple[list[ScoredChunk], float, str]:
        """Retrieve context and compose the prompt (shared by sync/stream paths)."""
        chunks, retrieval_ms = self._retrieval.retrieve(question, document_ids=document_ids)
        history_block = build_history_block(list(history)) if history else ""
        prompt = build_rag_prompt(question, chunks, history=history_block)
        return chunks, retrieval_ms, prompt

    def answer(
        self,
        question: str,
        *,
        document_ids: Sequence[str] | None = None,
        history: Iterable[tuple[str, str]] | None = None,
    ) -> Answer:
        """Produce a grounded, cited answer for a question."""
        chunks, retrieval_ms, prompt = self._prepare(question, document_ids, history)

        if not chunks:
            return Answer(
                text=_NOT_FOUND,
                citations=[],
                context_found=False,
                retrieval_ms=retrieval_ms,
            )

        started = time.perf_counter()
        text, prompt_tokens, completion_tokens = self._llm.complete(
            system=RAG_SYSTEM_PROMPT, prompt=prompt
        )
        llm_ms = (time.perf_counter() - started) * 1000.0
        log_llm(
            provider=self._llm.name.split("/")[0],
            model=self._llm.name,
            duration_ms=llm_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        citations = resolve_citations(text, chunks)
        context_found = _NOT_FOUND.lower() not in text.lower()
        return Answer(
            text=text,
            citations=citations if context_found else [],
            context_found=context_found,
            llm_ms=llm_ms,
            retrieval_ms=retrieval_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def stream_answer(
        self,
        question: str,
        *,
        document_ids: Sequence[str] | None = None,
        history: Iterable[tuple[str, str]] | None = None,
    ) -> Iterator[str]:
        """Yield answer text incrementally (citations are resolved by callers).

        Streaming yields raw model tokens for responsive UIs. If no context is
        found, a single not-found message is yielded and the model is not hit.
        """
        chunks, _retrieval_ms, prompt = self._prepare(question, document_ids, history)
        if not chunks:
            yield _NOT_FOUND
            return
        yield from self._llm.stream(system=RAG_SYSTEM_PROMPT, prompt=prompt)

    def resolve_stream_citations(
        self,
        answer_text: str,
        question: str,
        *,
        document_ids: Sequence[str] | None = None,
    ):  # noqa: ANN201 - returns list[Citation]
        """Resolve citations after a streamed answer completes.

        Re-runs retrieval (cheap, cached embeddings) to obtain the same context
        ordering used for generation, then maps markers to sources.
        """
        chunks, _ = self._retrieval.retrieve(question, document_ids=document_ids)
        return resolve_citations(answer_text, chunks)
