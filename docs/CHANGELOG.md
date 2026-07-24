# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-07-22

### Added
- Clean Architecture foundation: `domain`, `application`, `infrastructure`,
  `backend`, `frontend`, and `shared` layers.
- Type-safe, environment-driven configuration and structured logging (uploads,
  embedding time, retrieval, LLM latency, token usage).
- PDF processing via PyMuPDF with metadata/page extraction and malformed-file
  handling.
- Recursive, citation-aware chunking with configurable size/overlap.
- Selectable embeddings (BGE / MiniLM) with a deterministic offline fallback.
- FAISS vector store with a NumPy fallback: incremental indexing, deletion,
  metadata filtering, similarity + MMR retrieval, persistence.
- Citation-grounded RAG pipeline with no-hallucination guarantee and a citation
  engine that never fabricates sources.
- Pluggable LLM client (IBM Granite / Llama 3 via OpenAI-compatible endpoints)
  with an offline echo fallback and streaming support.
- FastAPI backend: documents, chat, search, AI features, export, dashboard,
  optional auth, uniform error handling, OpenAPI docs.
- AI features: summaries, methodology/limitations/future-work extraction,
  concept explanation, quizzes, flashcards, paper comparison, literature review,
  and APA/IEEE/MLA/BibTeX citations.
- Streamlit frontend implemented strictly from `DESIGN.md` (monochrome core,
  pastel color-block sections, pill buttons, Inter/JetBrains Mono type).
- Export to Markdown and PDF with citations preserved.
- Test suite (unit + integration) with 80%+ coverage, running fully offline.
- Docker + Docker Compose, GitHub Actions CI (lint/test/build), pre-commit, and
  full documentation set.

[0.1.0]: https://example.com/releases/0.1.0
