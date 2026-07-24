# Architecture

The AI Research Assistant follows **Clean Architecture**. Dependencies point
strictly inward, so business rules never depend on frameworks, databases, or
model libraries.

```
┌─────────────────────────────────────────────────────────────┐
│ presentation                                                  │
│   app/backend   (FastAPI: routers, schemas, container, auth)  │
│   app/frontend  (Streamlit: views, components, theme)         │
└───────────────┬───────────────────────────────────────────────┘
                │ depends on
┌───────────────▼───────────────────────────────────────────────┐
│ application  (use cases)                                        │
│   documents · retrieval · rag · chat · ai_features · export     │
│   prompts · citations                                           │
└───────────────┬───────────────────────────────────────────────┘
                │ depends on
┌───────────────▼───────────────────────────────────────────────┐
│ domain                                                          │
│   models (entities, value objects) · ports (interfaces)         │
└───────────────▲───────────────────────────────────────────────┘
                │ implements ports
┌───────────────┴───────────────────────────────────────────────┐
│ infrastructure  (adapters)                                      │
│   pdf (PyMuPDF) · chunking · embeddings (ST/hashing) ·          │
│   vectorstore (FAISS/numpy) · database + chat_store (SQLite) ·  │
│   llm (OpenAI-compatible/echo)                                  │
└─────────────────────────────────────────────────────────────────┘

shared: config · logging · security · exceptions  (used everywhere)
```

## Layers

### Domain (`app/domain`)
Framework-agnostic. `models.py` defines immutable, validated entities
(`Document`, `Chunk`, `Citation`, `Answer`, `ChatSession`, …). `ports.py`
declares the interfaces (`PDFParser`, `Chunker`, `EmbeddingModel`,
`VectorStore`, `DocumentRepository`, `ChatRepository`, `LLMClient`) that the
application depends on. No imports from outer layers.

### Application (`app/application`)
Use cases orchestrating the domain through ports:
- `documents.py` — validate → dedup → persist → parse → chunk → embed → index.
- `retrieval.py` — embed query, run similarity/MMR search, threshold, log metrics.
- `rag.py` — retrieval → prompt → LLM → citation resolution; short-circuits to a
  "not found" answer when context is insufficient (no hallucination).
- `citations.py` — maps `[n]` markers to real chunks; never fabricates sources.
- `prompts.py` — system prompt + prompt-injection sanitisation of context.
- `chat.py`, `ai_features.py`, `export.py` — conversations, research tools, exports.

### Infrastructure (`app/infrastructure`)
Concrete adapters. Heavy dependencies (sentence-transformers, FAISS, PyMuPDF)
are imported lazily so the inner layers import and test without them. Each
adapter has an offline-capable counterpart (`HashingEmbedding`,
`NumpyVectorStore`, `EchoClient`) selected automatically when the ML stack or a
model endpoint is unavailable.

### Presentation
- **Backend** (`app/backend`): FastAPI routers, Pydantic DTOs, a single
  composition root (`container.py`) that wires ports to adapters, uniform
  exception handling that maps domain errors to safe HTTP responses, and
  optional token auth with guest fallback.
- **Frontend** (`app/frontend`): Streamlit views that call the API only.
  The visual system (`theme.py`, `components.py`) is a faithful implementation
  of `DESIGN.md` — tokens for colors, typography, spacing, radii, and the
  signature pastel color-block sections.

## Key design decisions

- **Ports & adapters** make every collaborator mockable and swappable
  (e.g. FAISS ↔ NumPy, Granite ↔ Llama ↔ echo).
- **Immutable domain models** (`pydantic`, `frozen=True`) prevent accidental
  state mutation and make reasoning about the pipeline simpler.
- **Citations are first-class** and always derived from retrieved chunks, so
  answers remain verifiable.
- **Graceful degradation** everywhere: missing models or endpoints downgrade to
  offline implementations instead of crashing.
- **Structured logging** records uploads, embedding time, retrieval, LLM
  latency, and token usage for observability.

## Data flow: asking a question

1. Frontend `POST /api/chats/{id}/ask` with the question and document scope.
2. `ChatService` loads history and delegates to `RAGService`.
3. `RetrievalService` embeds the query and runs MMR/similarity search over FAISS.
4. If no context passes the threshold → return the "not found" answer.
5. Otherwise `prompts.build_rag_prompt` composes a numbered, sanitised context.
6. `LLMClient` generates an answer with `[n]` markers.
7. `citations.resolve_citations` maps markers back to real chunks.
8. The turn is persisted; the `Answer` (text + citations + latency) is returned.
