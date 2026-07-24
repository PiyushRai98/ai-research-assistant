# API Reference

Base URL: `http://localhost:8000/api`
Interactive docs (OpenAPI/Swagger): `http://localhost:8000/api/docs`

All error responses share the shape:

```json
{ "code": "not_found", "message": "Document 'x' was not found." }
```

Authentication is optional. When `AUTH_ENABLED=true`, obtain a token from
`POST /auth/login` and send `Authorization: Bearer <token>`. Otherwise all
requests run as the `guest` owner.

## System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status, version, active models. |
| GET | `/dashboard` | Aggregate document, storage, and chat statistics. |

## Auth

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/auth/login` | `{username, password}` | Returns a bearer token (only when auth is enabled). |

## Documents

| Method | Path | Description |
|--------|------|-------------|
| POST | `/documents` | Multipart upload (`file`). Validates, dedupes, chunks, embeds, indexes. → `201` |
| GET | `/documents` | List documents for the owner. |
| GET | `/documents/{id}` | Get one document's metadata. |
| GET | `/documents/{id}/file` | Download the original PDF. |
| DELETE | `/documents/{id}` | Delete document, vectors, and file. → `204` |

## Chat

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/chats` | `{document_ids}` | Start a session. → `201` |
| GET | `/chats` | – | List session summaries. |
| GET | `/chats/{id}` | – | Full session with messages + citations. |
| POST | `/chats/{id}/ask` | `{question, document_ids?}` | Grounded answer with citations. |
| POST | `/chats/{id}/ask/stream` | `{question, document_ids?}` | Stream the answer as text. |
| DELETE | `/chats/{id}` | – | Delete a session. → `204` |

### Answer shape

```json
{
  "session_id": "…",
  "answer": {
    "text": "Transformers use self-attention … [1].",
    "citations": [
      {"marker": 1, "document_id": "…", "document_name": "paper.pdf",
       "page_number": 1, "chunk_id": "…", "quote": "…", "score": 0.82}
    ],
    "context_found": true,
    "retrieval_ms": 4.1, "llm_ms": 12.3,
    "prompt_tokens": null, "completion_tokens": null
  }
}
```

## Search

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/search` | `{query, document_ids?, top_k?}` | Semantic search, global or scoped. |

## AI features

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/ai/{id}/summary` | – | Structured, cited summary. |
| POST | `/ai/{id}/methodology` | – | Extract methodology. |
| POST | `/ai/{id}/limitations` | – | Extract limitations. |
| POST | `/ai/{id}/future-work` | – | Extract future work. |
| POST | `/ai/explain` | `{document_id, concept}` | Explain a concept. |
| POST | `/ai/quiz` | `{document_id, num_questions}` | Generate a quiz. |
| POST | `/ai/flashcards` | `{document_id, num_cards}` | Generate flashcards. |
| POST | `/ai/compare` | `{document_ids[2+], aspect?}` | Compare papers. |
| POST | `/ai/literature-review` | `{document_ids[1+], topic?}` | Draft a review. |
| GET | `/ai/{id}/citation?style=` | – | Format as `apa`/`ieee`/`mla`/`bibtex`. |

## Export

| Method | Path | Query | Description |
|--------|------|-------|-------------|
| GET | `/export/chats/{id}` | `fmt=markdown\|pdf` | Download a conversation with citations. |
