# AI Research Assistant

Upload research papers and chat with them using Retrieval-Augmented Generation
(RAG). Every answer is grounded in your documents and backed by verifiable,
page-level citations. Supports multiple documents, semantic search, summaries,
comparisons, quizzes, flashcards, literature reviews, and exports.

Built with a Clean Architecture backend (FastAPI) and a Streamlit frontend
whose entire visual system is generated from [`DESIGN.md`](DESIGN.md).

---

## Features

- **Multi-document upload** — drag-and-drop PDFs, duplicate detection, validation, progress.
- **Robust PDF processing** — text + metadata extraction, page tracking, malformed-file handling.
- **Citation-grounded RAG** — retrieval → context → LLM → answer → citations. Never fabricates sources; states clearly when the answer isn't in the documents.
- **Configurable retrieval** — similarity or MMR, top-k, score threshold, per-document filtering.
- **Selectable models** — embeddings (`BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2`), LLM (IBM Granite preferred, Llama 3 fallback, any OpenAI-compatible endpoint).
- **AI tools** — summary, methodology / limitations / future-work extraction, concept explanation, quiz, flashcards, paper comparison, literature review, and APA/IEEE/MLA/BibTeX citations.
- **Search** — semantic search, global or scoped to selected documents.
- **Dashboard** — documents, storage, chunks, embedding status, chats, processing time.
- **Document viewer** — preview PDFs and jump to cited pages.
- **Export** — conversations and research artefacts as Markdown or PDF.
- **Security** — upload sanitisation, MIME validation, prompt-injection defence, no secrets in code.
- **Offline-friendly** — deterministic hashing embeddings + echo LLM fallback keep the app (and tests) working with zero external services.

## Architecture

Clean Architecture with dependencies pointing inward:

```
presentation (backend/ FastAPI, frontend/ Streamlit)
   -> application/   use cases: documents, retrieval, RAG, chat, AI features, export
   -> domain/        entities, value objects, and ports (interfaces)
   -> infrastructure/ adapters: PyMuPDF, chunking, embeddings, FAISS, SQLite, LLM
shared/  cross-cutting: config, logging, security, errors
```

Business logic never touches Streamlit or a specific library — adapters
implement domain ports and are wired once in `app/backend/container.py`. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Tech stack

Python 3.12+ · FastAPI · Streamlit · FAISS · PyMuPDF · Sentence-Transformers ·
IBM Granite / Llama 3 (OpenAI-compatible) · Pydantic · SQLite · Docker ·
GitHub Actions · Pytest · Ruff · Black.

## Quick start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install ".[ml,ui,dev]"
cp .env.example .env        # then edit as needed
```

> Tip: without the `ml` extra (or when models can't be downloaded), the app
> automatically falls back to deterministic hashing embeddings and an offline
> echo LLM, so it still runs end-to-end.

### 2. Run the backend

```bash
uvicorn app.backend.main:app --reload
# API docs: http://localhost:8000/api/docs
```

### 3. Run the frontend

```bash
API_BASE_URL=http://localhost:8000 streamlit run app/frontend/app.py
# UI: http://localhost:8501
```

Or use the helper scripts: `./scripts/dev.sh` (Unix) / `./scripts/dev.ps1` (Windows).

### With Docker

```bash
docker compose up --build
# Frontend: http://localhost:8501   API: http://localhost:8000
```

### Deploying the frontend to Streamlit Community Cloud

The backend (FastAPI + ML stack) is not hosted by Community Cloud — deploy it
separately (Docker, a VM, Render, Fly.io, etc.), then point the UI at it:

1. Deploy `app/backend` wherever you like and note its public URL.
2. Deploy this repo to Community Cloud with entrypoint `app/frontend/app.py`.
3. In the app's "Advanced settings", set the secret/env var
   `API_BASE_URL=https://your-backend-host`.

Community Cloud uses the root-level [`requirements.txt`](requirements.txt) —
a minimal file containing only `streamlit` and `httpx`, since the frontend is
a thin API client. It intentionally does **not** use `pyproject.toml` (which
Community Cloud would otherwise try to install via Poetry and fail, since this
project uses a setuptools/PEP 621 layout, not a Poetry package layout).

## Configuration

Everything is configurable via environment variables (validated in
`app/shared/config.py`): chunk size/overlap, embedding model, LLM provider /
model / temperature / max tokens / timeout, top-k, retrieval strategy, storage
paths, and upload limits. See [`.env.example`](.env.example).

### Using a real LLM

Point the app at any OpenAI-compatible endpoint (Ollama, vLLM, TGI, watsonx proxy):

```bash
LLM_PROVIDER=granite
LLM_MODEL=ibm-granite/granite-3.0-8b-instruct
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=your-key-if-required
```

## Testing

```bash
pytest                       # runs offline; no model download or network
ruff check app tests         # lint
black --check app tests      # format check
```

The suite gates coverage at 80%.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API reference](docs/API.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Security](docs/SECURITY.md)
- [Changelog](docs/CHANGELOG.md)

## License

MIT.
