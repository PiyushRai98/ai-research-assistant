# Deployment

## Prerequisites

- Python 3.12+ (for local runs) or Docker 24+ with Compose v2.
- Optional: an OpenAI-compatible LLM endpoint (Ollama, vLLM, TGI, watsonx proxy)
  for real generation. Without one, the app uses the offline echo LLM.

## Docker Compose (recommended)

```bash
cp .env.example .env          # set APP_ENV=production, LLM_* etc.
docker compose up --build     # add: --profile cache   to include Redis
```

- Frontend: `http://localhost:8501`
- API: `http://localhost:8000` (docs at `/api/docs`)

Persistent named volumes retain uploads, the FAISS index, the SQLite database,
and logs across restarts. The frontend waits for the backend's health check
before starting.

## Production serving (without Compose)

Serve the API with Gunicorn managing Uvicorn workers:

```bash
pip install ".[ml]" gunicorn
gunicorn app.backend.main:app -c config/gunicorn.conf.py
```

Run the frontend behind the same origin or a reverse proxy, setting
`API_BASE_URL` to the backend's public URL.

### Reverse proxy

Terminate TLS at a proxy (nginx, Caddy, ALB) and forward:
- `/api/*` → backend `:8000`
- everything else → frontend `:8501`

Set `APP_ENV=production` so CORS is restricted to `API_BASE_URL`.

## Configuration checklist for production

- [ ] `APP_ENV=production`, `APP_DEBUG=false`
- [ ] Strong `AUTH_SECRET_KEY` if `AUTH_ENABLED=true`
- [ ] `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_BASE`, `LLM_API_KEY` configured
- [ ] `EMBEDDING_MODEL` chosen (`BAAI/bge-small-en-v1.5` recommended)
- [ ] `UPLOAD_MAX_FILE_MB` sized for your documents
- [ ] Storage volumes backed up (`storage/database`, `storage/vectorstore`, `storage/uploads`)

## Health & smoke tests

- Backend container health: `GET /api/health`
- Frontend container health: `GET /_stcore/health`
- Post-deploy check: `python scripts/smoke_test.py --base-url https://your-host`

## Scaling notes

- Embedding and generation are the heavy paths. Scale backend workers via
  `WEB_CONCURRENCY`; recycle workers (`max_requests`) to bound memory.
- FAISS runs in-process. For very large corpora, migrate the `VectorStore` port
  to a networked index (e.g. a managed vector DB) — only the adapter changes.
- Enable Redis (`REDIS_ENABLED=true`, `--profile cache`) to cache embeddings and
  responses.
