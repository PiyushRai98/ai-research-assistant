# `config/`

Deployment-time configuration that lives outside application code.

| File | Purpose |
|------|---------|
| `gunicorn.conf.py` | Production process-manager settings for serving the FastAPI app with Uvicorn workers. |

Application behaviour (chunk size, models, retrieval, paths, limits) is
configured via **environment variables** validated by
`app/shared/config.py`. See [`.env.example`](../.env.example) for every
supported variable and its default.
