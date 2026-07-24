"""Production Gunicorn configuration for the FastAPI backend.

Run with:
    gunicorn app.backend.main:app -c config/gunicorn.conf.py

Uses Uvicorn workers (ASGI) behind Gunicorn's process manager for resilient,
multi-worker production serving. Values can be overridden with the matching
environment variables.
"""

from __future__ import annotations

import multiprocessing
import os

# Bind address/port (mirrors APP_HOST / APP_PORT).
bind = f"{os.environ.get('APP_HOST', '0.0.0.0')}:{os.environ.get('APP_PORT', '8000')}"

# One worker per core is a safe default for CPU-bound embedding/inference work.
workers = int(os.environ.get("WEB_CONCURRENCY", multiprocessing.cpu_count()))
worker_class = "uvicorn.workers.UvicornWorker"

# Long timeout accommodates first-request model loading and large PDFs.
timeout = int(os.environ.get("WORKER_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5

# Recycle workers periodically to bound memory growth from model caches.
max_requests = 1000
max_requests_jitter = 100

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("APP_LOG_LEVEL", "info").lower()
