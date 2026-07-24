"""Structured logging configuration built on Loguru.

The brief requires structured logs covering uploads, retrieval, errors, LLM
latency, embedding time, and token usage. This module centralises logger
setup so every layer emits consistent, JSON-serialisable records to both the
console and a rotating file sink. Helper functions provide a uniform way to
record the required operational events.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

_CONFIGURED = False


def configure_logging(
    *,
    log_dir: Path,
    level: str = "INFO",
    json_logs: bool = False,
) -> None:
    """Initialise the global logger. Safe to call multiple times (idempotent).

    Args:
        log_dir: Directory that receives the rotating log file.
        level: Minimum log level for console/file sinks.
        json_logs: When True, file records are serialised as JSON lines.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()

    # Human-readable console sink.
    logger.add(
        sys.stderr,
        level=level,
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    # Persistent, rotating file sink for auditing and post-mortems.
    logger.add(
        log_dir / "app.log",
        level=level,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,  # process-safe, non-blocking
        serialize=json_logs,
        backtrace=False,
        diagnose=False,
    )

    _CONFIGURED = True


def get_logger(name: str) -> Logger:
    """Return a logger bound to a component name for filterable records."""
    return logger.bind(component=name)


# --------------------------------------------------------------------------
# Structured event helpers. Each records one operational metric the brief
# explicitly asks to track, using a stable ``event`` key for downstream
# aggregation (e.g. shipping to a log pipeline).
# --------------------------------------------------------------------------
def log_upload(*, document_id: str, filename: str, size_bytes: int) -> None:
    """Record a successful document upload."""
    logger.bind(event="upload", document_id=document_id, size_bytes=size_bytes).info(
        "Document uploaded: {filename} ({size_bytes} bytes)",
        filename=filename,
        size_bytes=size_bytes,
    )


def log_embedding(*, document_id: str, chunk_count: int, duration_ms: float) -> None:
    """Record embedding throughput for a document."""
    logger.bind(
        event="embedding",
        document_id=document_id,
        chunk_count=chunk_count,
        duration_ms=round(duration_ms, 2),
    ).info(
        "Embedded {chunk_count} chunks in {duration_ms:.1f} ms",
        chunk_count=chunk_count,
        duration_ms=duration_ms,
    )


def log_retrieval(*, query_hash: str, k: int, hits: int, duration_ms: float) -> None:
    """Record a retrieval operation."""
    logger.bind(
        event="retrieval",
        query_hash=query_hash,
        k=k,
        hits=hits,
        duration_ms=round(duration_ms, 2),
    ).info(
        "Retrieved {hits}/{k} chunks in {duration_ms:.1f} ms",
        hits=hits,
        k=k,
        duration_ms=duration_ms,
    )


def log_llm(
    *,
    provider: str,
    model: str,
    duration_ms: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> None:
    """Record LLM latency and token usage."""
    logger.bind(
        event="llm",
        provider=provider,
        model=model,
        duration_ms=round(duration_ms, 2),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    ).info(
        "LLM {provider}/{model} responded in {duration_ms:.1f} ms "
        "(prompt={prompt_tokens}, completion={completion_tokens})",
        provider=provider,
        model=model,
        duration_ms=duration_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def log_error(error: Exception, **context: Any) -> None:
    """Record an error with structured context (never re-raises)."""
    logger.bind(event="error", **context).opt(exception=error).error(
        "Error: {message}", message=str(error)
    )
