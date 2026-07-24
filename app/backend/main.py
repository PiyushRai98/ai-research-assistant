"""FastAPI application factory and entrypoint.

Assembles the API: builds the service container on startup, registers routers,
installs a uniform exception handler that maps domain errors to safe HTTP
responses (without leaking internals), and configures CORS for the Streamlit
frontend.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.backend.container import get_container
from app.backend.routers import (
    ai,
    auth,
    chat,
    dashboard,
    documents,
    export,
    health,
    search,
)
from app.shared.exceptions import ResearchAssistantError
from app.shared.logging import get_logger, log_error

logger = get_logger("api")

API_PREFIX = "/api"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Build the container eagerly so first-request latency is predictable."""
    container = get_container()
    logger.info(
        "API ready: env={env}, docs indexed chunks={n}",
        env=container.settings.app_env.value,
        n=container.vector_store.count(),
    )
    yield
    logger.info("API shutting down.")


def create_app() -> FastAPI:
    """Application factory (testable, no import-time side effects)."""
    container = get_container()
    settings = container.settings

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Upload research papers and chat with them using RAG + citations.",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # CORS: the Streamlit UI (default :8501) calls this API from the browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [settings.api_base_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- uniform error handling ---
    @app.exception_handler(ResearchAssistantError)
    async def _domain_error_handler(_request: Request, exc: ResearchAssistantError) -> JSONResponse:
        log_error(exc, code=exc.code)
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"code": "validation_error", "message": str(exc.errors())},
        )

    @app.exception_handler(Exception)
    async def _unexpected_handler(_request: Request, exc: Exception) -> JSONResponse:
        # Never leak internals; log the detail, return a generic message.
        log_error(exc, code="internal_error")
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "An unexpected error occurred. Please try again.",
            },
        )

    # --- routers ---
    for module in (health, auth, documents, chat, search, ai, export, dashboard):
        app.include_router(module.router, prefix=API_PREFIX)

    return app


app = create_app()


def run() -> None:
    """Console-script entrypoint: launch uvicorn from settings."""
    import uvicorn

    settings = get_container().settings
    uvicorn.run(
        "app.backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )


if __name__ == "__main__":
    run()
