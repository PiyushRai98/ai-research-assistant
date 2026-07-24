"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.backend.dependencies import ContainerDep
from app.backend.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(container: ContainerDep) -> HealthResponse:
    """Report service health and the active model configuration."""
    settings = container.settings
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.app_env.value,
        embedding_model=settings.embedding_model,
        llm=settings.llm_provider.value,
    )
