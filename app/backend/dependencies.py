"""FastAPI dependency providers.

Thin adapters that expose the wired container services and resolve the current
owner (guest or authenticated). Using ``Annotated`` aliases keeps route
signatures clean and self-documenting.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header

from app.application.ai_features import AIFeatureService
from app.application.chat import ChatService
from app.application.documents import DocumentService
from app.application.retrieval import RetrievalService
from app.backend.auth import GUEST, verify_token
from app.backend.container import Container, get_container
from app.shared.config import Settings


def container_dep() -> Container:
    """Provide the process-wide service container."""
    return get_container()


ContainerDep = Annotated[Container, Depends(container_dep)]


def settings_dep(container: ContainerDep) -> Settings:
    return container.settings


SettingsDep = Annotated[Settings, Depends(settings_dep)]


def document_service_dep(container: ContainerDep) -> DocumentService:
    return container.document_service


def retrieval_service_dep(container: ContainerDep) -> RetrievalService:
    return container.retrieval_service


def chat_service_dep(container: ContainerDep) -> ChatService:
    return container.chat_service


def ai_feature_service_dep(container: ContainerDep) -> AIFeatureService:
    return container.ai_feature_service


DocumentServiceDep = Annotated[DocumentService, Depends(document_service_dep)]
RetrievalServiceDep = Annotated[RetrievalService, Depends(retrieval_service_dep)]
ChatServiceDep = Annotated[ChatService, Depends(chat_service_dep)]
AIFeatureServiceDep = Annotated[AIFeatureService, Depends(ai_feature_service_dep)]


def current_owner(
    container: ContainerDep,
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """Resolve the request owner.

    Returns ``guest`` when auth is disabled or no credentials are supplied.
    When auth is enabled and a bearer token is present, the token is verified
    and its subject returned.
    """
    settings = container.settings
    if not settings.auth_enabled:
        return GUEST
    if not authorization or not authorization.lower().startswith("bearer "):
        return GUEST
    token = authorization.split(" ", 1)[1].strip()
    return verify_token(token, settings=settings)


OwnerDep = Annotated[str, Depends(current_owner)]
