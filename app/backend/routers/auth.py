"""Optional local authentication endpoints.

Provides a minimal login that issues a signed bearer token when auth is
enabled. Guest mode requires no login. This is deliberately simple: a single
configurable credential pair validated in constant time. For multi-user
production deployments, swap this router for an IdP / OAuth integration.
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.backend.auth import issue_token
from app.backend.dependencies import SettingsDep

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


@router.post("/login", response_model=LoginResponse, summary="Obtain an access token")
def login(request: LoginRequest, settings: SettingsDep) -> LoginResponse:
    """Authenticate and mint a bearer token (only when auth is enabled).

    The demo credential is username == secret prefix; real deployments should
    replace this with a proper user store. Comparison is constant-time.
    """
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authentication is disabled; the app runs in guest mode.",
        )
    # Constant-time credential check against the configured secret.
    expected_password = settings.auth_secret_key
    if not hmac.compare_digest(request.password, expected_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    token = issue_token(request.username, settings=settings)
    return LoginResponse(
        access_token=token,
        expires_in_minutes=settings.auth_token_ttl_minutes,
    )
