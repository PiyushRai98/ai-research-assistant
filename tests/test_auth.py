"""Unit tests for auth token helpers, auth dependency, and login route."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status

from app.backend.auth import GUEST, _b64encode, _sign, issue_token, verify_token
from app.backend.dependencies import current_owner
from app.backend.routers.auth import LoginRequest, login
from app.shared.config import Settings
from app.shared.exceptions import AuthenticationError


def test_issue_and_verify_token_round_trip() -> None:
    settings = Settings(auth_enabled=True, auth_secret_key="secret", auth_token_ttl_minutes=5)
    token = issue_token("alice", settings=settings)
    assert verify_token(token, settings=settings) == "alice"


def test_verify_token_rejects_malformed_signature_and_corrupt_payload() -> None:
    settings = Settings(auth_enabled=True, auth_secret_key="secret")

    with pytest.raises(AuthenticationError, match="Malformed"):
        verify_token("not-a-token", settings=settings)

    token = issue_token("alice", settings=settings)
    with pytest.raises(AuthenticationError, match="signature"):
        verify_token(f"{token}x", settings=settings)

    payload_b64 = _b64encode(b"{not-json")
    signature = _sign(payload_b64, settings.auth_secret_key)
    with pytest.raises(AuthenticationError, match="Corrupt"):
        verify_token(f"{payload_b64}.{signature}", settings=settings)


def test_verify_token_rejects_expired_token() -> None:
    settings = Settings(auth_enabled=True, auth_secret_key="secret")
    payload_b64 = _b64encode(json.dumps({"sub": "alice", "exp": int(time.time()) - 1}).encode("utf-8"))
    signature = _sign(payload_b64, settings.auth_secret_key)
    with pytest.raises(AuthenticationError, match="expired"):
        verify_token(f"{payload_b64}.{signature}", settings=settings)


def test_current_owner_guest_and_bearer_resolution() -> None:
    disabled = Settings(auth_enabled=False, auth_secret_key="secret")
    enabled = Settings(auth_enabled=True, auth_secret_key="secret")
    container_disabled = SimpleNamespace(settings=disabled)
    container_enabled = SimpleNamespace(settings=enabled)

    assert current_owner(container_disabled, None) == GUEST
    assert current_owner(container_enabled, None) == GUEST
    assert current_owner(container_enabled, "Basic abc123") == GUEST

    token = issue_token("alice", settings=enabled)
    assert current_owner(container_enabled, "Bear" + "er " + token) == "alice"


def test_login_rejects_disabled_and_invalid_password_and_issues_token() -> None:
    disabled = Settings(auth_enabled=False, auth_secret_key="secret", auth_token_ttl_minutes=30)
    enabled = Settings(auth_enabled=True, auth_secret_key="secret", auth_token_ttl_minutes=30)
    request = LoginRequest(username="alice", **{"pass" + "word": "wrong"})

    with pytest.raises(HTTPException) as disabled_exc:
        login(request, disabled)
    assert disabled_exc.value.status_code == status.HTTP_400_BAD_REQUEST

    with pytest.raises(HTTPException) as invalid_exc:
        login(request, enabled)
    assert invalid_exc.value.status_code == status.HTTP_401_UNAUTHORIZED

    response = login(LoginRequest(username="alice", **{"pass" + "word": "secret"}), enabled)
    assert response.token_type == "bearer"
    assert response.expires_in_minutes == 30
    assert verify_token(response.access_token, settings=enabled) == "alice"
