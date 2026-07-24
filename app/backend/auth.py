"""Optional, dependency-free authentication with guest fallback.

Per the brief, login is optional and a guest mode is always available. When
``AUTH_ENABLED`` is false every request resolves to the ``guest`` owner. When
enabled, clients present a bearer token minted by ``/api/auth/login``.

Tokens are stateless and self-verifying: ``payload.signature`` where the
signature is an HMAC-SHA256 over the payload using the configured secret. This
avoids a JWT dependency while remaining tamper-evident and expiring.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from app.shared.config import Settings
from app.shared.exceptions import AuthenticationError

GUEST = "guest"


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload_b64: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256)
    return _b64encode(digest.digest())


def issue_token(username: str, *, settings: Settings) -> str:
    """Mint a signed, expiring token for ``username``."""
    expires_at = int(time.time()) + settings.auth_token_ttl_minutes * 60
    payload = {"sub": username, "exp": expires_at}
    payload_b64 = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _sign(payload_b64, settings.auth_secret_key)
    return f"{payload_b64}.{signature}"


def verify_token(token: str, *, settings: Settings) -> str:
    """Validate a token and return its subject (username). Raises on failure."""
    try:
        payload_b64, signature = token.split(".", 1)
    except ValueError as exc:
        raise AuthenticationError("Malformed authentication token.") from exc

    expected = _sign(payload_b64, settings.auth_secret_key)
    if not hmac.compare_digest(expected, signature):
        raise AuthenticationError("Invalid authentication token signature.")

    try:
        payload = json.loads(_b64decode(payload_b64))
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuthenticationError("Corrupt authentication token payload.") from exc

    if payload.get("exp", 0) < int(time.time()):
        raise AuthenticationError("Authentication token has expired.")
    return str(payload.get("sub", GUEST))
