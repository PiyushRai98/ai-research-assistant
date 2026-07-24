"""Security helpers for upload validation and safe file handling.

Centralises the defensive checks required by the brief: MIME/type validation
via magic bytes, size limits, filename sanitisation to prevent path traversal,
and content hashing for duplicate detection.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import PurePosixPath

from app.shared.exceptions import ValidationError

_PDF_MAGIC = b"%PDF-"
# Allow word chars, dot, dash, space; collapse everything else.
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")
_MULTI_DOT = re.compile(r"\.{2,}")


def sha256_hex(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data`` (used for dedup + logging)."""
    return hashlib.sha256(data).hexdigest()


def sanitize_filename(filename: str, *, default: str = "document.pdf") -> str:
    """Return a safe, flat filename with no path components or traversal.

    Strips directories, normalises unicode, removes unsafe characters, and
    guards against empty or dot-only names.
    """
    if not filename:
        return default
    # Drop any directory component regardless of separator style.
    base = PurePosixPath(filename.replace("\\", "/")).name
    normalised = unicodedata.normalize("NFKD", base)
    cleaned = _UNSAFE_CHARS.sub("_", normalised).strip(" .")
    cleaned = _MULTI_DOT.sub(".", cleaned)
    if not cleaned or cleaned in {".", ".."}:
        return default
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    return cleaned[:255]


def validate_pdf_upload(
    data: bytes,
    *,
    max_bytes: int,
    declared_content_type: str | None = None,
) -> None:
    """Validate an uploaded PDF's size and true type. Raises ``ValidationError``.

    Trusts magic bytes over the client-declared content type, since the latter
    is attacker-controlled.
    """
    size = len(data)
    if size == 0:
        raise ValidationError("The uploaded file is empty.")
    if size > max_bytes:
        raise ValidationError(
            f"File is too large ({size} bytes). " f"Maximum allowed is {max_bytes} bytes.",
            details={"size": size, "max_bytes": max_bytes},
        )
    if not data.startswith(_PDF_MAGIC):
        raise ValidationError("File content is not a valid PDF (magic header check failed).")
    if declared_content_type and "pdf" not in declared_content_type.lower():
        raise ValidationError(f"Unexpected content type '{declared_content_type}'; expected PDF.")
