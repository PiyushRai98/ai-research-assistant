"""Unit tests for security helpers and configuration validation."""

from __future__ import annotations

import pytest
from app.shared.config import Settings
from app.shared.exceptions import ValidationError
from app.shared.security import sanitize_filename, sha256_hex, validate_pdf_upload

_PDF = b"%PDF-1.7\n%stub"


def test_sanitize_filename_strips_paths_and_traversal() -> None:
    assert sanitize_filename("../../etc/passwd") == "passwd.pdf"
    assert sanitize_filename("C:\\Windows\\report.pdf") == "report.pdf"
    assert sanitize_filename("") == "document.pdf"
    assert sanitize_filename("weird name!@#.pdf").endswith(".pdf")


def test_sha256_is_stable() -> None:
    assert sha256_hex(b"abc") == sha256_hex(b"abc")
    assert sha256_hex(b"abc") != sha256_hex(b"abd")


def test_validate_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        validate_pdf_upload(b"", max_bytes=1000)


def test_validate_rejects_oversize() -> None:
    with pytest.raises(ValidationError, match="too large"):
        validate_pdf_upload(_PDF + b"x" * 100, max_bytes=10)


def test_validate_rejects_non_pdf_magic() -> None:
    with pytest.raises(ValidationError, match="valid PDF"):
        validate_pdf_upload(b"PK\x03\x04 not a pdf", max_bytes=1000)


def test_validate_accepts_valid_pdf() -> None:
    validate_pdf_upload(_PDF, max_bytes=1000, declared_content_type="application/pdf")


def test_chunk_overlap_must_be_less_than_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(chunk_size=100, chunk_overlap=200)


def test_upload_max_bytes_property() -> None:
    settings = Settings(upload_max_file_mb=2)
    assert settings.upload_max_bytes == 2 * 1024 * 1024
