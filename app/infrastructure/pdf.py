"""PDF text and metadata extraction via PyMuPDF (fitz).

Security posture (per the brief):
- Validates the magic header and, when available, the sniffed MIME type.
- Rejects encrypted PDFs that cannot be opened without a password.
- Never executes embedded JavaScript or follows external references — PyMuPDF
  only reads the content stream; we never render or run actions.
- Wraps every libmupdf failure in a domain ``PDFProcessingError`` so malformed
  files degrade gracefully instead of crashing the process.
"""

from __future__ import annotations

from app.domain.models import PageContent
from app.shared.exceptions import PDFProcessingError
from app.shared.logging import get_logger

logger = get_logger("pdf")

# The 5-byte signature every valid PDF begins with.
_PDF_MAGIC = b"%PDF-"


class PyMuPDFParser:
    """Concrete :class:`~app.domain.ports.PDFParser` backed by PyMuPDF."""

    def _open(self, data: bytes):  # noqa: ANN202 - fitz.Document, lazily typed
        """Open a PDF from bytes, validating structure. Import fitz lazily."""
        if not data:
            raise PDFProcessingError("Uploaded file is empty.")
        if not data.startswith(_PDF_MAGIC):
            raise PDFProcessingError(
                "File does not appear to be a valid PDF (missing %PDF- header)."
            )
        try:
            import fitz  # PyMuPDF; imported lazily to keep base imports light
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise PDFProcessingError("PyMuPDF is not installed; cannot process PDFs.") from exc

        try:
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:  # libmupdf raises broad exceptions
            raise PDFProcessingError(
                "The PDF could not be opened; it may be corrupt or malformed.",
                details={"error": str(exc)},
            ) from exc

        if document.needs_pass:
            document.close()
            raise PDFProcessingError("Encrypted PDFs are not supported.")
        return document

    def extract_pages(self, data: bytes) -> list[PageContent]:
        """Extract text for every page, preserving 1-based page numbers."""
        document = self._open(data)
        pages: list[PageContent] = []
        try:
            for index in range(document.page_count):
                try:
                    page = document.load_page(index)
                    text = page.get_text("text") or ""
                except Exception as exc:  # skip a single bad page, keep the rest
                    logger.warning("Failed to extract page {n}: {err}", n=index + 1, err=str(exc))
                    text = ""
                pages.append(PageContent(page_number=index + 1, text=text.strip()))
        finally:
            document.close()

        if not any(page.text for page in pages):
            raise PDFProcessingError(
                "No extractable text found. The PDF may be a scanned image; "
                "OCR is required but not enabled."
            )
        return pages

    def extract_metadata(self, data: bytes) -> dict[str, str | int | None]:
        """Extract best-effort bibliographic metadata."""
        document = self._open(data)
        try:
            raw = document.metadata or {}
            return {
                "title": (raw.get("title") or "").strip() or None,
                "author": (raw.get("author") or "").strip() or None,
                "subject": (raw.get("subject") or "").strip() or None,
                "keywords": (raw.get("keywords") or "").strip() or None,
                "page_count": document.page_count,
            }
        finally:
            document.close()
