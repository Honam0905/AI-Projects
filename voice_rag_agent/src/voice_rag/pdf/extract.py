"""PDF text extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass

import fitz


class PdfExtractionError(ValueError):
    """Raised when PDF extraction fails."""


@dataclass(frozen=True)
class PageText:
    """Extracted text for a single page."""

    page: int
    text: str


def extract_page_texts(pdf_bytes: bytes) -> list[PageText]:
    """Extract page-aware text from a PDF payload."""

    if not pdf_bytes:
        raise PdfExtractionError("Uploaded PDF is empty.")

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except RuntimeError as error:
        raise PdfExtractionError("Could not read PDF file.") from error

    pages: list[PageText] = []
    try:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            pages.append(PageText(page=page_index, text=text))
    finally:
        document.close()

    has_text = any(page.text for page in pages)
    if not has_text:
        raise PdfExtractionError("PDF has no extractable text.")

    return pages
