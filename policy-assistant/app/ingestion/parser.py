"""PDF parsing module with conditional OCR fallback.

Why this matters in insurance:
Insurance contracts frequently contain a mix of native digital text pages and scanned
physical riders, endorsements, or signed declarations. A naive text extractor produces
empty pages for the scans, causing silent retrieval dropouts.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Union
import fitz  # PyMuPDF

from app.config import settings
from app.ingestion.ocr import ocr_pdf_page

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """Represents the extracted contents and provenance of a single PDF page."""
    page_number: int  # 1-indexed for human-readable policy citations
    text: str
    source_filename: str
    used_ocr: bool
    char_count: int


def parse_pdf(
    source: Union[str, Path, bytes],
    filename: str,
    min_chars_for_ocr: int = settings.OCR_MIN_CHARS,
) -> list[ParsedPage]:
    """Extract text page-by-page from a PDF, falling back to OCR when needed.

    Args:
        source: Filepath (str/Path) or raw byte buffer (e.g. from FastAPI/Streamlit upload).
        filename: Original name of the document to preserve page citations.
        min_chars_for_ocr: Threshold below which a page is treated as scanned/graphic.

    Returns:
        List of ParsedPage objects containing text, page numbers, and OCR metadata.
    """
    parsed_pages: list[ParsedPage] = []

    # Support both file paths on disk and in-memory byte streams from file uploads
    if isinstance(source, bytes):
        doc = fitz.open(stream=source, filetype="pdf")
    else:
        doc = fitz.open(str(source))

    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1  # 1-indexed for citation

            # Fast native text extraction from embedded fonts
            raw_text = page.get_text("text") or ""
            clean_text = raw_text.strip()
            used_ocr = False

            # Check if text is suspiciously short (e.g. scanned image, stamped addendum)
            if len(clean_text) < min_chars_for_ocr:
                logger.info(
                    f"Page {page_num} of '{filename}' has only {len(clean_text)} characters. "
                    "Triggering OCR fallback..."
                )
                ocr_text = ocr_pdf_page(page)
                if len(ocr_text.strip()) > len(clean_text):
                    clean_text = ocr_text.strip()
                    used_ocr = True
                    logger.info(
                        f"Page {page_num} OCR recovered {len(clean_text)} characters."
                    )

            parsed_pages.append(
                ParsedPage(
                    page_number=page_num,
                    text=clean_text,
                    source_filename=filename,
                    used_ocr=used_ocr,
                    char_count=len(clean_text),
                )
            )
    finally:
        doc.close()

    return parsed_pages
