"""Ingestion module: PDF parsing, OCR fallback, and section-aware chunking."""
from app.ingestion.parser import parse_pdf, ParsedPage
from app.ingestion.chunker import chunk_parsed_pages, Chunk
from app.ingestion.ocr import ocr_pdf_page

__all__ = [
    "parse_pdf",
    "ParsedPage",
    "chunk_parsed_pages",
    "Chunk",
    "ocr_pdf_page",
]
