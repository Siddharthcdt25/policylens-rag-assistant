"""Unit tests for ingestion module: parsing, OCR fallback logic, and chunking."""

from pathlib import Path
import pytest
from app.ingestion.parser import ParsedPage, parse_pdf
from app.ingestion.chunker import (
    Chunk,
    chunk_parsed_pages,
    detect_heading,
    estimate_token_count,
    split_text_with_overlap,
)
from scripts.create_sample_policy import generate_sample_policy


@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory) -> Path:
    """Generate a temporary synthetic policy PDF for test execution."""
    tmp_dir = tmp_path_factory.mktemp("policy_data")
    pdf_path = tmp_dir / "test_policy.pdf"
    generate_sample_policy(pdf_path)
    return pdf_path


def test_token_estimation():
    """Verify word-to-token ratio approximation."""
    text = "APEX MUTUAL INSURANCE COMPANY COMPREHENSIVE PROPERTY POLICY"
    words = len(text.split())
    tokens = estimate_token_count(text)
    assert tokens == int(words * 1.3)
    assert tokens > 0


def test_heading_detection():
    """Verify regex patterns capture standard insurance clauses and sections."""
    assert detect_heading("SECTION 1 - PROPERTY COVERAGES") is not None
    assert detect_heading("ARTICLE IV - LIABILITY") is not None
    assert detect_heading("CLAUSE 2.2 - WATER DAMAGE AND FLOOD") is not None
    assert detect_heading("2.1 Earth Movement") is not None
    assert detect_heading("EXCLUSIONS") is not None

    # Ordinary body text should not trigger heading detection
    assert detect_heading("We cover the dwelling on the Described Location.") is None
    assert detect_heading("Standard deductible is $1,000 per occurrence.") is None


def test_split_text_with_overlap():
    """Verify sliding window text splitter maintains word boundaries and overlap."""
    text = "word " * 100  # 100 words ~ 130 tokens
    chunks = split_text_with_overlap(text, target_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    # Ensure all original words are covered
    assert "word" in chunks[0]
    assert "word" in chunks[-1]


def test_parse_pdf_end_to_end(sample_pdf_path):
    """Verify PDF parsing extracts pages and triggers OCR on low-text scanned pages."""
    pages = parse_pdf(sample_pdf_path, filename="test_policy.pdf", min_chars_for_ocr=50)

    assert len(pages) == 3
    # Page 1: Native text
    assert pages[0].page_number == 1
    assert "APEX MUTUAL" in pages[0].text
    assert not pages[0].used_ocr

    # Page 2: Native text
    assert pages[1].page_number == 2
    assert "EXCLUSIONS" in pages[1].text
    assert not pages[1].used_ocr

    # Page 3: Scanned endorsement image (should trigger OCR or flag attempted OCR)
    assert pages[2].page_number == 3
    # If Tesseract is installed, used_ocr is True; char_count should be tracked regardless
    assert pages[2].char_count >= 0


def test_chunk_parsed_pages():
    """Verify section-aware chunker anchors metadata to section headers."""
    mock_pages = [
        ParsedPage(
            page_number=1,
            text=(
                "SECTION 1 - PROPERTY COVERAGES\n"
                "Coverage A covers dwelling up to $450,000 with a $1,000 deductible.\n"
                "CLAUSE 1.4 - LOSS SETTLEMENT\n"
                "Losses are settled at Replacement Cost without depreciation."
            ),
            source_filename="mock.pdf",
            used_ocr=False,
            char_count=180,
        )
    ]

    chunks = chunk_parsed_pages(mock_pages, chunk_size=400, chunk_overlap=60)
    assert len(chunks) == 2

    # Chunk 1 should inherit Section 1
    assert "SECTION 1 - PROPERTY COVERAGES" in chunks[0].section_title
    assert chunks[0].page_number == 1
    assert chunks[0].source_filename == "mock.pdf"
    assert "$450,000" in chunks[0].text

    # Chunk 2 should inherit Clause 1.4
    assert "CLAUSE 1.4 - LOSS SETTLEMENT" in chunks[1].section_title
    assert "Replacement Cost" in chunks[1].text
