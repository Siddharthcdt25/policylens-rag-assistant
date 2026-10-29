"""Section-aware document chunker for insurance policies.

Why section-aware chunking?
Standard fixed-character chunking frequently splits right through critical insurance clauses
(e.g., cutting an exclusion in half or separating a condition from its deductible).
Section-aware chunking detects headings (e.g., 'SECTION 2 - EXCLUSIONS', 'Clause 3.1')
and anchors chunks to them, preserving legal context and hierarchy in the vector metadata.
"""

from dataclasses import dataclass
import re
from typing import List, Optional
from app.config import settings
from app.ingestion.parser import ParsedPage


@dataclass
class Chunk:
    """A self-contained chunk with provenance metadata for grounded citation."""
    chunk_id: str
    text: str
    page_number: int
    source_filename: str
    section_title: str
    token_count: int


# Regex patterns matching insurance policy section headers and clause numbering
HEADING_PATTERNS = [
    # E.g., "SECTION 1 - PROPERTY COVERAGE", "ARTICLE IV: LIABILITY"
    re.compile(r"^(SECTION|ARTICLE|CLAUSE|PART|SCHEDULE|ENDORSEMENT)\s+[0-9A-ZIVXLC\.\-]+(?:\s*[-:]\s*[A-Z\s]+)?", re.IGNORECASE),
    # E.g., "1.1 Definitions", "4.2.1 Water Damage Exclusion"
    re.compile(r"^\d+\.\d+(?:\.\d+)?\s+[A-Z][A-Za-z0-9\s,\-]{2,60}"),
    # E.g., ALL CAPS headers on their own line: "EXCLUSIONS", "GENERAL CONDITIONS"
    re.compile(r"^[A-Z0-9\s\-\/\(\)]{4,50}$"),
]


def estimate_token_count(text: str) -> int:
    """Fast, dependency-free token estimation.

    In standard English technical/legal text:
    ~1 word ≈ 1.3 tokens (or ~4 characters ≈ 1 token).
    Estimating from word count prevents needing a heavy BPE tokenizer library.
    """
    words = text.split()
    return int(len(words) * 1.3)


def detect_heading(line: str) -> Optional[str]:
    """Check if a line matches typical policy heading and clause structures."""
    cleaned = line.strip()
    if not cleaned or len(cleaned) > 80:
        return None

    for pattern in HEADING_PATTERNS:
        if pattern.match(cleaned):
            return cleaned
    return None


def split_text_with_overlap(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split a long text block into overlapping sub-chunks by word boundaries.

    Args:
        text: Input string to split.
        target_tokens: Desired maximum tokens per chunk (~400).
        overlap_tokens: Overlap tokens (~60, roughly 15%).

    Returns:
        List of text substrings with sliding window overlap.
    """
    words = text.split()
    if not words:
        return []

    # Convert token targets back to word counts (words = tokens / 1.3)
    target_words = max(10, int(target_tokens / 1.3))
    overlap_words = max(2, int(overlap_tokens / 1.3))
    step = max(1, target_words - overlap_words)

    sub_chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + target_words]
        sub_chunks.append(" ".join(chunk_words))
        if start + target_words >= len(words):
            break

    return sub_chunks


def chunk_parsed_pages(
    pages: list[ParsedPage],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Convert parsed pages into section-anchored chunks with page-level citations.

    Algorithm:
    1. Scan lines sequentially. When a heading is recognized, update the active section context.
    2. Group lines into coherent logical blocks belonging to that section.
    3. If a block fits within `chunk_size`, keep it intact (avoids artificial clause fracturing).
    4. If a block exceeds `chunk_size`, apply sliding-window overlap (~15%).
    5. Always record the source filename, page number, and section title on the chunk.

    Args:
        pages: Output of parse_pdf().
        chunk_size: Target token size per chunk (default 400).
        chunk_overlap: Token overlap between adjacent chunks (default 60).

    Returns:
        List of Chunk objects ready for embedding and indexing.
    """
    chunks: list[Chunk] = []
    chunk_counter = 0
    current_section = "General Provisions"

    for page in pages:
        lines = page.text.splitlines()
        current_block: list[str] = []

        def flush_block(block_lines: list[str], section: str) -> None:
            nonlocal chunk_counter
            block_text = " ".join(block_lines).strip()
            if not block_text:
                return

            tokens = estimate_token_count(block_text)

            if tokens <= chunk_size:
                # Small enough to remain a single complete clause
                chunk_id = f"{page.source_filename}_p{page.page_number}_c{chunk_counter}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=block_text,
                        page_number=page.page_number,
                        source_filename=page.source_filename,
                        section_title=section,
                        token_count=tokens,
                    )
                )
                chunk_counter += 1
            else:
                # Exceeds target: split with overlap
                pieces = split_text_with_overlap(
                    text=block_text,
                    target_tokens=chunk_size,
                    overlap_tokens=chunk_overlap,
                )
                for piece in pieces:
                    sub_tokens = estimate_token_count(piece)
                    chunk_id = f"{page.source_filename}_p{page.page_number}_c{chunk_counter}"
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=piece,
                            page_number=page.page_number,
                            source_filename=page.source_filename,
                            section_title=section,
                            token_count=sub_tokens,
                        )
                    )
                    chunk_counter += 1

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            detected_heading = detect_heading(line_str)
            if detected_heading:
                # Flush existing accumulated lines under previous heading
                if current_block:
                    flush_block(current_block, current_section)
                    current_block = []
                current_section = detected_heading
                current_block.append(line_str)
            else:
                current_block.append(line_str)

        # Flush any remaining text on the page
        if current_block:
            flush_block(current_block, current_section)

    return chunks
