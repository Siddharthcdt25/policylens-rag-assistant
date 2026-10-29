"""Inspection script for Stage 1.

Runs the ingestion pipeline (PDF parsing -> conditional OCR -> section-aware chunking)
and prints a detailed breakdown of each chunk with page provenance and section headers.

Usage:
    python scripts/inspect_chunks.py [path_to_pdf]
"""

import sys
from pathlib import Path

# Add project root to sys.path so modules import reliably
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_parsed_pages
from scripts.create_sample_policy import generate_sample_policy


def main() -> None:
    # Use provided file argument or automatically generate a sample policy
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = ROOT_DIR / "data" / "sample_policy.pdf"
        if not pdf_path.exists():
            print("No PDF specified. Generating synthetic 3-page policy (with OCR page)...")
            generate_sample_policy(pdf_path)

    if not pdf_path.exists():
        print(f"Error: File not found at {pdf_path}")
        sys.exit(1)

    print("=" * 80)
    print(f"STAGE 1 INSPECTION: Ingesting '{pdf_path.name}'")
    print("=" * 80)

    # 1. Parse pages with conditional OCR fallback
    pages = parse_pdf(pdf_path, filename=pdf_path.name)
    print(f"\n[1] PARSED {len(pages)} PAGES:")
    for p in pages:
        status = "OCR Fallback Triggered" if p.used_ocr else "Native Text Extracted"
        print(f"  • Page {p.page_number:02d}: {p.char_count:>5} chars | Mode: {status}")

    # 2. Section-aware chunking
    chunks = chunk_parsed_pages(pages)
    print(f"\n[2] GENERATED {len(chunks)} CHUNKS (Target ~400 tokens, ~15% overlap):")
    print("-" * 80)

    total_tokens = 0
    for idx, c in enumerate(chunks, start=1):
        total_tokens += c.token_count
        print(f"\n--- Chunk #{idx} [{c.chunk_id}] ---")
        print(f"  Page Reference : Page {c.page_number} ({c.source_filename})")
        print(f"  Section Title  : {c.section_title}")
        print(f"  Token Count    : ~{c.token_count} tokens")
        print("  Snippet Content:")
        snippet = c.text[:220].replace("\n", " ")
        print(f"    \"{snippet}...\"")

    avg_tokens = total_tokens / len(chunks) if chunks else 0
    print("\n" + "=" * 80)
    print("INGESTION METRICS SUMMARY:")
    print(f"  Total Chunks    : {len(chunks)}")
    print(f"  Total Tokens    : {total_tokens}")
    print(f"  Average Tokens  : {avg_tokens:.1f} tokens/chunk")
    print("=" * 80)


if __name__ == "__main__":
    main()
