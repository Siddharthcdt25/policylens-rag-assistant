"""Inspection script for Stage 2: Embeddings + FAISS Retrieval (No LLM).

Verifies the retrieval pipeline end-to-end:
1. Ingests and embeds the policy chunks using sentence-transformers (all-MiniLM-L6-v2).
2. Indexes vectors in FAISS IndexFlatIP (cosine similarity).
3. Executes a test query (or pre-set test suite) and displays top-k chunks with scores,
   section titles, page citations, and threshold pass/fail status.

Usage:
    # Run built-in test suite:
    python scripts/inspect_retrieval.py

    # Run with a custom question:
    python scripts/inspect_retrieval.py "What is the deductible for solar panel wind damage?"
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_parsed_pages
from app.retrieval.embedder import get_embedder
from app.retrieval.store import VectorStore
from scripts.create_sample_policy import generate_sample_policy


def ingest_sample_if_needed(store: VectorStore) -> None:
    """Ingest sample policy if vector store is unpopulated."""
    store_dir = ROOT_DIR / "data" / "vector_store"
    if store.load(store_dir) and len(store) > 0:
        print(f"Loaded existing vector index ({len(store)} chunks) from {store_dir}")
        return

    sample_pdf = ROOT_DIR / "data" / "sample_policy.pdf"
    if not sample_pdf.exists():
        print("Generating synthetic sample policy PDF...")
        generate_sample_policy(sample_pdf)

    print(f"Ingesting '{sample_pdf.name}' into FAISS vector store...")
    pages = parse_pdf(sample_pdf, filename=sample_pdf.name)
    chunks = chunk_parsed_pages(pages)

    embedder = get_embedder()
    print(f"Vectorizing {len(chunks)} chunks with '{embedder.model_name}'...")
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)

    store.add_chunks(chunks, embeddings)
    store.save(store_dir)
    print(f"Indexed and saved {len(store)} chunks to {store_dir}\n")


def test_query(store: VectorStore, query: str, top_k: int = settings.TOP_K) -> None:
    """Run retrieval for a query and pretty-print ranked results."""
    embedder = get_embedder()
    print("=" * 80)
    print(f"QUERY: \"{query}\"")
    print(f"Configured Threshold: {settings.SIMILARITY_THRESHOLD:.2f} | Requested Top-K: {top_k}")
    print("=" * 80)

    # 1. Embed query
    query_vector = embedder.embed_query(query)

    # 2. Search FAISS index
    results = store.search(query_vector, top_k=top_k)

    if not results:
        print("No results returned (vector store empty).")
        return

    top_score = results[0].score
    passes_threshold = top_score >= settings.SIMILARITY_THRESHOLD

    print(
        f"Retrieval Status: {'PASSED THRESHOLD (Will Call LLM)' if passes_threshold else 'FAILED THRESHOLD (Direct Refusal in Code)'}"
    )
    print(f"Top Similarity Score: {top_score:.4f}\n")

    for rank, res in enumerate(results, start=1):
        c = res.chunk
        status_tag = "[ABOVE THRESHOLD]" if res.score >= settings.SIMILARITY_THRESHOLD else "[BELOW THRESHOLD]"
        print(f"--- Rank #{rank} | Cosine Similarity: {res.score:.4f} {status_tag} ---")
        print(f"  Citation     : [{c.source_filename} p.{c.page_number}]")
        print(f"  Section      : {c.section_title}")
        print(f"  Chunk Tokens : ~{c.token_count} tokens")
        snippet = c.text[:220].replace("\n", " ")
        print(f"  Snippet      : \"{snippet}...\"\n")


def main() -> None:
    store = VectorStore(dimension=384)
    ingest_sample_if_needed(store)

    if len(sys.argv) > 1:
        # User supplied custom query
        custom_query = " ".join(sys.argv[1:])
        test_query(store, custom_query)
    else:
        # Run test suite covering different pages, OCR, and out-of-domain queries
        test_cases = [
            "What is the deductible for Coverage A dwelling losses?",
            "What is the maximum limit for solar panel equipment?",
            "Is sewer backup covered under this insurance policy?",
            "Does this policy cover damage from extraterrestrial alien invasions?",
        ]

        print("No query supplied via CLI. Running automated retrieval test suite...\n")
        for q in test_cases:
            test_query(store, q, top_k=3)


if __name__ == "__main__":
    main()
