"""Inspection script for Stage 3: End-to-End Grounded Question Answering & Refusal.

Directly executes the /ask pipeline:
1. Loads the FAISS vector store.
2. Runs vector search and tests the code-level similarity threshold.
3. Formats grounded prompt with source citations [filename p.N].
4. Invokes the configured LLM backend (Gemini or Ollama).
5. Displays the grounded answer, citations, and refusal status.

Usage:
    # Run interactive query:
    python scripts/test_ask.py "What is the deductible for Coverage A dwelling losses?"

    # Run refusal test:
    python scripts/test_ask.py "How do I claim insurance for an alien spacecraft crash?"
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.retrieval.embedder import get_embedder
from app.retrieval.store import VectorStore
from app.llm import get_llm_client, format_grounded_prompt, REFUSAL_MESSAGE
from app.main import extract_citations_from_text
from scripts.create_sample_policy import generate_sample_policy
from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_parsed_pages


def ensure_index_ready(store: VectorStore) -> None:
    store_dir = ROOT_DIR / "data" / "vector_store"
    if store.load(store_dir) and len(store) > 0:
        return

    sample_pdf = ROOT_DIR / "data" / "sample_policy.pdf"
    if not sample_pdf.exists():
        generate_sample_policy(sample_pdf)

    pages = parse_pdf(sample_pdf, filename=sample_pdf.name)
    chunks = chunk_parsed_pages(pages)
    embedder = get_embedder()
    embeddings = embedder.embed_texts([c.text for c in chunks])
    store.add_chunks(chunks, embeddings)
    store.save(store_dir)


def run_ask_test(question: str, threshold: float = settings.SIMILARITY_THRESHOLD) -> None:
    store = VectorStore(dimension=384)
    ensure_index_ready(store)

    embedder = get_embedder()
    llm = get_llm_client()

    print("=" * 80)
    print(f"QUESTION: \"{question}\"")
    print(f"Active LLM Backend: {llm.provider_name.upper()} (Model: {llm.model_name})")
    print(f"Similarity Cutoff Threshold: {threshold:.2f}")
    print("=" * 80)

    # 1. Vector Search
    query_vec = embedder.embed_query(question)
    results = store.search(query_vec, top_k=settings.TOP_K)

    if not results:
        print("Vector store is empty.")
        return

    top_score = results[0].score
    print(f"Top Retrieved Similarity Score: {top_score:.4f}")

    # 2. Code-Level Refusal Check
    if top_score < threshold:
        print("\n>>> CODE-LEVEL REFUSAL TRIGGERED <<<")
        print(f"Reason: Top similarity ({top_score:.4f}) is below threshold ({threshold:.2f}).")
        print("LLM call was skipped to avoid hallucination and conserve API tokens.")
        print(f"Final Answer: {REFUSAL_MESSAGE}\n")
        return

    # 3. Grounded Generation
    prompt = format_grounded_prompt(question, results)
    print("\nRetrieval passed threshold. Sending grounded context to LLM...")

    try:
        answer = llm.generate(prompt)
    except Exception as exc:
        print(f"\nError calling LLM ({llm.provider_name}): {exc}")
        return

    citations = extract_citations_from_text(answer)

    print("\n" + "=" * 80)
    print("GROUNDED ANSWER:")
    print("=" * 80)
    print(answer)
    print("\n" + "-" * 80)
    print(f"Extracted Citations: {citations if citations else 'None'}")
    print(f"Refusal State      : {'Refused' if REFUSAL_MESSAGE.lower() in answer.lower() else 'Answered'}")
    print("=" * 80)


if __name__ == "__main__":
    test_question = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What is the deductible for windstorm damage to solar panels?"
    )
    run_ask_test(test_question)
