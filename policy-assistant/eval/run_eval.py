"""Evaluation harness for Policy Assistant Retrieval Pipeline.

Metrics:
- Hit@1: Fraction of queries where the correct chunk is ranked at #1.
- Hit@3: Fraction of queries where the correct chunk is in the top 3.
- Hit@5: Fraction of queries where the correct chunk is in the top 5.
- MRR (Mean Reciprocal Rank): Average of 1/rank for the first relevant chunk found.

Supports:
1. Standard single-run evaluation.
2. Comparative evaluation comparing two different chunk sizes (e.g. 200 vs 600 tokens).

Usage:
    # Run standard evaluation on current settings:
    python eval/run_eval.py

    # Compare two chunk sizes (e.g. 200 tokens vs 600 tokens):
    python eval/run_eval.py --compare 200 600
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import List, Dict, Any, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_parsed_pages
from app.retrieval.embedder import get_embedder
from app.retrieval.store import VectorStore
from scripts.create_sample_policy import generate_sample_policy


@dataclass
class EvalMetrics:
    total_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    avg_latency_ms: float
    total_chunks: int


def is_chunk_relevant(chunk_text: str, chunk_page: int, expected_page: int, expected_keywords: List[str]) -> bool:
    """A chunk is judged relevant if it matches the expected page AND contains at least one target keyword."""
    if chunk_page != expected_page:
        return False

    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in expected_keywords)


def evaluate_pipeline(
    store: VectorStore,
    questions: List[Dict[str, Any]],
    top_k: int = 5,
    verbose: bool = True,
) -> Tuple[EvalMetrics, List[Dict[str, Any]]]:
    """Execute evaluation benchmark against the active VectorStore."""
    embedder = get_embedder()
    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    reciprocal_ranks = []
    latencies = []
    query_details = []

    if verbose:
        print("\n" + "=" * 95)
        print(f"{'ID':<4} | {'Question':<45} | {'Exp Page':<8} | {'Found @ Rank':<12} | {'Top Score':<9} | {'Status'}")
        print("=" * 95)

    for item in questions:
        q_id = item["id"]
        q_text = item["question"]
        exp_page = item["expected_page"]
        exp_kws = item["expected_keywords"]

        # Measure retrieval latency
        t0 = time.perf_counter()
        q_vec = embedder.embed_query(q_text)
        results = store.search(q_vec, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        found_rank = None
        for rank, res in enumerate(results, start=1):
            if is_chunk_relevant(res.chunk.text, res.chunk.page_number, exp_page, exp_kws):
                found_rank = rank
                break

        # Record metrics
        if found_rank == 1:
            hits_1 += 1
        if found_rank is not None and found_rank <= 3:
            hits_3 += 1
        if found_rank is not None and found_rank <= 5:
            hits_5 += 1

        rr = (1.0 / found_rank) if found_rank is not None else 0.0
        reciprocal_ranks.append(rr)

        top_score = results[0].score if results else 0.0
        status_str = f"PASS (Rank #{found_rank})" if found_rank else "FAIL"

        if verbose:
            q_snippet = (q_text[:42] + "...") if len(q_text) > 45 else q_text
            rank_display = f"#{found_rank}" if found_rank else "Not Found"
            print(f"{q_id:<4} | {q_snippet:<45} | {exp_page:<8} | {rank_display:<12} | {top_score:<9.4f} | {status_str}")

        query_details.append({
            "id": q_id,
            "question": q_text,
            "expected_page": exp_page,
            "found_rank": found_rank,
            "reciprocal_rank": rr,
            "top_similarity": top_score,
            "latency_ms": elapsed_ms,
        })

    n = len(questions)
    metrics = EvalMetrics(
        total_queries=n,
        hit_at_1=hits_1 / n if n else 0.0,
        hit_at_3=hits_3 / n if n else 0.0,
        hit_at_5=hits_5 / n if n else 0.0,
        mrr=sum(reciprocal_ranks) / n if n else 0.0,
        avg_latency_ms=sum(latencies) / n if n else 0.0,
        total_chunks=len(store),
    )

    return metrics, query_details


def build_store_for_chunk_size(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> VectorStore:
    """Build a fresh ephemeral VectorStore with a specific chunk size."""
    pages = parse_pdf(pdf_path, filename=pdf_path.name)
    chunks = chunk_parsed_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedder = get_embedder()
    embeddings = embedder.embed_texts([c.text for c in chunks])

    store = VectorStore(dimension=384)
    store.add_chunks(chunks, embeddings)
    return store


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval performance.")
    parser.add_argument("--pdf", type=str, default="data/sample_policy.pdf", help="Path to policy PDF")
    parser.add_argument("--questions", type=str, default="eval/questions.json", help="Path to ground truth questions")
    parser.add_argument("--compare", nargs=2, type=int, metavar=("CHUNK_SIZE_A", "CHUNK_SIZE_B"),
                        help="Compare two chunk sizes (e.g. --compare 200 600)")
    args = parser.parse_args()

    # Ensure document exists
    pdf_path = ROOT_DIR / args.pdf
    if not pdf_path.exists():
        print(f"Policy PDF not found at {pdf_path}. Generating synthetic test policy...")
        generate_sample_policy(pdf_path)

    # Load questions
    q_file = ROOT_DIR / args.questions
    with open(q_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if args.compare:
        size_a, size_b = args.compare
        overlap_a = max(10, int(size_a * 0.15))
        overlap_b = max(10, int(size_b * 0.15))

        print(f"\n=======================================================")
        print(f" EXPERIMENT: Comparing Chunk Size {size_a} vs {size_b}")
        print(f"=======================================================")

        print(f"\n--- Indexing Config A: Chunk Size = {size_a} (Overlap = {overlap_a}) ---")
        store_a = build_store_for_chunk_size(pdf_path, chunk_size=size_a, chunk_overlap=overlap_a)
        metrics_a, _ = evaluate_pipeline(store_a, questions, verbose=False)

        print(f"--- Indexing Config B: Chunk Size = {size_b} (Overlap = {overlap_b}) ---")
        store_b = build_store_for_chunk_size(pdf_path, chunk_size=size_b, chunk_overlap=overlap_b)
        metrics_b, _ = evaluate_pipeline(store_b, questions, verbose=False)

        # Comparative Table
        print("\n" + "=" * 65)
        print(f"{'Metric':<25} | {'Config A (' + str(size_a) + ' tok)':<18} | {'Config B (' + str(size_b) + ' tok)':<18}")
        print("=" * 65)
        print(f"{'Total Chunks Created':<25} | {metrics_a.total_chunks:<18} | {metrics_b.total_chunks:<18}")
        print(f"{'Hit@1 Accuracy':<25} | {metrics_a.hit_at_1 * 100:>6.1f}%{'':<11} | {metrics_b.hit_at_1 * 100:>6.1f}%")
        print(f"{'Hit@3 Accuracy':<25} | {metrics_a.hit_at_3 * 100:>6.1f}%{'':<11} | {metrics_b.hit_at_3 * 100:>6.1f}%")
        print(f"{'Hit@5 Accuracy':<25} | {metrics_a.hit_at_5 * 100:>6.1f}%{'':<11} | {metrics_b.hit_at_5 * 100:>6.1f}%")
        print(f"{'Mean Reciprocal Rank (MRR)':<25} | {metrics_a.mrr:>6.4f}{'':<12} | {metrics_b.mrr:>6.4f}")
        print(f"{'Avg Query Latency':<25} | {metrics_a.avg_latency_ms:>6.2f} ms{'':<9} | {metrics_b.avg_latency_ms:>6.2f} ms")
        print("=" * 65)

    else:
        print(f"\nRunning retrieval benchmark on '{pdf_path.name}' ({len(questions)} test cases)...")
        store = VectorStore(dimension=384)
        loaded = store.load(ROOT_DIR / "data" / "vector_store")
        if not loaded or len(store) == 0:
            print("Vector store not found. Building fresh index with default settings...")
            store = build_store_for_chunk_size(pdf_path, chunk_size=400, chunk_overlap=60)

        metrics, details = evaluate_pipeline(store, questions, verbose=True)

        print("\n" + "=" * 45)
        print("📊 RETRIEVAL EVALUATION SUMMARY")
        print("=" * 45)
        print(f"Total Test Cases   : {metrics.total_queries}")
        print(f"Total Chunks       : {metrics.total_chunks}")
        print(f"Hit@1              : {metrics.hit_at_1 * 100:.1f}%")
        print(f"Hit@3              : {metrics.hit_at_3 * 100:.1f}%")
        print(f"Hit@5              : {metrics.hit_at_5 * 100:.1f}%")
        print(f"MRR (Mean Rec. Rank): {metrics.mrr:.4f}")
        print(f"Average Latency    : {metrics.avg_latency_ms:.2f} ms")
        print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
