"""Unit tests for retrieval module: FAISS VectorStore, Embedder, and L2 normalization."""

from pathlib import Path
import numpy as np
import pytest
from app.ingestion.chunker import Chunk
from app.retrieval.embedder import Embedder
from app.retrieval.store import VectorStore


def test_embedder_normalization():
    """Verify that embeddings are L2 normalized (||v|| = 1.0) and in float32 format."""
    embedder = Embedder()
    texts = [
        "Solar equipment coverage up to $35,000.",
        "General exclusion for earth movement and flood damage.",
    ]
    vectors = embedder.embed_texts(texts)

    # Check shape and dtype
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32

    # Verify L2 norm is ~1.0 for each vector
    norms = np.linalg.norm(vectors, axis=1)
    for norm in norms:
        assert pytest.approx(norm, abs=1e-5) == 1.0


def test_vector_store_exact_retrieval(tmp_path: Path):
    """Verify FAISS IndexFlatIP returns the exact most relevant chunk with correct score."""
    store = VectorStore(dimension=384)

    # Create synthetic chunks
    chunk_1 = Chunk(
        chunk_id="doc_p1_c0",
        text="Solar photovoltaic panels are covered up to $35,000.",
        page_number=1,
        source_filename="test.pdf",
        section_title="SCHEDULED FLOATER",
        token_count=15,
    )
    chunk_2 = Chunk(
        chunk_id="doc_p2_c1",
        text="Earthquake and mudslide damage is strictly excluded.",
        page_number=2,
        source_filename="test.pdf",
        section_title="EXCLUSIONS",
        token_count=14,
    )

    embedder = Embedder()
    embeddings = embedder.embed_texts([chunk_1.text, chunk_2.text])
    store.add_chunks([chunk_1, chunk_2], embeddings)

    assert len(store) == 2

    # Query for solar panels
    q_vec = embedder.embed_query("What is the solar panel limit?")
    results = store.search(q_vec, top_k=2)

    assert len(results) == 2
    # Chunk 1 (solar) must be ranked #1
    assert results[0].chunk.chunk_id == "doc_p1_c0"
    assert results[0].score > results[1].score
    assert -1.0 <= results[0].score <= 1.0

    # Test persistence (save and load)
    save_dir = tmp_path / "test_store"
    store.save(save_dir)

    loaded_store = VectorStore(dimension=384)
    assert loaded_store.load(save_dir) is True
    assert len(loaded_store) == 2
    assert loaded_store.chunks[0].chunk_id == "doc_p1_c0"
