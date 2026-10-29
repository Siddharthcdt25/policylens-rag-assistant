"""Unit tests for FastAPI endpoints, refusal logic, and citation extraction."""

from fastapi.testclient import TestClient
import pytest
from app.main import app, extract_citations_from_text, vector_store
from app.ingestion.chunker import Chunk
from app.retrieval.embedder import get_embedder
from app.llm import REFUSAL_MESSAGE


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


def test_extract_citations_regex():
    """Verify regular expression extracts [filename p.N] format accurately."""
    sample_answer = (
        "Coverage A dwelling limit is $450,000 [policy_ho5.pdf p.1]. "
        "However, solar equipment is limited to $35,000 [endorsement_r1.pdf p.3]."
    )
    citations = extract_citations_from_text(sample_answer)
    assert citations == ["[policy_ho5.pdf p.1]", "[endorsement_r1.pdf p.3]"]

    # Deduplication test
    duplicate_answer = "Loss settlement [doc.pdf p.1] rules apply [doc.pdf p.1]."
    assert extract_citations_from_text(duplicate_answer) == ["[doc.pdf p.1]"]


def test_health_endpoint(client):
    """Verify /health returns 200 and schema payload."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "embedding_model" in data
    assert "llm_backend" in data


def test_code_level_refusal_below_threshold(client):
    """Verify that a question with low similarity triggers refusal directly in code."""
    # Seed vector store with one chunk if empty
    if len(vector_store) == 0:
        chunk = Chunk(
            chunk_id="test_c1",
            text="Standard deductible for dwelling is $1,000 per occurrence.",
            page_number=1,
            source_filename="test.pdf",
            section_title="COVERAGE",
            token_count=12,
        )
        embedder = get_embedder()
        emb = embedder.embed_texts([chunk.text])
        vector_store.add_chunks([chunk], emb)

    # Ask completely unrelated question with high threshold to guarantee refusal
    payload = {
        "question": "What is the policy regarding deep sea scuba diving in Antarctica?",
        "threshold": 0.85,
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["refused"] is True
    assert data["answer"] == REFUSAL_MESSAGE
    assert "below the required threshold" in data["refusal_reason"]
    assert len(data["retrieved_chunks"]) > 0
