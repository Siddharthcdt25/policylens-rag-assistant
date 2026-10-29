"""Retrieval module: embedding generation and FAISS vector indexing."""
from app.retrieval.embedder import Embedder, get_embedder
from app.retrieval.store import VectorStore, RetrievalResult

__all__ = [
    "Embedder",
    "get_embedder",
    "VectorStore",
    "RetrievalResult",
]
