"""Direct vector storage and retrieval using FAISS without external RAG frameworks.

Why FAISS IndexFlatIP?
- Flat Inner Product performs exhaustive exact search without quantization error or loss of recall.
- When vectors are unit-normalized (L2 norm = 1), Inner Product is mathematically identical
  to Cosine Similarity: `dot(u, v) = cos(theta)`.
- For enterprise policy documents (typically 50-20,000 chunks), IndexFlatIP executes searches
  in sub-millisecond latency on CPU, making Approximate Nearest Neighbor (ANN / HNSW) indexing
  unnecessary and premature.

Why keep metadata in JSON rather than Pickle?
- JSON is human-readable, auditable, and cross-language compatible.
- Eliminates Python pickle security risks (arbitrary code execution during deserialization)
  and avoids version incompatibilities when upgrading Python environments.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import List, Optional
import faiss
import numpy as np

from app.config import settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Retrieved chunk paired with its cosine similarity score."""
    chunk: Chunk
    score: float  # Cosine similarity in range [-1.0, 1.0]


class VectorStore:
    """In-memory FAISS vector index paired with persistent JSON chunk metadata."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # faiss.IndexFlatIP computes exact inner products (cosine similarity on normalized vectors)
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        # 1-to-1 mapping: integer row index in FAISS -> Chunk object
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return self.index.ntotal

    def clear(self) -> None:
        """Reset the index and metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add vectorized chunks to the FAISS index and metadata store.

        Args:
            chunks: List of Chunk objects with provenance metadata.
            embeddings: 2D numpy array of shape (len(chunks), dimension) in float32.
        """
        if len(chunks) == 0:
            return

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Count mismatch: got {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}."
            )

        # Ensure float32 format for FAISS C++ API
        vectors = embeddings.astype(np.float32)

        # Add vectors to FAISS index
        self.index.add(vectors)
        # Append chunk metadata in identical sequential order
        self.chunks.extend(chunks)

        logger.info(
            f"Added {len(chunks)} chunks to FAISS index. Total indexed: {self.index.ntotal}"
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = settings.TOP_K,
    ) -> list[RetrievalResult]:
        """Query the vector index for the top-k most similar chunks.

        Args:
            query_vector: 2D numpy array of shape (1, dimension) in float32.
            top_k: Number of most similar chunks to retrieve.

        Returns:
            List of RetrievalResult objects ordered by descending cosine similarity.
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on an empty vector store.")
            return []

        # Bound top_k to the number of available documents in the index
        k = min(top_k, self.index.ntotal)

        # query_vector must be 2D float32
        q = query_vector.astype(np.float32)
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)

        # faiss.search returns:
        # - distances: 2D array of inner products (cosine similarities)
        # - indices: 2D array of chunk row indices corresponding to self.chunks
        distances, indices = self.index.search(q, k)

        results: list[RetrievalResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            # FAISS returns -1 if fewer than k vectors exist in the index
            if idx == -1 or idx >= len(self.chunks):
                continue

            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(dist),
                )
            )

        return results

    def save(self, directory: Path = settings.VECTOR_STORE_DIR) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        index_file = directory / "index.faiss"
        meta_file = directory / "metadata.json"

        # 1. Write the binary FAISS index
        faiss.write_index(self.index, str(index_file))

        # 2. Write metadata as JSON
        serializable_chunks = [asdict(c) for c in self.chunks]
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dimension": self.dimension,
                    "total_chunks": len(self.chunks),
                    "chunks": serializable_chunks,
                },
                f,
                indent=2,
            )

        logger.info(f"Vector store saved to {directory} ({len(self.chunks)} items).")

    def load(self, directory: Path = settings.VECTOR_STORE_DIR) -> bool:
        """Load a persisted FAISS index and metadata from disk.

        Returns:
            True if successfully loaded, False if files do not exist.
        """
        index_file = directory / "index.faiss"
        meta_file = directory / "metadata.json"

        if not index_file.exists() or not meta_file.exists():
            logger.info(f"No existing index found at {directory}.")
            return False

        try:
            # 1. Load the binary FAISS index
            self.index = faiss.read_index(str(index_file))
            self.dimension = self.index.d

            # 2. Load chunk metadata from JSON
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.chunks = [Chunk(**item) for item in data.get("chunks", [])]

            logger.info(
                f"Loaded vector store from {directory}: {self.index.ntotal} vectors indexed."
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to load vector store from {directory}: {exc}")
            return False
