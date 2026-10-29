"""Direct embedding generation using sentence-transformers.

Why all-MiniLM-L6-v2?
- 384 dimensions: compact vector footprint (1.5 KB per vector), fast distance calculations.
- Optimized for semantic sentence similarity on CPU (10x faster than 768-dim models).
- Generates high-quality sentence embeddings suitable for technical and legal text.

Why normalize embeddings?
FAISS inner product (`IndexFlatIP`) over L2-normalized unit vectors is mathematically
identical to Cosine Similarity:
    dot_product(u, v) / (||u|| * ||v||) = dot_product(u, v)  [when ||u|| = ||v|| = 1]
This bounds scores between -1.0 and 1.0, making threshold filtering intuitive and predictable.
"""

import logging
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper around SentenceTransformer with L2 normalization for cosine similarity."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader: loads model weights only upon first vectorization call."""
        if self._model is None:
            logger.info(f"Loading embedding model '{self.model_name}' on CPU...")
            self._model = SentenceTransformer(self.model_name, device="cpu")
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Vector dimension: {self._dimension}")
        return self._model

    @property
    def dimension(self) -> int:
        """Returns vector dimensionality (384 for all-MiniLM-L6-v2)."""
        if self._dimension is None:
            _ = self.model  # Trigger model load
        return self._dimension or 384

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Vectorize a list of document chunk strings into normalized float32 arrays.

        Args:
            texts: List of text strings to embed.

        Returns:
            2D numpy array of shape (len(texts), dimension) in float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # normalize_embeddings=True divides each vector by its Euclidean (L2) norm
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # FAISS strictly requires float32 (float64 will trigger a C++ type mismatch)
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Vectorize a single query string for FAISS retrieval.

        Args:
            query: User's question string.

        Returns:
            2D numpy array of shape (1, dimension) in float32.
        """
        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.astype(np.float32)


# Global singleton instance to avoid reloading 80MB model weights into RAM on each request
_embedder_instance: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Dependency injector returning the cached Embedder singleton."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
