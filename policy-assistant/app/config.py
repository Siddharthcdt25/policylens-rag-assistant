"""Application configuration loaded from environment variables and .env file.

Why Pydantic Settings?
In production ML systems, type safety and validation on hyper-parameters prevent silent
failures (like negative chunk sizes or missing API keys) at startup rather than mid-request.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the RAG pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore undefined env variables without crashing
    )

    # --- LLM Provider Selection ---
    # Allowed: "gemini" or "ollama"
    LLM_BACKEND: str = "gemini"

    # Gemini configuration
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Ollama configuration (local fallback)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # --- Ingestion & Chunking Hyperparameters ---
    # Target chunk size in tokens (or words as proxy)
    CHUNK_SIZE: int = 400
    # ~15% overlap prevents splitting critical sentence boundaries across chunks
    CHUNK_OVERLAP: int = 60
    # Minimum characters threshold to consider a PDF page text-searchable.
    # Scanned pages or graphic-heavy policy endorsements return <50 chars.
    OCR_MIN_CHARS: int = 50

    # --- Retrieval Hyperparameters ---
    # Number of chunks returned for the generation prompt context
    TOP_K: int = 5
    # Strict cosine similarity threshold below which we refuse in code before calling LLM.
    # Normalized embeddings with inner product index yield cosine similarity in [-1.0, 1.0].
    # Scores below 0.35 on all-MiniLM-L6-v2 generally indicate irrelevant context.
    SIMILARITY_THRESHOLD: float = 0.35
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Storage Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    VECTOR_STORE_DIR: Path = BASE_DIR / "data" / "vector_store"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"


# Singleton instance reused throughout the application
settings = Settings()
