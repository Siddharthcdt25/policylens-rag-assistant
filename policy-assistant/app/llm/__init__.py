"""LLM provider integration with swappable backends (Gemini and Ollama)."""

from typing import Optional
from app.config import settings
from app.llm.base import LLMClient
from app.llm.gemini_client import GeminiClient
from app.llm.ollama_client import OllamaClient
from app.llm.prompts import (
    REFUSAL_MESSAGE,
    SYSTEM_INSTRUCTIONS,
    format_grounded_prompt,
    format_context_block,
)


def get_llm_client(backend: Optional[str] = None) -> LLMClient:
    """Factory function returning the configured LLM client instance.

    Args:
        backend: Optional override ("gemini" or "ollama"). Defaults to settings.LLM_BACKEND.

    Returns:
        Configured LLMClient instance.
    """
    selected = (backend or settings.LLM_BACKEND).strip().lower()

    if selected == "gemini":
        return GeminiClient()
    elif selected == "ollama":
        return OllamaClient()
    else:
        raise ValueError(
            f"Unsupported LLM_BACKEND '{selected}'. Must be either 'gemini' or 'ollama'."
        )


__all__ = [
    "LLMClient",
    "GeminiClient",
    "OllamaClient",
    "get_llm_client",
    "REFUSAL_MESSAGE",
    "SYSTEM_INSTRUCTIONS",
    "format_grounded_prompt",
    "format_context_block",
]
