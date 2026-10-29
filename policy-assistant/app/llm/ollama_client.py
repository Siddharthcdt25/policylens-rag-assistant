"""Ollama local LLM Client using direct HTTP API calls via httpx.

Why direct HTTP via httpx?
Calling Ollama's native `/api/generate` endpoint directly eliminates third-party wrappers,
giving us exact control over request payloads, timeouts, and temperature settings.
"""

import logging
import httpx

from app.config import settings
from app.llm.base import LLMClient

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """Client for locally hosted Ollama instances (e.g. Llama 3.2)."""

    def __init__(
        self,
        base_url: str = "",
        model_name: str = "",
        timeout_seconds: float = 90.0,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self._model_name = model_name or settings.OLLAMA_MODEL or "llama3.2"
        self.timeout = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        """Call Ollama's /api/generate endpoint with temperature=0.0."""
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,  # Zero temperature for deterministic extraction
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()

        except httpx.ConnectError:
            err = (
                f"Failed to connect to Ollama at {self.base_url}. "
                "Ensure the Ollama daemon is running ('ollama serve') and model is pulled ('ollama pull llama3.2')."
            )
            logger.error(err)
            raise ConnectionError(err)
        except Exception as exc:
            logger.error(f"Ollama generation failed: {exc}")
            raise RuntimeError(f"Ollama generation error: {exc}") from exc
