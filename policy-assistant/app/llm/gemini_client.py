"""Gemini LLM Client implementation using the modern google-genai SDK.

Why google-genai?
The new Google Gen AI Python SDK (v1.0+) provides a unified, production-grade interface
for Gemini 2.5 and 2.0 models with direct configuration controls.
"""

import logging
import os
from google import genai
from google.genai import types

from app.config import settings
from app.llm.base import LLMClient

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """Client for Google Gemini models using the modern google-genai SDK."""

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "",
    ):
        self._api_key = api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        self._model_name = model_name or settings.GEMINI_MODEL or "gemini-2.5-flash"

        if not self._api_key:
            logger.warning(
                "GEMINI_API_KEY is not set. Gemini calls will fail unless an API key is provided "
                "in .env or the GEMINI_API_KEY environment variable."
            )

        # Initialize official GenAI client
        self.client = genai.Client(api_key=self._api_key if self._api_key else None)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        """Call the Gemini API with temperature=0.0 for deterministic factual grounding."""
        if not self._api_key:
            raise ValueError(
                "Cannot generate with Gemini: GEMINI_API_KEY is missing. "
                "Set it in your .env file or export GEMINI_API_KEY='your_key'."
            )

        try:
            # Temperature=0.0 minimizes hallucination in legal QA
            config = types.GenerateContentConfig(
                temperature=0.0,
            )

            response = self.client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=config,
            )

            if response.text is None:
                return "Not found in the provided documents."

            return response.text.strip()

        except Exception as exc:
            logger.error(f"Gemini API generation failed: {exc}")
            raise RuntimeError(f"Gemini generation error: {exc}") from exc
