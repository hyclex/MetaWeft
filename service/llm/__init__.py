"""Logged OpenAI-compatible chat and embedding calls."""

from .client import LLMUnavailableError, ModelClient
from .cache import LLMCacheError

__all__ = ["LLMUnavailableError", "LLMCacheError", "ModelClient"]
