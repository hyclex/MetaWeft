"""Logged OpenAI-compatible chat and embedding calls."""

from .client import LLMUnavailableError, ModelClient

__all__ = ["LLMUnavailableError", "ModelClient"]
