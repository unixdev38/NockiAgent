"""Build the configured LLM provider (Ollama local or OpenRouter cloud)."""

from __future__ import annotations

from app.config import LLMProviderName, Settings, get_settings
from app.llm.base import LLMProvider
from app.llm.ollama import build_llm as build_ollama
from app.llm.openrouter import build_openrouter


def build_llm(settings: Settings | None = None) -> LLMProvider:
    """Return the provider selected by `LLM_PROVIDER` (no network I/O here)."""
    settings = settings or get_settings()
    if settings.llm_provider is LLMProviderName.OPENROUTER:
        return build_openrouter(settings)
    return build_ollama(settings)
