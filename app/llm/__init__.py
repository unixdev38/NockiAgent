"""LLM providers behind a common, replaceable interface.

The rest of the application only depends on `LLMProvider` (see `base.py`),
never on Ollama directly. Swapping Ollama for another backend later means
adding one module here — nothing in the agent/graph changes.
"""

from app.llm.base import (
    LLMError,
    LLMProvider,
    LLMUnavailableError,
    ModelNotFoundError,
    content_to_text,
)
from app.llm.factory import build_llm
from app.llm.ollama import OllamaProvider
from app.llm.openrouter import OpenRouterProvider

__all__ = [
    "LLMProvider",
    "LLMError",
    "LLMUnavailableError",
    "ModelNotFoundError",
    "OllamaProvider",
    "OpenRouterProvider",
    "build_llm",
    "content_to_text",
]
