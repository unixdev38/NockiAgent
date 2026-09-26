"""LLM providers behind a common, replaceable interface.

The rest of the application only depends on `LLMProvider` (see `base.py`),
never on Ollama directly. Swapping Ollama for another backend later means
adding one module here — nothing in the agent/graph changes.
"""

from app.llm.base import LLMError, LLMProvider, LLMUnavailableError
from app.llm.ollama import OllamaProvider, build_llm

__all__ = [
    "LLMProvider",
    "LLMError",
    "LLMUnavailableError",
    "OllamaProvider",
    "build_llm",
]
