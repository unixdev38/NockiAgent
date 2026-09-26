"""Abstract LLM provider interface.

This decouples the agent from any specific inference backend. Ollama is the
default implementation, but any class implementing `LLMProvider` can be
plugged in (a cloud model, llama.cpp, vLLM, a fake for tests, ...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from langchain_core.messages import BaseMessage


class LLMError(Exception):
    """Base class for LLM-related failures."""


class LLMUnavailableError(LLMError):
    """Raised when the LLM backend cannot be reached (e.g. Ollama offline)."""


class ModelNotFoundError(LLMError):
    """Raised when the requested model is not available on the backend."""


class LLMProvider(ABC):
    """Common interface every LLM backend must implement.

    Implementations must be side-effect free on construction (no network I/O),
    so the CLI/API can start even when the backend is offline and report a
    clean error at call time.
    """

    #: Human-readable model identifier (e.g. "gemma3:1b").
    model: str

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend is reachable and the model is usable."""

    @abstractmethod
    def chat(self, messages: Sequence[BaseMessage]) -> str:
        """Return the full assistant reply for the given message history."""

    @abstractmethod
    def stream(self, messages: Sequence[BaseMessage]) -> Iterator[str]:
        """Yield the assistant reply incrementally as text chunks."""

    def info(self) -> dict[str, str]:
        """Return small, non-sensitive metadata about the provider."""
        return {"provider": type(self).__name__, "model": self.model}
