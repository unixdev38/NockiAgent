"""Abstract LLM provider interface.

This decouples the agent from any specific inference backend. Ollama is the
default implementation, but any class implementing `LLMProvider` can be
plugged in (a cloud model, llama.cpp, vLLM, a fake for tests, ...).

Since Phase 2 the interface is tool-aware: `invoke()` returns the raw
`AIMessage` (which may carry `tool_calls`) and optionally receives the tools
to bind. `chat()` is a convenience wrapper returning plain text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool


class LLMError(Exception):
    """Base class for LLM-related failures."""


class LLMUnavailableError(LLMError):
    """Raised when the LLM backend cannot be reached (e.g. Ollama offline)."""


class ModelNotFoundError(LLMError):
    """Raised when the requested model is not available on the backend."""


def content_to_text(content: object) -> str:
    """Normalize LangChain message content (str or list of parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
        return "".join(parts)
    return str(content)


class LLMProvider(ABC):
    """Common interface every LLM backend must implement.

    Implementations must be side-effect free on construction (no network I/O),
    so the CLI/API can start even when the backend is offline and report a
    clean error at call time.
    """

    #: Human-readable model identifier (e.g. "gemma3:1b").
    model: str

    #: Whether this backend supports native tool calling.
    supports_tools: bool = True

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend is reachable and the model is usable."""

    @abstractmethod
    def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
    ) -> AIMessage:
        """Return the raw AIMessage for the given history (may contain tool_calls)."""

    @abstractmethod
    def stream(self, messages: Sequence[BaseMessage]) -> Iterator[str]:
        """Yield the assistant reply incrementally as text chunks."""

    def chat(self, messages: Sequence[BaseMessage]) -> str:
        """Return the full assistant reply as plain text."""
        return content_to_text(self.invoke(messages).content)

    def info(self) -> dict[str, str]:
        """Return small, non-sensitive metadata about the provider."""
        return {
            "provider": type(self).__name__,
            "model": self.model,
            "supports_tools": str(self.supports_tools),
        }
