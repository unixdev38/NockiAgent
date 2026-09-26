"""Ollama implementation of the LLMProvider interface.

Ollama is used purely for *inference*. All orchestration lives in LangGraph and
all tools are LLM-agnostic. This module wraps `langchain_ollama.ChatOllama` and
adds availability checks + clean error handling so the rest of the app never
crashes because Ollama is offline or a model is missing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from app.config import Settings, get_settings
from app.llm.base import (
    LLMProvider,
    LLMUnavailableError,
    ModelNotFoundError,
    content_to_text,
)


class OllamaProvider(LLMProvider):
    """LLMProvider backed by a local Ollama server (with native tool calling)."""

    supports_tools = True

    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float = 0.3,
        request_timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.request_timeout = request_timeout
        # Constructed lazily so importing/instantiating never does network I/O.
        self._client: ChatOllama | None = None

    # -- internal ------------------------------------------------------------
    @property
    def client(self) -> ChatOllama:
        if self._client is None:
            self._client = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                client_kwargs={"timeout": self.request_timeout},
            )
        return self._client

    def set_model(self, model: str) -> None:
        """Switch to another model at runtime (client is rebuilt lazily)."""
        self.model = model
        self._client = None

    def _list_models(self) -> list[str]:
        """Query the Ollama server for installed models. Raises if unreachable."""
        url = f"{self.base_url}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 (local URL)
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?"
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMUnavailableError(
                f"Unexpected response from Ollama at {self.base_url}."
            ) from exc
        return [m.get("name", "") for m in payload.get("models", [])]

    # -- public API ----------------------------------------------------------
    def is_available(self) -> bool:
        try:
            self._list_models()
        except LLMUnavailableError:
            return False
        return True

    def ensure_model(self) -> None:
        """Raise ModelNotFoundError if the configured model is not installed."""
        installed = self._list_models()
        # Ollama tags may be exact ("qwen3") or include a version ("qwen3:latest").
        names = {n.split(":")[0] for n in installed} | set(installed)
        if self.model not in installed and self.model.split(":")[0] not in names:
            raise ModelNotFoundError(
                f"Model '{self.model}' is not installed. "
                f"Run: ollama pull {self.model}. Installed: {', '.join(installed) or 'none'}"
            )

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
    ) -> AIMessage:
        """Invoke the model, optionally with tools bound.

        The raw AIMessage is returned as-is so the graph can inspect
        `tool_calls` and route accordingly.
        """
        try:
            bound = self.client.bind_tools(list(tools)) if tools else self.client
            result: AIMessage = bound.invoke(list(messages))
        except Exception as exc:  # noqa: BLE001 — normalize backend errors
            raise self._wrap(exc) from exc
        return result

    def stream(self, messages: Sequence[BaseMessage]) -> Iterator[str]:
        try:
            for chunk in self.client.stream(list(messages)):
                text = content_to_text(chunk.content)
                if text:
                    yield text
        except Exception as exc:  # noqa: BLE001
            raise self._wrap(exc) from exc

    def _wrap(self, exc: Exception) -> Exception:
        message = str(exc).lower()
        if "connection" in message or "refused" in message or "failed to connect" in message:
            return LLMUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?"
            )
        if "not found" in message or "no such model" in message:
            return ModelNotFoundError(
                f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
            )
        return exc


def build_llm(settings: Settings | None = None) -> OllamaProvider:
    """Factory that builds the configured LLM provider from settings."""
    settings = settings or get_settings()
    return OllamaProvider(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        request_timeout=settings.llm_request_timeout,
    )
