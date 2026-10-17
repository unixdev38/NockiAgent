"""OpenRouter implementation of the LLMProvider interface (cloud, open-source models).

OpenRouter exposes an OpenAI-compatible API, so we reuse `langchain_openai.ChatOpenAI`
pointed at its base URL. Only the transport changes: the agent graph, the tools and
the permission gate are identical to the local (Ollama) setup.

Notes:
- Construction performs no network I/O; a missing API key is reported at call time.
- Tool calling depends on the chosen model (must advertise `tools` support on
  OpenRouter: https://openrouter.ai/models?supported_parameters=tools).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import openai
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings
from app.llm.base import (
    LLMError,
    LLMProvider,
    LLMUnavailableError,
    ModelNotFoundError,
    content_to_text,
)


class OpenRouterProvider(LLMProvider):
    """LLMProvider backed by the OpenRouter cloud API."""

    supports_tools = True

    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.3,
        request_timeout: int = 120,
        site_url: str = "",
        app_name: str = "Nocki Agent",
    ) -> None:
        self.model = model
        self.api_key = api_key or ""
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.site_url = site_url
        self.app_name = app_name
        self._client: ChatOpenAI | None = None

    # -- internal ------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"X-Title": self.app_name}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        return headers

    @property
    def client(self) -> ChatOpenAI:
        if not self.api_key:
            raise LLMUnavailableError(
                "Clé OpenRouter absente. Définissez OPENROUTER_API_KEY dans le fichier .env."
            )
        if self._client is None:
            self._client = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,  # type: ignore[arg-type]
                base_url=self.base_url,
                temperature=self.temperature,
                timeout=self.request_timeout,
                max_retries=2,
                default_headers=self._headers(),
            )
        return self._client

    def set_model(self, model: str) -> None:
        """Switch to another OpenRouter model at runtime."""
        self.model = model
        self._client = None

    def _wrap(self, exc: Exception) -> Exception:
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, openai.AuthenticationError):
            return LLMError("Clé OpenRouter refusée (401). Vérifiez OPENROUTER_API_KEY.")
        if isinstance(exc, openai.NotFoundError):
            return ModelNotFoundError(
                f"Modèle '{self.model}' introuvable sur OpenRouter. "
                "Vérifiez OPENROUTER_MODEL (ex: qwen/qwen3-8b)."
            )
        if isinstance(exc, openai.RateLimitError):
            return LLMError("Limite de débit OpenRouter atteinte. Réessayez dans un instant.")
        if isinstance(exc, openai.APIConnectionError | openai.APITimeoutError):
            return LLMUnavailableError(f"Impossible de joindre OpenRouter ({self.base_url}).")
        return LLMError(f"Erreur OpenRouter: {exc}")

    # -- public API ----------------------------------------------------------
    def is_available(self) -> bool:
        """Cheap, offline check: a key is configured (no network call on /health)."""
        return bool(self.api_key)

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
    ) -> AIMessage:
        try:
            bound = self.client.bind_tools(list(tools)) if tools else self.client
            result: AIMessage = bound.invoke(list(messages))
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize backend errors
            raise self._wrap(exc) from exc
        return result

    def stream(self, messages: Sequence[BaseMessage]) -> Iterator[str]:
        try:
            for chunk in self.client.stream(list(messages)):
                text = content_to_text(chunk.content)
                if text:
                    yield text
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise self._wrap(exc) from exc

    def info(self) -> dict[str, str]:
        data = super().info()
        data["base_url"] = self.base_url
        return data


def build_openrouter(settings: Settings | None = None) -> OpenRouterProvider:
    settings = settings or get_settings()
    key = settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else None
    return OpenRouterProvider(
        model=settings.openrouter_model,
        api_key=key,
        base_url=settings.openrouter_base_url,
        temperature=settings.llm_temperature,
        request_timeout=settings.llm_request_timeout,
        site_url=settings.openrouter_site_url,
        app_name=settings.openrouter_app_name,
    )
