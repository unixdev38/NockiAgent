"""OpenRouter provider: factory selection, offline-safe construction, error mapping."""

from __future__ import annotations

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
from app.llm.base import LLMError, LLMUnavailableError, ModelNotFoundError
from app.llm.factory import build_llm
from app.llm.ollama import OllamaProvider
from app.llm.openrouter import OpenRouterProvider


def _settings(**overrides) -> Settings:
    base = {"_env_file": None, "llm_provider": "openrouter", "openrouter_api_key": "sk-test"}
    base.update(overrides)
    return Settings(**base)


def test_factory_selects_openrouter_when_configured():
    llm = build_llm(_settings(openrouter_model="qwen/qwen3-8b"))
    assert isinstance(llm, OpenRouterProvider)
    assert llm.model == "qwen/qwen3-8b"
    assert llm.supports_tools is True
    assert llm.base_url == "https://openrouter.ai/api/v1"


def test_factory_defaults_to_ollama():
    assert isinstance(build_llm(Settings(_env_file=None)), OllamaProvider)


def test_construction_is_offline_and_key_is_secret():
    settings = _settings()
    llm = build_llm(settings)
    assert llm.is_available() is True
    assert "sk-test" not in repr(settings)  # SecretStr never leaks in repr/logs


def test_missing_key_is_reported_at_call_time_not_at_startup():
    llm = build_llm(_settings(openrouter_api_key=None))
    assert llm.is_available() is False
    with pytest.raises(LLMUnavailableError, match="OPENROUTER_API_KEY"):
        llm.invoke([HumanMessage(content="hi")])


def test_client_points_to_openrouter_with_attribution_headers():
    llm = OpenRouterProvider(
        model="m",
        api_key="k",
        site_url="https://example.org",
        app_name="Nocki",
    )
    client = llm.client
    assert str(client.openai_api_base).rstrip("/") == "https://openrouter.ai/api/v1"
    assert client.default_headers["X-Title"] == "Nocki"
    assert client.default_headers["HTTP-Referer"] == "https://example.org"


class _FakeBound:
    def __init__(self, reply: AIMessage | Exception):
        self.reply = reply

    def invoke(self, messages):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class _FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return _FakeBound(self.reply)

    def invoke(self, messages):
        return _FakeBound(self.reply).invoke(messages)


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def test_invoke_returns_ai_message_and_binds_tools():
    llm = OpenRouterProvider(model="m", api_key="k")
    fake = _FakeClient(AIMessage(content="ok"))
    llm._client = fake  # type: ignore[assignment]
    tool = type("T", (), {})  # any object is enough for the binding spy below
    result = llm.invoke([HumanMessage(content="hi")], tools=[tool()])  # type: ignore[list-item]
    assert result.content == "ok"
    assert fake.bound_tools is not None


@pytest.mark.parametrize(
    ("exc", "expected_type", "fragment"),
    [
        (
            openai.AuthenticationError(
                "bad key", response=httpx.Response(401, request=_request()), body=None
            ),
            LLMError,
            "401",
        ),
        (
            openai.NotFoundError(
                "no model", response=httpx.Response(404, request=_request()), body=None
            ),
            ModelNotFoundError,
            "OPENROUTER_MODEL",
        ),
        (
            openai.APIConnectionError(request=_request()),
            LLMUnavailableError,
            "joindre OpenRouter",
        ),
    ],
)
def test_backend_errors_are_normalized(exc, expected_type, fragment):
    llm = OpenRouterProvider(model="m", api_key="k")
    llm._client = _FakeClient(exc)  # type: ignore[assignment]
    with pytest.raises(expected_type, match=fragment):
        llm.invoke([HumanMessage(content="hi")])


def test_set_model_rebuilds_client():
    llm = OpenRouterProvider(model="a", api_key="k")
    _ = llm.client
    llm.set_model("b")
    assert llm.model == "b"
    assert llm._client is None
