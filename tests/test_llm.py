"""LLM provider tests: content normalization, chat(), offline handling."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import Settings
from app.llm.base import LLMUnavailableError, content_to_text
from app.llm.ollama import OllamaProvider, build_llm
from tests.conftest import FakeLLM


def test_content_to_text_variants():
    assert content_to_text("hello") == "hello"
    assert content_to_text(["a", "b"]) == "ab"
    assert content_to_text([{"text": "x"}, {"text": "y"}]) == "xy"
    assert content_to_text(None) == "None"


def test_build_llm_from_settings():
    s = Settings(_env_file=None, ollama_model="qwen3", ollama_base_url="http://localhost:11434")
    llm = build_llm(s)
    assert isinstance(llm, OllamaProvider)
    assert llm.model == "qwen3"
    assert llm.supports_tools is True


def test_is_available_false_when_offline():
    # Point at a port nothing is listening on -> unavailable, no exception.
    llm = OllamaProvider(model="x", base_url="http://localhost:1", request_timeout=1)
    assert llm.is_available() is False


def test_invoke_offline_raises_llm_unavailable():
    llm = OllamaProvider(model="x", base_url="http://localhost:1", request_timeout=1)
    with pytest.raises(LLMUnavailableError):
        llm.invoke([HumanMessage(content="hi")])


def test_set_model_rebuilds_client():
    llm = OllamaProvider(model="a", base_url="http://localhost:1")
    first = llm.client
    llm.set_model("b")
    assert llm.model == "b"
    assert llm.client is not first


def test_info_is_non_sensitive():
    llm = OllamaProvider(model="qwen3", base_url="http://localhost:11434")
    info = llm.info()
    assert info["model"] == "qwen3"
    assert info["provider"] == "OllamaProvider"
    assert "supports_tools" in info


def test_fake_llm_chat_uses_invoke(fake_llm):
    reply = fake_llm.chat([HumanMessage(content="hi")])
    assert reply == "Bonjour depuis Nocki"
    assert len(fake_llm.calls) == 1


def test_fake_llm_returns_scripted_tool_calls():
    llm = FakeLLM(
        scripted=[
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}])
        ]
    )
    ai = llm.invoke([SystemMessage(content="s")])
    assert ai.tool_calls[0]["name"] == "t"
    # Script exhausted on the next call.
    with pytest.raises(AssertionError):
        llm.invoke([SystemMessage(content="s")])
