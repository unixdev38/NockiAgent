from __future__ import annotations

from app.config import Settings
from app.llm.ollama import OllamaProvider, _content_to_text, build_llm


def test_content_to_text_variants():
    assert _content_to_text("hello") == "hello"
    assert _content_to_text(["a", "b"]) == "ab"
    assert _content_to_text([{"text": "x"}, {"text": "y"}]) == "xy"


def test_build_llm_from_settings():
    s = Settings(_env_file=None, ollama_model="qwen3", ollama_base_url="http://localhost:11434")
    llm = build_llm(s)
    assert isinstance(llm, OllamaProvider)
    assert llm.model == "qwen3"


def test_is_available_false_when_offline():
    # Point at a port nothing is listening on -> unavailable, no exception.
    llm = OllamaProvider(model="x", base_url="http://localhost:1", request_timeout=1)
    assert llm.is_available() is False


def test_info_is_non_sensitive():
    llm = OllamaProvider(model="qwen3", base_url="http://localhost:11434")
    info = llm.info()
    assert info["model"] == "qwen3"
    assert "provider" in info
