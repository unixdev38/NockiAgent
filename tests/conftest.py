"""Shared test fixtures and a fake LLM provider.

The fake provider lets the whole graph/CLI be tested deterministically without
a running Ollama server.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence

import pytest
from langchain_core.messages import BaseMessage

from app.config import Settings
from app.llm.base import LLMProvider, LLMUnavailableError


class FakeLLM(LLMProvider):
    """Deterministic in-memory LLM for tests."""

    def __init__(self, reply: str = "pong", available: bool = True) -> None:
        self.model = "fake-model"
        self._reply = reply
        self._available = available
        self.calls: list[Sequence[BaseMessage]] = []

    def is_available(self) -> bool:
        return self._available

    def chat(self, messages: Sequence[BaseMessage]) -> str:
        if not self._available:
            raise LLMUnavailableError("fake offline")
        self.calls.append(messages)
        return self._reply

    def stream(self, messages: Sequence[BaseMessage]) -> Iterator[str]:
        if not self._available:
            raise LLMUnavailableError("fake offline")
        self.calls.append(messages)
        for word in self._reply.split(" "):
            yield word + " "


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        workspace_path=tmp_path / "workspace",
        database_path=tmp_path / "data" / "nocki.db",
        max_iterations=3,
        agent_timeout=10,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(reply="Bonjour depuis Nocki")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Isolate tests from any developer .env / shell vars.
    for key in ("OLLAMA_MODEL", "OLLAMA_BASE_URL", "MAX_ITERATIONS", "AGENT_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    os.environ.setdefault("LOG_LEVEL", "WARNING")
