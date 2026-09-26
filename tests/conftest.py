"""Shared test fixtures and a deterministic fake LLM provider.

`FakeLLM` implements the Phase 2 `LLMProvider` contract (invoke + tools) and
supports a `scripted` list of AIMessages to drive the tool loop precisely —
all without a running Ollama server.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from app.config import Settings
from app.llm.base import LLMProvider, LLMUnavailableError


class FakeLLM(LLMProvider):
    """Deterministic in-memory LLM for tests.

    - `reply`: text returned when no script is provided.
    - `scripted`: optional list of AIMessages consumed one per `invoke` call
      (use AIMessage(tool_calls=[...]) to simulate tool calling).
    - `available`: set False to simulate an offline backend.
    """

    supports_tools = True

    def __init__(
        self,
        reply: str = "pong",
        available: bool = True,
        scripted: list[AIMessage] | None = None,
    ) -> None:
        self.model = "fake-model"
        self._reply = reply
        self._available = available
        self._scripted = list(scripted) if scripted else None
        self.calls: list[list[BaseMessage]] = []
        self.bound_tools: list[list[BaseTool] | None] = []

    def is_available(self) -> bool:
        return self._available

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
    ) -> AIMessage:
        if not self._available:
            raise LLMUnavailableError("fake offline")
        self.calls.append(list(messages))
        self.bound_tools.append(list(tools) if tools else None)
        if self._scripted is not None:
            if not self._scripted:
                raise AssertionError("FakeLLM script exhausted — too many invoke() calls")
            return self._scripted.pop(0)
        return AIMessage(content=self._reply)

    def stream(self, messages: Sequence[BaseMessage]) -> Iterator[str]:
        if not self._available:
            raise LLMUnavailableError("fake offline")
        self.calls.append(list(messages))
        for word in self._reply.split(" "):
            yield word + " "


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
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
    for key in (
        "OLLAMA_MODEL",
        "OLLAMA_BASE_URL",
        "MAX_ITERATIONS",
        "AGENT_TIMEOUT",
        "ALLOW_SHELL",
        "ALLOW_FILE_WRITE",
        "ALLOW_PYTHON_EXEC",
        "REQUIRE_CONFIRMATION",
        "MAX_FILE_READ_BYTES",
        "SEARCH_MAX_RESULTS",
        "SEARCH_MAX_FILES",
        "PYTHON_EXEC_TIMEOUT",
        "WORKSPACE_PATH",
        "DATABASE_PATH",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(key, raising=False)
    os.environ.setdefault("LOG_LEVEL", "WARNING")
