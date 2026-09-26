"""NockiService — the shared application core behind every interface.

Both the CLI and the HTTP API are thin layers over this class (DRY: one
behavior everywhere). It wires together the LLM provider, the tool registry,
persistent memory and the agent runtime, and exposes a small, injectable
surface (`llm` and `db` can be swapped for tests).
"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

from app.agent.graph import AgentEvent, AgentRuntime
from app.config import Settings, get_settings
from app.llm.base import LLMProvider
from app.llm.ollama import build_llm
from app.logging_config import get_logger
from app.memory.database import Database
from app.memory.manager import MemoryManager
from app.memory.models import MessageRecord, SessionInfo
from app.tools.registry import ToolRegistry, build_default_registry

logger = get_logger("nocki.service")


class NockiService:
    """Injectable application core (LLM + tools + memory + runtime)."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMProvider | None = None,
        db: Database | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or build_llm(self.settings)
        self.database = db or Database(self.settings.database_path)
        self.memory = MemoryManager(self.database)
        self.registry: ToolRegistry = build_default_registry(self.settings)
        self.runtime = AgentRuntime(
            self.llm, self.settings, registry=self.registry, memory=self.memory
        )

    # -- introspection ---------------------------------------------------------
    def health(self) -> dict:
        """Non-sensitive status snapshot (never raises on offline Ollama)."""
        try:
            sessions = len(self.memory.list_sessions())
        except Exception:  # noqa: BLE001 — health must always answer
            sessions = -1
        return {
            "status": "ok",
            "ollama_available": self.llm.is_available(),
            "model": self.llm.model,
            "workspace": str(self.settings.workspace),
            "database_path": str(self.settings.database_path),
            "tools": len(self.registry),
            "sessions": sessions,
        }

    def tool_specs(self) -> list[dict[str, str]]:
        return self.registry.specs()

    # -- chat ---------------------------------------------------------------------
    def new_session_id(self) -> str:
        return uuid4().hex[:12]

    def chat(self, message: str, session_id: str | None = None) -> dict:
        """Run one turn to completion; returns response + tools used + errors."""
        sid = session_id or self.new_session_id()
        self.memory.ensure_session(sid)
        state = self.runtime.invoke(message, session_id=sid)
        tool_calls = state.get("tool_calls") or []
        return {
            "session_id": sid,
            "response": state.get("final_answer") or "",
            "tools_used": list(dict.fromkeys(call.name for call in tool_calls)),
            "errors": list(state.get("errors") or []),
        }

    def stream_events(
        self, message: str, session_id: str | None = None
    ) -> Iterator[AgentEvent]:
        """Run one turn, yielding AgentEvents as they happen."""
        sid = session_id or self.new_session_id()
        self.memory.ensure_session(sid)
        yield from self.runtime.stream_events(message, session_id=sid)

    # -- sessions --------------------------------------------------------------------
    def ensure_session(self, session_id: str) -> SessionInfo:
        return self.memory.ensure_session(session_id)

    def list_sessions(self) -> list[SessionInfo]:
        return self.memory.list_sessions()

    def get_session(self, session_id: str) -> SessionInfo | None:
        return self.memory.get_session(session_id)

    def get_messages(self, session_id: str) -> list[MessageRecord]:
        return self.memory.get_messages(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self.memory.delete_session(session_id)

    # -- runtime configuration -----------------------------------------------------------
    def set_model(self, model: str) -> None:
        """Switch the LLM model at runtime (provider-level, no restart)."""
        self.llm.set_model(model)
        logger.info("Model switched to %s", model)

    def close(self) -> None:
        """Release resources (database)."""
        self.database.close()
