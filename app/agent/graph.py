"""LangGraph workflow assembly and the agent runtime.

`build_graph` compiles the state machine:  START → reason ⇄ tools → END
with a MemorySaver checkpointer and a bounded loop (`max_iterations` +
`recursion_limit`).

`AgentRuntime` wraps the graph with:
  * a per-run global timeout,
  * session threads (fresh thread per call when persistent memory is enabled,
    so history is provided by SQLite instead of the checkpointer),
  * `stream_events()` — a stable, UI-agnostic event stream (tool_call /
    tool_result / final / error) built on `graph.stream(stream_mode="updates")`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.agent.nodes import NodeContext, make_reason_node, make_tools_node, should_continue
from app.agent.state import AgentState
from app.config import Settings, get_settings
from app.llm.base import LLMError, LLMProvider
from app.logging_config import get_logger
from app.tools.registry import ToolRegistry

if TYPE_CHECKING:  # Phase 3 module — imported lazily to keep Phase 2 standalone.
    from app.memory.manager import MemoryManager

logger = get_logger("nocki.graph")


class EventType(StrEnum):
    """Types of events emitted by `AgentRuntime.stream_events`."""

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL = "final"
    ERROR = "error"


class AgentEvent(BaseModel):
    """A single UI-agnostic agent event (CLI, SSE, ...)."""

    type: EventType
    text: str
    tool: str | None = None


def build_graph(ctx: NodeContext):
    """Compile and return the agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("reason", make_reason_node(ctx))
    graph.add_node("tools", make_tools_node(ctx))
    graph.add_edge(START, "reason")
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {"tools": "tools", "finish": END},
    )
    graph.add_edge("tools", "reason")
    return graph.compile(checkpointer=MemorySaver())


class AgentRuntime:
    """High-level entrypoint used by the CLI and the API."""

    def __init__(
        self,
        llm: LLMProvider,
        settings: Settings | None = None,
        registry: ToolRegistry | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.registry = registry
        self.memory = memory
        self.ctx = NodeContext(
            llm=llm,
            settings=self.settings,
            registry=registry,
        )
        self.graph = build_graph(self.ctx)

    # -- helpers -------------------------------------------------------------
    def _config(self, thread_id: str) -> dict:
        # Each iteration costs 2 graph steps (reason + tools); +5 headroom.
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.max_iterations * 2 + 5,
        }

    def _thread_id(self, session_id: str) -> str:
        # With persistent memory, history comes from SQLite; a FRESH thread per
        # call avoids double-counting messages (checkpointer + memory).
        if self.memory is not None:
            return f"{session_id}-{uuid4().hex[:8]}"
        return session_id

    def _initial_update(self, message: str, session_id: str) -> AgentState:
        if self.memory is not None:
            return AgentState(
                messages=self._prepare_messages(session_id, message),
                task=message,
                max_iterations=self.settings.max_iterations,
            )
        return AgentState(
            messages=[HumanMessage(content=message)],
            task=message,
            max_iterations=self.settings.max_iterations,
        )

    def _prepare_messages(self, session_id: str, message: str) -> list[BaseMessage]:
        """History from SQLite + relevant context block + the new question."""
        assert self.memory is not None  # only called when memory is enabled
        history = self.memory.load_conversation(session_id)
        messages: list[BaseMessage] = list(history)
        context_block = self.memory.build_context_block(message)
        if context_block:
            messages.append(SystemMessage(content=context_block))
        messages.append(HumanMessage(content=message))
        return messages

    def _persist(self, session_id: str, message: str, state: dict[str, Any]) -> None:
        """Store a completed turn in persistent memory (best effort)."""
        if self.memory is None:
            return
        try:
            self.memory.append_turn(session_id, message, state.get("final_answer") or "")
        except Exception as exc:  # noqa: BLE001 — persistence must not kill a run
            logger.error("Failed to persist turn: %s", exc)

    # -- run (blocking, with timeout) ---------------------------------------
    def invoke(self, message: str, session_id: str = "default") -> AgentState:
        """Run the graph to completion, enforcing the global agent timeout."""
        thread_id = self._thread_id(session_id)
        config = self._config(thread_id)
        update = self._initial_update(message, session_id)

        def _run() -> AgentState:
            return self.graph.invoke(update, config=config)

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            state = pool.submit(_run).result(timeout=self.settings.agent_timeout)
        except FutureTimeout:
            logger.error("Agent run timed out after %ds", self.settings.agent_timeout)
            state = AgentState(
                final_answer=f"Délai dépassé ({self.settings.agent_timeout}s).",
                errors=["timeout"],
            )
        except LLMError as exc:
            logger.error("Agent run failed: %s", exc)
            state = AgentState(final_answer=str(exc), errors=[str(exc)])
        finally:
            pool.shutdown(wait=False)

        self._persist(session_id, message, state)
        return state

    # -- event stream ----------------------------------------------------------
    def stream_events(
        self, message: str, session_id: str = "default"
    ) -> Iterator[AgentEvent]:
        """Stream the run as AgentEvents (node-level granularity).

        The graph runs synchronously in this generator's thread; the global
        timeout is enforced between events (a single LLM call is additionally
        bounded by the provider's request timeout).
        """
        thread_id = self._thread_id(session_id)
        config = self._config(thread_id)
        update = self._initial_update(message, session_id)
        started = time.monotonic()

        state: dict[str, Any] = {}
        try:
            for chunk in self.graph.stream(update, config=config, stream_mode="updates"):
                for node_name, node_update in chunk.items():
                    yield from self._events_from_update(node_name, node_update or {})
                if time.monotonic() - started > self.settings.agent_timeout:
                    yield AgentEvent(
                        type=EventType.ERROR,
                        text=f"Délai dépassé ({self.settings.agent_timeout}s).",
                    )
                    return
            snapshot = self.graph.get_state(config)
            state = dict(snapshot.values) if snapshot and snapshot.values else {}
        except LLMError as exc:
            logger.error("Agent run failed: %s", exc)
            yield AgentEvent(type=EventType.ERROR, text=str(exc))
            state = AgentState(final_answer=str(exc), errors=[str(exc)])
        except Exception as exc:  # noqa: BLE001 — the stream must never crash callers
            logger.error("Unexpected agent failure: %s", exc)
            text = f"Erreur inattendue: {exc}"
            yield AgentEvent(type=EventType.ERROR, text=text)
            state = AgentState(final_answer=text, errors=[str(exc)])
        finally:
            self._persist(session_id, message, state)

    def _events_from_update(
        self, node_name: str, update: dict[str, Any]
    ) -> Iterator[AgentEvent]:
        """Translate one node update into UI events."""
        if node_name == "reason":
            for tool_call in update.get("tool_calls") or []:
                rendered = ", ".join(
                    f"{key}={value!r}" for key, value in tool_call.arguments.items()
                )
                yield AgentEvent(
                    type=EventType.TOOL_CALL,
                    text=f"{tool_call.name}({rendered})",
                    tool=tool_call.name,
                )
            final_answer = update.get("final_answer")
            if final_answer:
                yield AgentEvent(type=EventType.FINAL, text=final_answer)
            for error in update.get("errors") or []:
                yield AgentEvent(type=EventType.ERROR, text=error)
        elif node_name == "tools":
            for observation in update.get("observations") or []:
                text = observation.content if observation.ok else (observation.error or "")
                yield AgentEvent(
                    type=EventType.TOOL_RESULT,
                    text=text,
                    tool=observation.tool,
                )

    # -- history -----------------------------------------------------------------
    def history(self, session_id: str = "default") -> list[BaseMessage]:
        """Return the stored message history for a session."""
        if self.memory is not None:
            return self.memory.load_conversation(session_id)
        try:
            snapshot = self.graph.get_state(self._config(session_id))
            if snapshot and snapshot.values:
                return list(snapshot.values.get("messages", []))
        except Exception:  # noqa: BLE001
            pass
        return []
