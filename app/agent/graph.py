"""LangGraph workflow assembly and a small runtime wrapper.

`build_graph` compiles the state machine. `AgentRuntime` wraps it with:
  * per-run timeout,
  * session-scoped memory (in-memory checkpointer for Phase 1; SQLite arrives
    in Phase 3),
  * uniform error handling,
  * a streaming helper for the CLI/API.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import NodeContext, make_reason_node, should_continue
from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from app.config import Settings, get_settings
from app.llm.base import LLMError, LLMProvider
from app.logging_config import get_logger

logger = get_logger("nocki.graph")


def build_graph(ctx: NodeContext):
    """Compile and return the agent graph.

    Phase 1 topology:  START → reason → (continue|finish).
    The conditional edge keeps the loop bounded; later phases add planner /
    tool / validator nodes on the "continue" branch.
    """
    graph = StateGraph(AgentState)
    graph.add_node("reason", make_reason_node(ctx))
    graph.add_edge(START, "reason")
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {"continue": "reason", "finish": END},
    )
    return graph.compile(checkpointer=MemorySaver())


class AgentRuntime:
    """High-level entrypoint used by the CLI and (later) the API."""

    def __init__(
        self,
        llm: LLMProvider,
        settings: Settings | None = None,
        available_tools: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.available_tools = available_tools or []
        self.ctx = NodeContext(
            llm=llm,
            settings=self.settings,
            available_tools=self.available_tools,
        )
        self.graph = build_graph(self.ctx)

    # -- helpers -------------------------------------------------------------
    def _config(self, session_id: str) -> dict:
        return {
            "configurable": {"thread_id": session_id},
            "recursion_limit": self.settings.max_iterations * 4 + 10,
        }

    def _initial_update(self, message: str) -> AgentState:
        return AgentState(
            messages=[HumanMessage(content=message)],
            task=message,
            max_iterations=self.settings.max_iterations,
        )

    # -- run (blocking, with timeout) ---------------------------------------
    def invoke(self, message: str, session_id: str = "default") -> AgentState:
        """Run the graph to completion, enforcing the global agent timeout."""
        config = self._config(session_id)
        update = self._initial_update(message)

        def _run() -> AgentState:
            return self.graph.invoke(update, config=config)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run)
                return future.result(timeout=self.settings.agent_timeout)
        except FutureTimeout:
            logger.error("Agent run timed out after %ds", self.settings.agent_timeout)
            return AgentState(
                final_answer=f"Délai dépassé ({self.settings.agent_timeout}s).",
                errors=["timeout"],
            )
        except LLMError as exc:
            logger.error("Agent run failed: %s", exc)
            return AgentState(final_answer=str(exc), errors=[str(exc)])

    # -- streaming (token by token) -----------------------------------------
    def stream(self, message: str, session_id: str = "default") -> Iterator[str]:
        """Stream the assistant reply text incrementally.

        Phase 1 streams directly from the LLM (single reasoning step) while
        still persisting the exchange into the graph's checkpointed memory so
        multi-turn context is preserved. Errors are yielded as plain text.
        """
        config = self._config(session_id)
        # Rehydrate prior conversation from the checkpointer, if any.
        prior: list = []
        try:
            snapshot = self.graph.get_state(config)
            if snapshot and snapshot.values:
                prior = list(snapshot.values.get("messages", []))
        except Exception:  # noqa: BLE001 — no prior state is fine
            prior = []

        system = SystemMessage(
            content=build_system_prompt(
                model=self.llm.model,
                workspace=str(self.settings.workspace),
                available_tools=self.available_tools,
            )
        )
        human = HumanMessage(content=message)

        collected: list[str] = []
        start = time.monotonic()
        try:
            for chunk in self.llm.stream([system, *prior, human]):
                if time.monotonic() - start > self.settings.agent_timeout:
                    yield "\n[timeout]"
                    break
                collected.append(chunk)
                yield chunk
        except LLMError as exc:
            logger.error("Streaming failed: %s", exc)
            yield f"\n[erreur LLM] {exc}"
            return

        # Persist the turn into checkpointed memory (best effort).
        answer = "".join(collected)
        try:
            self.graph.update_state(
                config,
                {"messages": [human, AIMessage(content=answer)]},
            )
        except Exception:  # noqa: BLE001
            pass

    def history(self, session_id: str = "default") -> list:
        """Return the stored message history for a session."""
        try:
            snapshot = self.graph.get_state(self._config(session_id))
            if snapshot and snapshot.values:
                return list(snapshot.values.get("messages", []))
        except Exception:  # noqa: BLE001
            pass
        return []
