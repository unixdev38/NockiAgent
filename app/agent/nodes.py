"""Graph nodes: reasoning + a permission-checked tool router.

Topology (Phase 2):  START → reason ⇄ tools → END

- `reason` calls the LLM (tools bound when supported). If the AIMessage carries
  `tool_calls` the router goes to `tools`, otherwise the loop finishes.
- `tools` is the *application-side* gate: for every requested call it checks
  the tool's `Permission` BEFORE executing anything (defense in depth — the
  tool re-checks critical settings itself). DENY and CONFIRM both result in a
  clear, non-executed ToolMessage.
- All LLM failures are captured into `state.errors` — the graph never crashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState, Observation, ToolCall
from app.config import Settings
from app.llm.base import LLMError, LLMProvider, content_to_text
from app.logging_config import get_logger
from app.security.permissions import PermissionManager
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry

logger = get_logger("nocki.agent")


@dataclass
class NodeContext:
    """Shared dependencies injected into nodes (avoids global state)."""

    llm: LLMProvider
    settings: Settings
    registry: ToolRegistry | None = None
    permissions: PermissionManager | None = None
    tools: list[BaseTool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.permissions is None:
            self.permissions = PermissionManager(self.settings)
        self.tools = []
        if self.registry is not None and self.llm.supports_tools:
            self.tools = self.registry.as_langchain()

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]


def make_reason_node(ctx: NodeContext):
    """Build the `reason` node bound to the given context."""

    def reason(state: AgentState) -> AgentState:
        iteration = state.get("iteration", 0) + 1
        max_iterations = state.get("max_iterations", 1)
        logger.info("Agent iteration: %d/%d", iteration, max_iterations)

        system = SystemMessage(
            content=build_system_prompt(
                model=ctx.llm.model,
                workspace=str(ctx.settings.workspace),
                available_tools=ctx.tool_names,
            )
        )
        history: list[BaseMessage] = list(state.get("messages", []))

        try:
            ai = ctx.llm.invoke([system, *history], tools=ctx.tools or None)
        except LLMError as exc:
            logger.error("LLM call failed: %s", exc)
            fallback = f"Je n'ai pas pu contacter le modèle local. Détail: {exc}"
            return AgentState(
                messages=[AIMessage(content=fallback)],
                iteration=iteration,
                errors=[str(exc)],
                final_answer=fallback,
            )

        update: AgentState = AgentState(messages=[ai], iteration=iteration)

        if ai.tool_calls:
            update["tool_calls"] = [
                ToolCall(
                    name=call["name"],
                    arguments=dict(call.get("args") or {}),
                    id=call.get("id"),
                )
                for call in ai.tool_calls
            ]
            if iteration >= max_iterations:
                # Bounded loop: stop here with a graceful final answer.
                message = (
                    f"J'ai atteint la limite de {max_iterations} itérations "
                    "avant de pouvoir terminer la tâche."
                )
                logger.warning(
                    "Max iterations (%d) reached with pending tool calls.", max_iterations
                )
                update["final_answer"] = message
                update["errors"] = [f"max iterations reached ({max_iterations})"]
        else:
            text = content_to_text(ai.content).strip()
            update["final_answer"] = text or "(réponse vide du modèle)"

        return update

    return reason


def make_tools_node(ctx: NodeContext):
    """Build the permission-checked `tools` node bound to the given context."""

    def execute_tools(state: AgentState) -> AgentState:
        messages: list[BaseMessage] = list(state.get("messages", []))
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return AgentState()

        new_messages: list[BaseMessage] = []
        observations: list[Observation] = []

        for call in last.tool_calls:
            name = call["name"]
            arguments: dict[str, Any] = dict(call.get("args") or {})
            call_id = call.get("id") or ""
            result = _execute_with_permissions(ctx, name, arguments)
            logger.info("Tool %s → ok=%s", name, result.ok)
            new_messages.append(
                ToolMessage(content=result.render(), tool_call_id=call_id, name=name)
            )
            observations.append(
                Observation(
                    tool=name,
                    ok=result.ok,
                    content=result.output,
                    error=result.error,
                )
            )

        return AgentState(messages=new_messages, observations=observations)

    return execute_tools


def _execute_with_permissions(
    ctx: NodeContext, name: str, arguments: dict[str, Any]
) -> ToolResult:
    """Run one tool call through the permission gate (never raises)."""
    assert ctx.permissions is not None  # ensured by NodeContext.__post_init__
    tool = ctx.registry.get(name) if ctx.registry is not None else None
    if tool is None:
        return ToolResult(tool=name, ok=False, error=f"Outil inconnu: {name!r}")

    decision = ctx.permissions.check(tool.permission, external=tool.external)
    if decision.denied:
        return ToolResult(
            tool=name,
            ok=False,
            error=f"Permission refusée ({tool.permission.name}): {decision.reason}",
        )
    if decision.needs_confirmation:
        return ToolResult(
            tool=name,
            ok=False,
            error=(
                f"L'outil {name!r} ({tool.permission.name}) requiert une confirmation "
                "utilisateur et n'a pas été exécuté. "
                "Autorisez-le via la configuration pour l'exécuter."
            ),
        )
    assert ctx.registry is not None
    return ctx.registry.run(name, arguments)


def should_continue(state: AgentState) -> str:
    """Loop-control router: "tools" to execute pending calls, "finish" to stop."""
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 1)

    if iteration >= max_iterations:
        return "finish"

    messages: list[BaseMessage] = list(state.get("messages", []))
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "finish"
