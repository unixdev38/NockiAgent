"""Graph nodes.

Phase 1 exposes a single controlled reasoning node (`reason`) plus loop-control
helpers. The node signatures and the `should_continue` router are already shaped
for the full agent loop, so Phase 2 can insert planner / tool-router / validator
nodes between them without changing the graph's public contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AIMessage, SystemMessage

from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from app.config import Settings
from app.llm.base import LLMError, LLMProvider
from app.logging_config import get_logger

logger = get_logger("nocki.agent")


@dataclass
class NodeContext:
    """Shared dependencies injected into nodes (avoids global state)."""

    llm: LLMProvider
    settings: Settings
    available_tools: list[str] | None = None


def make_reason_node(ctx: NodeContext):
    """Build the `reason` node bound to the given context.

    In Phase 1 this node produces the final answer directly (no tools yet). It
    prepends the system prompt, calls the LLM and appends an AIMessage. All LLM
    failures are captured into `state.errors` instead of crashing the graph.
    """

    def reason(state: AgentState) -> AgentState:
        iteration = state.get("iteration", 0) + 1
        logger.info("Agent iteration: %d", iteration)

        system = SystemMessage(
            content=build_system_prompt(
                model=ctx.llm.model,
                workspace=str(ctx.settings.workspace),
                available_tools=ctx.available_tools,
            )
        )
        history = list(state.get("messages", []))

        try:
            answer = ctx.llm.chat([system, *history])
        except LLMError as exc:
            logger.error("LLM call failed: %s", exc)
            errors = list(state.get("errors", []))
            errors.append(str(exc))
            fallback = (
                "Je n'ai pas pu contacter le modèle local. "
                f"Détail: {exc}"
            )
            return AgentState(
                messages=[AIMessage(content=fallback)],
                iteration=iteration,
                errors=errors,
                final_answer=fallback,
            )

        logger.info("Agent completed iteration %d", iteration)
        return AgentState(
            messages=[AIMessage(content=answer)],
            iteration=iteration,
            final_answer=answer,
        )

    return reason


def should_continue(state: AgentState) -> str:
    """Loop-control router.

    Returns "continue" to loop again or "finish" to stop. Phase 1 always
    finishes after one reasoning step, but the max-iterations guard is already
    enforced here so no future change can create an infinite loop.
    """
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 1)

    if iteration >= max_iterations:
        logger.warning("Max iterations (%d) reached — finishing.", max_iterations)
        return "finish"
    if state.get("final_answer"):
        return "finish"
    return "continue"
