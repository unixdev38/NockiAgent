"""Strongly-typed agent state for the LangGraph workflow.

The channel container (`AgentState`) is a `TypedDict` because that is what
LangGraph uses for its reducer/channel machinery. Every *value* that could
otherwise be an unstructured dict is a Pydantic model (`ToolCall`,
`Observation`, `Plan`), so we never pass loose dicts around.

The state is intentionally broader than Phase 1 needs (it already contains
`plan`, `tool_calls`, `observations`, ...) so later phases can grow into it
without breaking the graph signature.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation requested by the agent."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class Observation(BaseModel):
    """The result of executing a tool call."""

    tool: str
    ok: bool = True
    content: str = ""
    error: str | None = None


class Plan(BaseModel):
    """A lightweight, ordered plan produced by the planner node."""

    steps: list[str] = Field(default_factory=list)
    rationale: str | None = None


class AgentState(TypedDict, total=False):
    """LangGraph channel schema.

    `total=False` lets nodes return partial updates. LangGraph merges each
    returned dict into the running state (using the annotated reducers).
    """

    # Full chat history. `add_messages` appends/merges instead of overwriting.
    messages: Annotated[list[AnyMessage], add_messages]

    # The user's current task/request (plain text).
    task: str

    # Structured intermediate reasoning artifacts (populated in later phases).
    plan: Plan | None
    tool_calls: list[ToolCall]
    observations: list[Observation]

    # Loop control.
    iteration: int
    max_iterations: int

    # Terminal output + any accumulated non-fatal errors.
    final_answer: str | None
    errors: list[str]


def new_state(task: str, max_iterations: int) -> AgentState:
    """Build an initial state for a run (before messages are added)."""
    return AgentState(
        task=task,
        plan=None,
        tool_calls=[],
        observations=[],
        iteration=0,
        max_iterations=max_iterations,
        final_answer=None,
        errors=[],
    )
