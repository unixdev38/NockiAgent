"""Tool abstractions.

A tool is LLM-independent Python code with:
- a typed Pydantic `args_schema` (validated before execution),
- a `Permission` level checked by the application before execution,
- a `run()` method returning a `ToolResult` (never raising).

`as_langchain()` exposes the tool to the model as a `StructuredTool`; that
binding is *presentation only* — the actual execution always goes back through
`run()`, where validation and permissions live.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, ValidationError

from app.config import Settings
from app.security.permissions import Permission
from app.security.validation import WorkspaceValidator


class ToolResult(BaseModel):
    """Uniform, structured outcome of a tool execution."""

    tool: str
    ok: bool
    output: str = ""
    error: str | None = None

    def render(self) -> str:
        """Single-string representation used for LLM consumption and display."""
        if self.ok:
            return self.output
        return f"[error] {self.error or 'erreur inconnue'}"


class ToolContext(BaseModel):
    """Shared dependencies injected into every tool (no global state)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: Settings
    validator: WorkspaceValidator


class Tool(ABC):
    """Base class for all Nocki Agent tools."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    permission: ClassVar[Permission] = Permission.READ
    #: True for tools provided by external MCP servers (gated separately).
    external: ClassVar[bool] = False
    args_schema: ClassVar[type[BaseModel]]

    def __init__(self, context: ToolContext) -> None:
        self.context = context

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. Must never raise — return an error ToolResult."""

    # -- helpers ---------------------------------------------------------------
    def _ok(self, output: str) -> ToolResult:
        return ToolResult(tool=self.name, ok=True, output=output)

    def _err(self, error: str) -> ToolResult:
        return ToolResult(tool=self.name, ok=False, error=error)

    # -- LangChain binding -----------------------------------------------------
    def as_langchain(self) -> BaseTool:
        """Expose this tool to the LLM as a StructuredTool (rendered output)."""

        def _invoke(**kwargs: Any) -> str:
            try:
                validated = self.args_schema.model_validate(kwargs)
            except ValidationError as exc:
                return ToolResult(
                    tool=self.name, ok=False, error=f"Arguments invalides: {exc}"
                ).render()
            return self.run(**validated.model_dump()).render()

        return StructuredTool.from_function(
            func=_invoke,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )
