"""Tool registry: validation, dispatch and LangChain binding.

The registry is the only place tools are instantiated for the agent. `run()`
validates arguments against each tool's Pydantic schema and converts every
failure (unknown tool, invalid arguments, unexpected exception) into an error
`ToolResult` — the agent loop therefore never crashes because of a tool.
"""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.config import Settings
from app.security.validation import WorkspaceValidator
from app.tools.base import Tool, ToolContext, ToolResult
from app.tools.calculator import CalculatorTool
from app.tools.filesystem import ListDirectoryTool, ReadFileTool
from app.tools.git import GitDiffTool, GitLogTool, GitStatusTool
from app.tools.python_runner import RunPythonTool
from app.tools.search import SearchCodeTool


class ToolRegistry:
    """A named, ordered collection of tools with safe dispatch."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Outil en double: {tool.name!r}")
            self._tools[tool.name] = tool
        self._langchain_cache: list[BaseTool] | None = None

    def register(self, tool: Tool) -> None:
        """Add a tool at runtime (e.g. tools discovered from MCP servers)."""
        if tool.name in self._tools:
            raise ValueError(f"Outil en double: {tool.name!r}")
        self._tools[tool.name] = tool
        self._langchain_cache = None

    # -- collection protocol ---------------------------------------------------
    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    # -- execution ---------------------------------------------------------------
    def run(self, name: str, args: dict | None = None) -> ToolResult:
        """Validate arguments and execute a tool; never raises."""
        tool = self.get(name)
        if tool is None:
            return ToolResult(tool=name, ok=False, error=f"Outil inconnu: {name!r}")
        try:
            validated = tool.args_schema.model_validate(args or {})
        except ValidationError as exc:
            return ToolResult(
                tool=name, ok=False, error=f"Arguments invalides pour {name!r}: {exc}"
            )
        try:
            return tool.run(**validated.model_dump())
        except Exception as exc:  # noqa: BLE001 — a tool failure is data, not a crash
            return ToolResult(
                tool=name, ok=False, error=f"{type(exc).__name__}: {exc}"
            )

    # -- LLM binding ---------------------------------------------------------------
    def as_langchain(self) -> list[BaseTool]:
        """All tools as LangChain StructuredTools (cached)."""
        if self._langchain_cache is None:
            self._langchain_cache = [tool.as_langchain() for tool in self]
        return self._langchain_cache

    def specs(self) -> list[dict[str, str]]:
        """Lightweight descriptions for UIs (name, permission, description)."""
        return [
            {
                "name": tool.name,
                "permission": tool.permission.name,
                "description": tool.description,
            }
            for tool in self
        ]


def build_default_registry(settings: Settings) -> ToolRegistry:
    """Instantiate every built-in tool with a shared ToolContext."""
    validator = WorkspaceValidator(settings.workspace)
    context = ToolContext(settings=settings, validator=validator)
    tools: list[Tool] = [
        CalculatorTool(context),
        ReadFileTool(context),
        ListDirectoryTool(context),
        SearchCodeTool(context),
        GitStatusTool(context),
        GitDiffTool(context),
        GitLogTool(context),
        RunPythonTool(context),
    ]
    return ToolRegistry(tools)
