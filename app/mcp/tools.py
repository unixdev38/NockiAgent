"""Adapter exposing one MCP tool as a Nocki `Tool` (permission-checked like the others).

The LLM sees the server's own JSON schema; execution still goes through
`ToolRegistry.run()` and the application permission gate (`external=True`).
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict

from app.mcp.client import MCPManager, MCPToolInfo
from app.tools.base import Tool, ToolContext, ToolResult

_MAX_NAME = 64
_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")


class MCPArguments(BaseModel):
    """Permissive validation model: the MCP server validates its own arguments."""

    model_config = ConfigDict(extra="allow")


def mcp_tool_name(server: str, tool: str) -> str:
    """Stable, provider-safe name: `<server>__<tool>` (≤ 64 chars, [A-Za-z0-9_-])."""
    raw = f"{server}__{tool}"
    return _SAFE_RE.sub("_", raw)[:_MAX_NAME]


class MCPTool(Tool):
    """One tool discovered on an MCP server."""

    external: ClassVar[bool] = True
    args_schema: ClassVar[type[BaseModel]] = MCPArguments

    def __init__(self, context: ToolContext, manager: MCPManager, info: MCPToolInfo) -> None:
        super().__init__(context)
        self._manager = manager
        self._info = info
        self.name = mcp_tool_name(info.server, info.name)  # type: ignore[misc]
        self.description = f"[MCP:{info.server}] {info.description or info.name}"  # type: ignore[misc]
        self.permission = info.permission  # type: ignore[misc]
        self.remote_name = info.name

    def run(self, **kwargs: Any) -> ToolResult:
        ok, text = self._manager.call_tool(self._info.server, self.remote_name, kwargs)
        if ok:
            return self._ok(text)
        return self._err(text)

    def as_langchain(self) -> BaseTool:
        schema: dict[str, Any] = dict(self._info.input_schema) or {"type": "object"}
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})

        def _invoke(**kwargs: Any) -> str:
            return self.run(**kwargs).render()

        return StructuredTool.from_function(
            func=_invoke,
            name=self.name,
            description=self.description,
            args_schema=schema,
        )


def build_mcp_tools(context: ToolContext, manager: MCPManager) -> list[MCPTool]:
    """Wrap every tool discovered by the (already started) manager."""
    return [MCPTool(context, manager, info) for info in manager.tools()]
