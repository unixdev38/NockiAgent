"""LLM-independent tools (Phase 2).

Tools are plain Python objects with typed arguments, an explicit permission
level and a `run()` that never raises. Adding a tool means adding a module
here and registering it in `registry.build_default_registry` — the agent core
never changes.
"""

from app.tools.base import Tool, ToolContext, ToolResult
from app.tools.calculator import CalculatorTool, SafeEvalError, safe_eval
from app.tools.filesystem import ListDirectoryTool, ReadFileTool
from app.tools.git import GitDiffTool, GitLogTool, GitStatusTool, GitToolError
from app.tools.python_runner import RunPythonTool
from app.tools.registry import ToolRegistry, build_default_registry
from app.tools.search import SearchCodeTool

__all__ = [
    "CalculatorTool",
    "GitDiffTool",
    "GitLogTool",
    "GitStatusTool",
    "GitToolError",
    "ListDirectoryTool",
    "ReadFileTool",
    "RunPythonTool",
    "SafeEvalError",
    "SearchCodeTool",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "safe_eval",
]
