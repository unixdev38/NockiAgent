"""Sandboxed Python execution tool (EXECUTE permission).

Double-layered protection:
1. the tool router checks `Permission.EXECUTE` *before* calling this tool,
2. `run()` re-checks the settings (defense in depth) and then delegates to
   `app.security.sandbox.run_python_code`.

Disabled by default: both `ALLOW_PYTHON_EXEC` and `ALLOW_SHELL` must be false
for the tool to refuse — enabling either one (plus dropping
`REQUIRE_CONFIRMATION`) unlocks it.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.security.permissions import Permission
from app.security.sandbox import run_python_code
from app.tools.base import Tool, ToolResult


class RunPythonInput(BaseModel):
    code: str = Field(min_length=1, description="Code Python à exécuter (print() pour afficher)")


class RunPythonTool(Tool):
    """Execute Python code in a best-effort sandbox (isolated subprocess)."""

    name: ClassVar[str] = "run_python"
    description: ClassVar[str] = (
        "Exécute un court script Python dans un sous-processus isolé "
        "(workspace, timeout, rlimits POSIX). Désactivé par défaut — "
        "nécessite ALLOW_PYTHON_EXEC=true (ou ALLOW_SHELL=true) "
        "et REQUIRE_CONFIRMATION=false."
    )
    permission: ClassVar[Permission] = Permission.EXECUTE
    args_schema: ClassVar[type[BaseModel]] = RunPythonInput

    def run(self, code: str = "", **_: Any) -> ToolResult:
        settings = self.context.settings
        if not (settings.allow_python_exec or settings.allow_shell):
            return self._err(
                "L'exécution de code est désactivée "
                "(ALLOW_PYTHON_EXEC=false et ALLOW_SHELL=false)."
            )

        result = run_python_code(
            code,
            workspace=self.context.validator.workspace,
            timeout=settings.python_exec_timeout,
        )

        output = result.stdout.rstrip("\n")
        if result.stderr.strip():
            stderr = result.stderr.rstrip("\n")
            output = f"{output}\n[stderr]\n{stderr}" if output else f"[stderr]\n{stderr}"

        if result.timed_out:
            return self._err(
                f"Exécution interrompue (timeout {settings.python_exec_timeout}s). Sortie: {output}"
            )
        if not result.ok:
            return self._err(f"Échec (code {result.returncode}). Sortie: {output}")
        return self._ok(output or "(aucune sortie)")
