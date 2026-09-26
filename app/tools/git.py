"""Read-only Git tools (status / diff / log).

Phase 4 of the original roadmap ("Git + analyse de code") is delivered here as
strictly read-only inspection tools, per the handover document. There is no
way to commit, reset, push or delete anything through these tools.

Each command runs `git` as a subprocess rooted at the workspace, with a hard
timeout, and only after verifying the workspace actually is a git repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.security.permissions import Permission
from app.tools.base import Tool, ToolResult

#: Hard timeout (seconds) for every git subprocess.
GIT_TIMEOUT_SECONDS = 15


class GitToolError(Exception):
    """Raised when a git command fails or the workspace is not a repository."""


def _run_git(args: list[str], workspace: Path) -> str:
    """Run a read-only git command in the workspace and return its stdout."""
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, read-only subcommands
            ["git", *args],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitToolError("git n'est pas installé (commande introuvable).") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitToolError(f"git {' '.join(args)} a dépassé {GIT_TIMEOUT_SECONDS}s.") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "aucune sortie"
        raise GitToolError(f"git {' '.join(args)} a échoué: {detail}")
    return proc.stdout.rstrip("\n")


def _ensure_repository(workspace: Path) -> None:
    if not (workspace / ".git").exists():
        raise GitToolError(f"Le workspace n'est pas un dépôt git (pas de .git): {workspace}")


def _git_output(result: str) -> str:
    return result if result.strip() else "(vide)"


class GitStatusInput(BaseModel):
    pass


class GitStatusTool(Tool):
    """Show the working tree status (`git status --short --branch`)."""

    name: ClassVar[str] = "git_status"
    description: ClassVar[str] = (
        "Affiche l'état du dépôt git du workspace (branches, fichiers modifiés, "
        "non suivis). Lecture seule."
    )
    permission: ClassVar[Permission] = Permission.READ
    args_schema: ClassVar[type[BaseModel]] = GitStatusInput

    def run(self, **_: Any) -> ToolResult:
        workspace = self.context.validator.workspace
        try:
            _ensure_repository(workspace)
            output = _run_git(["status", "--short", "--branch"], workspace)
        except GitToolError as exc:
            return self._err(str(exc))
        return self._ok(_git_output(output))


class GitDiffInput(BaseModel):
    staged: bool = Field(default=False, description="Inclure les modifications déjà stagées")


class GitDiffTool(Tool):
    """Show pending changes (`git diff`, optionally `--staged`)."""

    name: ClassVar[str] = "git_diff"
    description: ClassVar[str] = (
        "Affiche les modifications non committées du dépôt git "
        "(par défaut: non stagées; staged=true pour inclure l'index). Lecture seule."
    )
    permission: ClassVar[Permission] = Permission.READ
    args_schema: ClassVar[type[BaseModel]] = GitDiffInput

    def run(self, staged: bool = False, **_: Any) -> ToolResult:
        workspace = self.context.validator.workspace
        try:
            _ensure_repository(workspace)
            args = ["diff"] if not staged else ["diff", "--staged"]
            output = _run_git([*args, "--stat", "--patch"], workspace)
        except GitToolError as exc:
            return self._err(str(exc))
        return self._ok(_git_output(output))


class GitLogInput(BaseModel):
    limit: int = Field(default=20, ge=1, le=500, description="Nombre maximum de commits")


class GitLogTool(Tool):
    """Show the commit history (`git log --oneline`)."""

    name: ClassVar[str] = "git_log"
    description: ClassVar[str] = (
        "Affiche l'historique des commits du dépôt git (oneline, par défaut "
        "les 20 derniers). Lecture seule."
    )
    permission: ClassVar[Permission] = Permission.READ
    args_schema: ClassVar[type[BaseModel]] = GitLogInput

    def run(self, limit: int = 20, **_: Any) -> ToolResult:
        workspace = self.context.validator.workspace
        try:
            _ensure_repository(workspace)
            output = _run_git(["log", "--oneline", f"--max-count={limit}"], workspace)
        except GitToolError as exc:
            return self._err(str(exc))
        return self._ok(_git_output(output))
