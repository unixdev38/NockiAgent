"""Workspace path validation.

`WorkspaceValidator` is the single point of control for every path the agent
touches: user/tool-provided paths are resolved against the workspace and any
attempt to escape it (absolute path outside, `..` traversal, symlink pointing
outside) is rejected with a `PathTraversalError`.

The agent (and therefore the LLM) is never trusted to "behave": these checks
are enforced by the application, not by the model.
"""

from __future__ import annotations

from pathlib import Path


class SecurityError(Exception):
    """Base class for security-related failures."""


class PathTraversalError(SecurityError):
    """Raised when a path resolves outside the allowed workspace."""


class FileTooLargeError(SecurityError):
    """Raised when a file exceeds the configured read limit."""


class WorkspaceValidator:
    """Validate and resolve paths against a fixed workspace root."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()

    # -- core ----------------------------------------------------------------
    def resolve(self, user_path: str | Path) -> Path:
        """Resolve `user_path` inside the workspace.

        Relative paths are anchored at the workspace root. The result is fully
        resolved (symlinks included) and must remain inside the workspace,
        otherwise a `PathTraversalError` is raised.
        """
        raw = Path(user_path).expanduser()
        candidate = raw if raw.is_absolute() else self.workspace / raw
        resolved = candidate.resolve()
        if resolved != self.workspace and self.workspace not in resolved.parents:
            raise PathTraversalError(
                f"Chemin refusé (hors du workspace): {user_path!r} → {resolved}"
            )
        return resolved

    def relative(self, path: str | Path) -> str:
        """Return a display-friendly path, relative to the workspace when possible."""
        try:
            return str(Path(path).resolve().relative_to(self.workspace))
        except ValueError:
            return str(path)

    # -- guarded operations ----------------------------------------------------
    def read_text(self, path: str | Path, max_bytes: int) -> str:
        """Read a text file inside the workspace, with a hard size limit.

        Raises `FileNotFoundError`, `IsADirectoryError` or `FileTooLargeError`.
        Binary files raise `UnicodeDecodeError` (strict decoding).
        """
        resolved = self.resolve(path)
        if resolved.is_dir():
            raise IsADirectoryError(f"Répertoire, pas un fichier: {path}")
        try:
            size = resolved.stat().st_size
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Fichier introuvable: {path}") from exc
        if size > max_bytes:
            raise FileTooLargeError(
                f"Fichier trop volumineux ({size} octets > limite {max_bytes}): {path}"
            )
        # Strict decoding: a binary file must fail loudly, not silently return garbage.
        return resolved.read_text(encoding="utf-8", errors="strict")
