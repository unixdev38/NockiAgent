"""Filesystem tools (read-only): read_file and list_directory.

All paths go through `WorkspaceValidator` — nothing outside the workspace can
be read or listed, and file reads are size-capped by configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.security.permissions import Permission
from app.security.validation import FileTooLargeError, PathTraversalError
from app.tools.base import Tool, ToolResult

#: Hard sanity cap on rendered directory trees (defense against huge dirs).
_MAX_TREE_LINES = 500


class ReadFileInput(BaseModel):
    path: str = Field(description="Chemin du fichier, relatif au workspace (ex: 'src/main.py')")


class ReadFileTool(Tool):
    """Read a text file inside the workspace (size-capped, binary-refused)."""

    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "Lit un fichier texte dans le workspace et renvoie son contenu. "
        "Refuse les fichiers binaires et les fichiers trop volumineux."
    )
    permission: ClassVar[Permission] = Permission.READ
    args_schema: ClassVar[type[BaseModel]] = ReadFileInput

    def run(self, path: str = "", **_: Any) -> ToolResult:
        try:
            content = self.context.validator.read_text(
                path, self.context.settings.max_file_read_bytes
            )
        except PathTraversalError as exc:
            return self._err(str(exc))
        except FileTooLargeError as exc:
            return self._err(str(exc))
        except FileNotFoundError:
            return self._err(f"Fichier introuvable: {path}")
        except IsADirectoryError:
            return self._err(f"Répertoire, pas un fichier: {path}")
        except UnicodeDecodeError:
            return self._err(f"Fichier binaire non lisible: {path}")
        return self._ok(content)


class ListDirectoryInput(BaseModel):
    path: str = Field(default=".", description="Répertoire à lister, relatif au workspace")
    depth: int = Field(default=1, ge=1, le=5, description="Profondeur d'exploration (1 à 5)")


class ListDirectoryTool(Tool):
    """List a workspace directory as a small tree."""

    name: ClassVar[str] = "list_directory"
    description: ClassVar[str] = (
        "Liste le contenu d'un répertoire du workspace sous forme d'arborescence "
        "(les répertoires sont suffixés par '/')."
    )
    permission: ClassVar[Permission] = Permission.READ
    args_schema: ClassVar[type[BaseModel]] = ListDirectoryInput

    def run(self, path: str = ".", depth: int = 1, **_: Any) -> ToolResult:
        try:
            base = self.context.validator.resolve(path)
        except PathTraversalError as exc:
            return self._err(str(exc))
        if not base.is_dir():
            return self._err(f"Répertoire introuvable: {path}")

        lines: list[str] = [self.context.validator.relative(base) or "."]
        self._walk(base, depth, "", lines)
        return self._ok("\n".join(lines))

    def _walk(self, directory: Path, depth: int, prefix: str, lines: list[str]) -> None:
        if len(lines) >= _MAX_TREE_LINES:
            return
        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except OSError:
            return
        for index, entry in enumerate(entries):
            if len(lines) >= _MAX_TREE_LINES:
                lines.append(f"{prefix}… (tronqué)")
                return
            is_last = index == len(entries) - 1
            connector = "└── " if is_last else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")
            if entry.is_dir() and depth > 1:
                extension = "    " if is_last else "│   "
                self._walk(entry, depth - 1, prefix + extension, lines)
