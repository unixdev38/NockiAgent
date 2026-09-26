"""grep-like code search inside the workspace.

Walks the workspace, skips dependency/VCS directories and binary files, and
returns capped `file:line: excerpt` matches. All limits come from settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.security.permissions import Permission
from app.security.validation import PathTraversalError
from app.tools.base import Tool, ToolResult

#: Directories never searched (dependencies, caches, build output, VCS).
SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        "target",
        "coverage",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".eggs",
    }
)

#: Max excerpt length per match (characters).
_EXCERPT_LIMIT = 200


class SearchCodeInput(BaseModel):
    query: str = Field(min_length=1, description="Texte à rechercher (sous-chaîne)")
    path: str = Field(default=".", description="Sous-répertoire de recherche (défaut: workspace)")
    case_sensitive: bool = Field(default=False, description="Recherche sensible à la casse")


class SearchCodeTool(Tool):
    """Search for a substring across text files in the workspace."""

    name: ClassVar[str] = "search_code"
    description: ClassVar[str] = (
        "Recherche une sous-chaîne dans les fichiers texte du workspace "
        "(type grep) et renvoie les correspondances 'fichier:ligne: extrait'. "
        "Ignore les dépendances, caches et fichiers binaires."
    )
    permission: ClassVar[Permission] = Permission.READ
    args_schema: ClassVar[type[BaseModel]] = SearchCodeInput

    def run(
        self,
        query: str = "",
        path: str = ".",
        case_sensitive: bool = False,
        **_: Any,
    ) -> ToolResult:
        try:
            base = self.context.validator.resolve(path)
        except PathTraversalError as exc:
            return self._err(str(exc))
        if not base.is_dir():
            return self._err(f"Répertoire introuvable: {path}")

        settings = self.context.settings
        needle = query if case_sensitive else query.lower()
        matches: list[str] = []
        files_scanned = 0
        files_truncated = False

        for root, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in SKIPPED_DIRECTORIES)
            for file_name in sorted(files):
                if files_scanned >= settings.search_max_files:
                    files_truncated = True
                    break
                file_path = Path(root) / file_name
                files_scanned += 1
                found = self._search_in_file(file_path, needle, case_sensitive)
                if found:
                    matches.extend(found)
                    if len(matches) >= settings.search_max_results:
                        matches = matches[: settings.search_max_results]
                        break
            if files_truncated or len(matches) >= settings.search_max_results:
                break

        if not matches:
            return self._ok(f"Aucun résultat pour {query!r} ({files_scanned} fichiers parcourus).")

        header = f"{len(matches)} résultat(s) pour {query!r}"
        if files_truncated:
            header += (
                f" — recherche tronquée après {settings.search_max_files} fichiers"
                f" (SEARCH_MAX_FILES)"
            )
        return self._ok(header + "\n" + "\n".join(matches))

    def _search_in_file(
        self, file_path: Path, needle: str, case_sensitive: bool
    ) -> list[str]:
        """Return matches for one file; skip binaries/unreadable files."""
        try:
            # Strict decoding on purpose: a decode error means "binary" → skip.
            text = file_path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            return []
        found: list[str] = []
        rel = self.context.validator.relative(file_path)
        for line_number, line in enumerate(text.splitlines(), start=1):
            haystack = line if case_sensitive else line.lower()
            if needle in haystack:
                found.append(f"{rel}:{line_number}: {line.strip()[:_EXCERPT_LIMIT]}")
        return found
