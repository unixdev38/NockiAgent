"""Permission model.

The LLM is untrusted: whether an action may run is decided HERE, by the
application, based on configuration — never by the model. Every tool declares
the `Permission` level it needs and the tool router checks it before executing
anything.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

from pydantic import BaseModel

from app.config import Settings


class Permission(IntEnum):
    """Privilege level required by a tool."""

    READ = 0
    WRITE = 1
    EXECUTE = 2


class Decision(StrEnum):
    """Outcome of a permission check."""

    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


class PermissionResult(BaseModel):
    """Structured result of a permission check."""

    decision: Decision
    reason: str

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW

    @property
    def needs_confirmation(self) -> bool:
        return self.decision is Decision.CONFIRM

    @property
    def denied(self) -> bool:
        return self.decision is Decision.DENY


class PermissionManager:
    """Compute the decision for a permission level from the settings."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def check(self, level: Permission) -> PermissionResult:
        if level is Permission.READ:
            # Reading inside the (already sandboxed) workspace is always fine.
            return PermissionResult(
                decision=Decision.ALLOW,
                reason="Lecture autorisée dans le workspace.",
            )

        if level is Permission.WRITE:
            if not self.settings.allow_file_write:
                return PermissionResult(
                    decision=Decision.DENY,
                    reason="Écriture de fichiers désactivée (ALLOW_FILE_WRITE=false).",
                )
            if self.settings.require_confirmation:
                return PermissionResult(
                    decision=Decision.CONFIRM,
                    reason=(
                        "Écriture autorisée mais confirmation requise "
                        "(REQUIRE_CONFIRMATION=true)."
                    ),
                )
            return PermissionResult(
                decision=Decision.ALLOW,
                reason="Écriture autorisée par la configuration.",
            )

        if level is Permission.EXECUTE:
            if not (self.settings.allow_shell or self.settings.allow_python_exec):
                return PermissionResult(
                    decision=Decision.DENY,
                    reason="Exécution désactivée (ALLOW_SHELL=false et ALLOW_PYTHON_EXEC=false).",
                )
            if self.settings.require_confirmation:
                return PermissionResult(
                    decision=Decision.CONFIRM,
                    reason=(
                        "Exécution autorisée mais confirmation requise "
                        "(REQUIRE_CONFIRMATION=true)."
                    ),
                )
            return PermissionResult(
                decision=Decision.ALLOW,
                reason="Exécution autorisée par la configuration.",
            )

        return PermissionResult(
            decision=Decision.DENY,
            reason=f"Niveau de permission inconnu: {level!r}",
        )
