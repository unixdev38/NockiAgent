"""MCP server configuration (JSON file, compatible with the common `mcpServers` layout).

Example (`mcp_servers.json`)::

    {
      "mcpServers": {
        "files": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"],
          "permission": "read"
        },
        "remote": {
          "url": "https://example.com/mcp",
          "headers": {"Authorization": "Bearer ${MY_TOKEN}"}
        }
      }
    }

`${VAR}` placeholders inside `env` / `headers` values are resolved from the
process environment, so secrets never need to be written in the JSON file.
"""

from __future__ import annotations

import json
import os
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from app.security.permissions import Permission

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class MCPConfigError(ValueError):
    """Invalid MCP configuration file."""


class MCPPermission(StrEnum):
    """Declared privilege of an MCP server's tools (applied by the app)."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"

    def to_permission(self) -> Permission:
        return {
            MCPPermission.READ: Permission.READ,
            MCPPermission.WRITE: Permission.WRITE,
            MCPPermission.EXECUTE: Permission.EXECUTE,
        }[self]


class MCPServerConfig(BaseModel):
    """One MCP server: either a local stdio process or a remote HTTP endpoint."""

    name: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    permission: MCPPermission = MCPPermission.EXECUTE
    enabled: bool = True
    timeout: float = Field(default=30.0, gt=0, le=600)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_RE.match(value):
            raise ValueError("nom de serveur invalide (lettres, chiffres, '_' et '-', 32 max)")
        return value

    @model_validator(mode="after")
    def _one_transport(self) -> MCPServerConfig:
        if bool(self.command) == bool(self.url):
            raise ValueError(
                "un serveur MCP doit définir exactement un des champs 'command' ou 'url'"
            )
        return self

    @property
    def transport(self) -> str:
        return "stdio" if self.command else "http"


def resolve_placeholders(
    values: dict[str, str], environ: dict[str, str] | None = None
) -> dict[str, str]:
    """Replace `${VAR}` by the value of VAR in the environment (empty if unset)."""
    env = os.environ if environ is None else environ

    def _sub(value: str) -> str:
        return _PLACEHOLDER_RE.sub(lambda m: env.get(m.group(1), ""), value)

    return {key: _sub(value) for key, value in values.items()}


def parse_mcp_config(data: dict) -> list[MCPServerConfig]:
    """Validate a decoded JSON document and return the server configurations."""
    servers_map = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers_map, dict):
        raise MCPConfigError("le fichier doit contenir un objet 'mcpServers'")
    servers: list[MCPServerConfig] = []
    for name, raw in servers_map.items():
        if not isinstance(raw, dict):
            raise MCPConfigError(f"configuration du serveur {name!r} invalide")
        try:
            servers.append(MCPServerConfig(name=name, **{k: v for k, v in raw.items()}))
        except ValueError as exc:
            raise MCPConfigError(f"serveur {name!r}: {exc}") from exc
    return servers


def load_mcp_config(path: Path) -> list[MCPServerConfig]:
    """Load the MCP file. A missing file means "no MCP servers" (not an error)."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MCPConfigError(f"impossible de lire {path}: {exc}") from exc
    return parse_mcp_config(data)
