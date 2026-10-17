"""MCP support: configuration, permission gate, and a real stdio MCP server end-to-end."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.agent.nodes import NodeContext, _execute_with_permissions
from app.config import Settings
from app.mcp.client import MCPManager, render_call_result
from app.mcp.config import (
    MCPConfigError,
    MCPServerConfig,
    load_mcp_config,
    parse_mcp_config,
    resolve_placeholders,
)
from app.mcp.tools import MCPTool, mcp_tool_name
from app.memory.database import Database
from app.security.permissions import Decision, Permission, PermissionManager
from app.security.validation import WorkspaceValidator
from app.service import NockiService
from app.tools.base import ToolContext
from tests.conftest import FakeLLM

DEMO_SERVER = Path(__file__).parent / "fixtures" / "mcp_demo_server.py"


def _demo_config(permission: str = "read", **extra) -> MCPServerConfig:
    return MCPServerConfig(
        name="demo",
        command=sys.executable,
        args=[str(DEMO_SERVER)],
        permission=permission,
        **extra,
    )


# -- configuration ---------------------------------------------------------------
def test_parse_valid_config_and_defaults():
    servers = parse_mcp_config(
        {
            "mcpServers": {
                "files": {"command": "npx", "args": ["-y", "x"]},
                "web": {"url": "https://a/mcp"},
            }
        }
    )
    by_name = {s.name: s for s in servers}
    assert by_name["files"].transport == "stdio"
    assert by_name["files"].permission.value == "execute"  # safe default: most restrictive
    assert by_name["web"].transport == "http"


@pytest.mark.parametrize(
    "bad",
    [
        {"mcpServers": {"x": {"command": "a", "url": "https://b"}}},  # two transports
        {"mcpServers": {"x": {}}},  # no transport
        {"mcpServers": {"bad name!": {"command": "a"}}},  # invalid name
        {"mcpServers": {"x": {"command": "a", "permission": "root"}}},  # unknown permission
        {"servers": {}},  # wrong root key
    ],
)
def test_invalid_config_is_rejected(bad):
    with pytest.raises(MCPConfigError):
        parse_mcp_config(bad)


def test_missing_config_file_means_no_servers(tmp_path):
    assert load_mcp_config(tmp_path / "absent.json") == []


def test_invalid_json_file_raises(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(MCPConfigError):
        load_mcp_config(path)


def test_placeholders_resolved_from_environment():
    assert resolve_placeholders({"H": "Bearer ${TOKEN}"}, {"TOKEN": "abc"}) == {"H": "Bearer abc"}
    assert resolve_placeholders({"H": "x${MISSING}y"}, {}) == {"H": "xy"}


def test_tool_names_are_provider_safe():
    name = mcp_tool_name("my.server", "do thing/now" * 10)
    assert len(name) <= 64
    assert all(c.isalnum() or c in "_-" for c in name)


def test_render_call_result_truncates_and_flags_errors():
    class Item:
        def __init__(self, text=None, type_="image"):
            self.text = text
            self.type = type_

    class Result:
        def __init__(self, content, isError=False):
            self.content = content
            self.isError = isError

    assert render_call_result(Result([Item("a"), Item("b")])) == (True, "a\nb")
    ok, text = render_call_result(Result([Item(None, "image")]))
    assert ok and "non textuel" in text
    assert render_call_result(Result([Item("boom")], isError=True)) == (False, "boom")
    big = render_call_result(Result([Item("x" * 50_000)]))[1]
    assert "tronquée" in big


# -- permission gate for external tools -------------------------------------------
def test_external_write_and_execute_denied_by_default():
    pm = PermissionManager(
        Settings(_env_file=None, allow_python_exec=True, require_confirmation=False)
    )
    assert pm.check(Permission.EXECUTE, external=True).decision is Decision.DENY
    assert pm.check(Permission.WRITE, external=True).decision is Decision.DENY


def test_external_read_always_allowed():
    pm = PermissionManager(Settings(_env_file=None))
    assert pm.check(Permission.READ, external=True).decision is Decision.ALLOW


def test_external_execute_allowed_only_with_flag_and_confirmation_rule():
    flagged = Settings(_env_file=None, allow_mcp_tools=True, require_confirmation=True)
    assert (
        PermissionManager(flagged).check(Permission.EXECUTE, external=True).decision
        is Decision.CONFIRM
    )
    open_ = Settings(_env_file=None, allow_mcp_tools=True, require_confirmation=False)
    assert (
        PermissionManager(open_).check(Permission.EXECUTE, external=True).decision is Decision.ALLOW
    )


def test_builtin_permissions_unchanged():
    pm = PermissionManager(Settings(_env_file=None, allow_mcp_tools=True))
    assert pm.check(Permission.EXECUTE).decision is Decision.DENY  # no ALLOW_PYTHON_EXEC


# -- real MCP server over stdio ----------------------------------------------------
@pytest.fixture(scope="module")
def manager():
    mgr = MCPManager(
        [
            _demo_config(),
            MCPServerConfig(name="broken", command="/definitely/not/a/binary"),
            MCPServerConfig(name="off", command="unused", enabled=False),
        ],
        startup_timeout=20,
    )
    mgr.start()
    yield mgr
    mgr.close()


def test_manager_connects_and_isolates_failures(manager):
    states = {s.name: s for s in manager.status()}
    assert states["demo"].state == "connected"
    assert set(states["demo"].tools) == {"add", "shout", "boom"}
    assert states["broken"].state == "error" and states["broken"].error
    assert states["off"].state == "disabled"


def test_manager_calls_tools(manager):
    assert manager.call_tool("demo", "add", {"a": 2, "b": 3}) == (True, "5")
    assert manager.call_tool("demo", "shout", {"text": "hi"}) == (True, "HI")


def test_manager_reports_tool_errors_as_data(manager):
    ok, text = manager.call_tool("demo", "boom", {})
    assert ok is False and "kaboom" in text


def test_manager_unknown_server(manager):
    ok, text = manager.call_tool("nope", "x", {})
    assert ok is False and "indisponible" in text


def test_mcp_tool_adapter_schema_and_permission(manager, settings):
    ctx = ToolContext(settings=settings, validator=WorkspaceValidator(settings.workspace))
    info = next(t for t in manager.tools() if t.name == "add")
    tool = MCPTool(ctx, manager, info)
    assert tool.name == "demo__add"
    assert tool.external is True
    assert tool.permission is Permission.READ  # the fixture server is declared "read"
    lc = tool.as_langchain()
    assert lc.name == "demo__add"
    assert "a" in lc.args
    assert lc.invoke({"a": 4, "b": 5}) == "9"


def test_service_registers_mcp_tools_and_agent_can_call_them(tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "demo": {
                        "command": sys.executable,
                        "args": [str(DEMO_SERVER)],
                        "permission": "read",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        workspace_path=tmp_path / "ws",
        database_path=tmp_path / "db.sqlite",
        mcp_config_path=config,
        max_iterations=3,
        agent_timeout=30,
    )
    llm = FakeLLM(
        scripted=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "demo__add",
                        "args": {"a": 20, "b": 22},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Le résultat est 42."),
        ]
    )
    service = NockiService(settings, llm=llm, db=Database(":memory:"))
    try:
        assert "demo__add" in service.registry.names
        assert service.health()["mcp_servers"] == 1
        result = service.chat("Calcule 20 + 22")
        assert result["tools_used"] == ["demo__add"]
        assert result["response"] == "Le résultat est 42."
    finally:
        service.close()


def test_service_denies_write_mcp_tool_by_default(tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {"mcpServers": {"demo": {"command": sys.executable, "args": [str(DEMO_SERVER)]}}}
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        workspace_path=tmp_path / "ws",
        database_path=tmp_path / "db.sqlite",
        mcp_config_path=config,
        max_iterations=3,
        agent_timeout=30,
    )
    llm = FakeLLM(
        scripted=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "demo__shout", "args": {"text": "x"}, "id": "c1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="Refusé."),
        ]
    )
    service = NockiService(settings, llm=llm, db=Database(":memory:"))
    try:
        ctx = NodeContext(llm=llm, settings=settings, registry=service.registry)
        result = _execute_with_permissions(ctx, "demo__shout", {"text": "x"})
        assert result.ok is False
        assert "ALLOW_MCP_TOOLS" in (result.error or "")
    finally:
        service.close()
