"""MCP client manager: keeps one live session per configured server.

The official MCP SDK is asyncio-based while the agent graph (LangGraph) runs
synchronously in worker threads. `MCPManager` owns a dedicated event loop in a
background thread and exposes a small *synchronous* facade:

- `start()`    connects every enabled server (failures are isolated per server),
- `tools()`    lists the discovered tools,
- `call_tool()` runs one tool call with a per-call timeout,
- `status()`   reports connection state (for the UI / health),
- `close()`    shuts every session down cleanly.

Each server lives in its own task, so the SDK's context managers are entered
and exited in the same task (a requirement of anyio).
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.logging_config import get_logger
from app.mcp.config import MCPServerConfig, resolve_placeholders

logger = get_logger("nocki.mcp")

#: Hard cap on text returned by one MCP tool call (protects the LLM context).
MAX_TOOL_OUTPUT_CHARS = 20_000


@dataclass
class MCPToolInfo:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    permission: Any = None  # MCPPermission (kept loose to avoid import cycles)


class MCPServerStatus(BaseModel):
    name: str
    transport: str
    state: str  # "connected" | "starting" | "error" | "disabled"
    permission: str
    error: str | None = None
    tools: list[str] = []


@dataclass
class _ServerState:
    config: MCPServerConfig
    state: str = "starting"
    error: str | None = None
    session: Any = None
    tools: list[MCPToolInfo] = field(default_factory=list)
    stop: asyncio.Event | None = None
    task: asyncio.Task | None = None


def render_call_result(result: Any) -> tuple[bool, str]:
    """Convert an MCP CallToolResult into (ok, text)."""
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            parts.append(f"[contenu non textuel: {getattr(item, 'type', 'inconnu')}]")
    output = "\n".join(parts).strip()
    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n[… sortie tronquée …]"
    ok = not bool(getattr(result, "isError", False))
    return ok, output or ("(sortie vide)" if ok else "L'outil MCP a signalé une erreur.")


class MCPManager:
    """Synchronous facade over the asyncio-based MCP client sessions."""

    def __init__(
        self,
        servers: list[MCPServerConfig],
        *,
        startup_timeout: float = 30.0,
        environ: dict[str, str] | None = None,
    ) -> None:
        self._configs = servers
        self._startup_timeout = startup_timeout
        self._environ = environ
        self._states: dict[str, _ServerState] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> None:
        """Start the background loop and connect every enabled server (idempotent)."""
        if self._started:
            return
        self._started = True
        for cfg in self._configs:
            self._states[cfg.name] = _ServerState(
                config=cfg, state="starting" if cfg.enabled else "disabled"
            )
        if not any(cfg.enabled for cfg in self._configs):
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="nocki-mcp", daemon=True
        )
        self._thread.start()
        future = asyncio.run_coroutine_threadsafe(self._start_all(), self._loop)
        try:
            future.result(timeout=self._startup_timeout + 10)
        except Exception as exc:  # noqa: BLE001 — startup must never crash the app
            logger.error("MCP startup failed: %s", exc)

    def close(self) -> None:
        """Close every session and stop the background loop."""
        if self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(timeout=15)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP shutdown incomplete: %s", exc)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None

    # -- async internals (run on the MCP loop) -------------------------------------
    async def _start_all(self) -> None:
        waiters: list[asyncio.Event] = []
        for cfg in self._configs:
            if not cfg.enabled:
                continue
            state = self._states[cfg.name]
            ready = asyncio.Event()
            state.stop = asyncio.Event()
            state.task = asyncio.create_task(self._serve(cfg, ready))
            waiters.append(ready)
        await asyncio.gather(
            *(asyncio.wait_for(ev.wait(), self._startup_timeout) for ev in waiters),
            return_exceptions=True,
        )
        for state in self._states.values():
            if state.state == "starting":
                state.state = "error"
                state.error = "délai de connexion dépassé"

    async def _serve(self, cfg: MCPServerConfig, ready: asyncio.Event) -> None:
        from mcp import ClientSession

        state = self._states[cfg.name]
        try:
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(self._transport(cfg))
                session = await stack.enter_async_context(ClientSession(read, write))
                await asyncio.wait_for(session.initialize(), self._startup_timeout)
                listed = await asyncio.wait_for(session.list_tools(), self._startup_timeout)
                state.tools = [
                    MCPToolInfo(
                        server=cfg.name,
                        name=tool.name,
                        description=(tool.description or "").strip(),
                        input_schema=dict(tool.inputSchema or {"type": "object"}),
                        permission=cfg.permission.to_permission(),
                    )
                    for tool in listed.tools
                ]
                state.session = session
                state.state = "connected"
                logger.info("MCP server %s connected (%d tools)", cfg.name, len(state.tools))
                ready.set()
                assert state.stop is not None
                await state.stop.wait()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 — isolate failures per server
            state.state = "error"
            state.error = _short_error(exc)
            logger.error("MCP server %s failed: %s", cfg.name, state.error)
        finally:
            if state.state == "connected":
                state.state = "stopped"
            state.session = None
            state.tools = []
            ready.set()

    def _transport(self, cfg: MCPServerConfig):  # noqa: ANN202 — context manager
        if cfg.command:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            env = resolve_placeholders(cfg.env, self._environ) if cfg.env else None
            params = StdioServerParameters(command=cfg.command, args=cfg.args, env=env)
            return stdio_client(params)
        from mcp.client.streamable_http import streamablehttp_client

        headers = resolve_placeholders(cfg.headers, self._environ) if cfg.headers else None
        return streamablehttp_client(cfg.url or "", headers=headers, timeout=cfg.timeout)

    async def _call(self, server: str, tool: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        state = self._states[server]
        if state.session is None:
            return False, f"Serveur MCP '{server}' indisponible."
        result = await asyncio.wait_for(
            state.session.call_tool(tool, arguments or None), state.config.timeout
        )
        return render_call_result(result)

    async def _shutdown(self) -> None:
        for state in self._states.values():
            if state.stop is not None:
                state.stop.set()
        tasks = [s.task for s in self._states.values() if s.task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # -- public synchronous API ------------------------------------------------------
    def tools(self) -> list[MCPToolInfo]:
        return [tool for state in self._states.values() for tool in state.tools]

    def status(self) -> list[MCPServerStatus]:
        return [
            MCPServerStatus(
                name=state.config.name,
                transport=state.config.transport,
                state=state.state,
                permission=state.config.permission.value,
                error=state.error,
                tools=[tool.name for tool in state.tools],
            )
            for state in self._states.values()
        ]

    def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Run one tool call synchronously (blocking the caller, never the MCP loop)."""
        if self._loop is None or server not in self._states:
            return False, f"Serveur MCP '{server}' indisponible."
        timeout = self._states[server].config.timeout
        future = asyncio.run_coroutine_threadsafe(self._call(server, tool, arguments), self._loop)
        try:
            return future.result(timeout=timeout + 5)
        except TimeoutError:
            future.cancel()
            return False, f"Délai dépassé pour l'outil MCP '{tool}' ({timeout:.0f}s)."
        except Exception as exc:  # noqa: BLE001 — a tool failure is data
            return False, f"Erreur MCP ({tool}): {_short_error(exc)}"


def _short_error(exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError | TimeoutError):
        return "délai dépassé"
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_short_error(e) for e in exc.exceptions)
    text = str(exc).strip() or type(exc).__name__
    return text[:300]
