"""FastAPI application factory.

Routes:
- GET  /health              — status snapshot (Ollama offline never 500s)
- POST /chat                — one-shot chat turn (offloaded to a worker thread)
- POST /chat/stream         — the same turn as Server-Sent Events
- GET  /sessions            — list sessions
- GET  /sessions/{id}       — session detail with messages (404 if unknown)
- DELETE /sessions/{id}     — delete a session (404 if unknown)

The agent run is blocking (LangGraph), so it is executed in a worker thread:
- /chat awaits it via `anyio.to_thread.run_sync`,
- /chat/stream iterates it in a thread and bridges events back to the event
  loop with `anyio.from_thread.run` through a memory object stream — the
  event loop is never blocked.

For tests, the lifespan keeps any service already present on `app.state`
(assignment before or after startup both work), so a service built with a
FakeLLM can be injected.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    HealthResponse,
    MessageOut,
    SessionDetail,
    SessionSummary,
)
from app.config import Settings, get_settings
from app.service import NockiService


@asynccontextmanager
async def _lifespan(app: FastAPI, settings: Settings) -> AsyncIterator[None]:
    # Keep an injected service (tests) if present.
    if getattr(app.state, "service", None) is None:
        app.state.service = NockiService(settings)
    yield
    service: NockiService | None = getattr(app.state, "service", None)
    if service is not None:
        service.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application (module-level `app` is created below)."""
    effective_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with _lifespan(app, effective_settings):
            yield

    app = FastAPI(
        title="Nocki Agent API",
        description="Agent IA local — outils, mémoire persistante et streaming SSE.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _service(request: Request) -> NockiService:
        service: NockiService = request.app.state.service
        return service

    # -- health ---------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        snapshot = await anyio.to_thread.run_sync(_service(request).health)
        return HealthResponse(**snapshot)

    # -- chat (one-shot) --------------------------------------------------------
    @app.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        service = _service(request)
        session_id = payload.resolved_session()
        result = await anyio.to_thread.run_sync(service.chat, payload.message, session_id)
        return ChatResponse(**result)

    # -- chat (SSE) ------------------------------------------------------------------
    @app.post("/chat/stream")
    async def chat_stream(payload: ChatRequest, request: Request) -> EventSourceResponse:
        service = _service(request)
        session_id = payload.resolved_session()
        generator = _sse_generator(service, payload.message, session_id)
        return EventSourceResponse(generator)

    # -- sessions -----------------------------------------------------------------------
    @app.get("/sessions", response_model=list[SessionSummary])
    async def list_sessions(request: Request) -> list[SessionSummary]:
        sessions = await anyio.to_thread.run_sync(_service(request).list_sessions)
        return [SessionSummary.from_session(info) for info in sessions]

    @app.get("/sessions/{session_id}", response_model=SessionDetail)
    async def get_session(session_id: str, request: Request) -> SessionDetail:
        service = _service(request)
        info = await anyio.to_thread.run_sync(service.get_session, session_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Session inconnue: {session_id}")
        records = await anyio.to_thread.run_sync(service.get_messages, session_id)
        return SessionDetail(
            session=SessionSummary.from_session(info),
            messages=[MessageOut.from_record(record) for record in records],
        )

    @app.delete("/sessions/{session_id}", response_model=DeleteResponse)
    async def delete_session(session_id: str, request: Request) -> DeleteResponse:
        deleted = await anyio.to_thread.run_sync(
            _service(request).delete_session, session_id
        )
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Session inconnue: {session_id}")
        return DeleteResponse(deleted=True, session_id=session_id)

    return app


async def _sse_generator(
    service: NockiService, message: str, session_id: str
) -> AsyncIterator[dict[str, str]]:
    """Bridge the blocking agent event stream to SSE without blocking the loop."""
    send, receive = anyio.create_memory_object_stream[dict[str, str]](max_buffer_size=64)

    def _produce_sync() -> None:
        """Runs in a worker thread: iterate blocking events, push to the stream."""
        try:
            for event in service.stream_events(message, session_id):
                anyio.from_thread.run(
                    send.send, {"event": event.type.value, "data": event.model_dump_json()}
                )
        except Exception as exc:  # noqa: BLE001 — surface as an SSE error event
            try:
                anyio.from_thread.run(
                    send.send,
                    {"event": "error", "data": json.dumps({"error": str(exc)})},
                )
            except Exception:  # noqa: BLE001 — client already gone
                pass
        finally:
            try:
                anyio.from_thread.run(send.send, {"event": "done", "data": "{}"})
                anyio.from_thread.run(send.aclose)
            except Exception:  # noqa: BLE001 — client already gone
                pass

    async with anyio.create_task_group() as tg:
        tg.start_soon(anyio.to_thread.run_sync, _produce_sync)
        yield {"event": "session", "data": json.dumps({"session_id": session_id})}
        async with receive:
            async for item in receive:
                yield item


app = create_app()
