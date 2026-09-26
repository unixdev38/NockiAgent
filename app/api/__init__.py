"""HTTP API layer (FastAPI + SSE streaming).

The API is a *presentation* layer: every behavior lives in `NockiService`
(app/service.py), shared with the CLI. Blocking agent runs are offloaded to
worker threads; SSE events are bridged back with anyio so the event loop is
never blocked.
"""

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    HealthResponse,
    MessageOut,
    SessionDetail,
    SessionSummary,
)
from app.api.server import app, create_app

__all__ = [
    "app",
    "create_app",
    "ChatRequest",
    "ChatResponse",
    "DeleteResponse",
    "HealthResponse",
    "MessageOut",
    "SessionDetail",
    "SessionSummary",
]
