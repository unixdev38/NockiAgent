"""API tests with TestClient + an injected service backed by FakeLLM.

The lifespan keeps a service already present on `app.state`, so we inject a
NockiService built with a FakeLLM and an in-memory database — no Ollama, no
files, real HTTP semantics (including SSE).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api.server import create_app
from app.memory.database import Database
from app.service import NockiService
from tests.conftest import FakeLLM


def _tool_call_reply() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "2 + 2"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )


@pytest.fixture(autouse=True)
def _reset_sse_app_status():
    """Reset sse-starlette's module-level event between tests.

    Each TestClient context runs its own event loop; sse-starlette caches
    `AppStatus.should_exit_event` globally, which breaks on the second loop.
    Real servers use a single loop for the process lifetime, so this is a
    test-environment concern only.
    """
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None
    AppStatus.should_exit = False
    yield
    AppStatus.should_exit_event = None
    AppStatus.should_exit = False


@pytest.fixture
def client(settings) -> TestClient:
    settings.ensure_directories()
    service = NockiService(
        settings, llm=FakeLLM(reply="réponse de test"), db=Database(":memory:")
    )
    app = create_app(settings)
    app.state.service = service  # bypass lifespan construction
    with TestClient(app) as test_client:
        yield test_client


# -- health --------------------------------------------------------------------
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "fake-model"
    assert body["ollama_available"] is True
    assert body["tools"] == 8
    assert body["sessions"] == 0


# -- chat (one-shot) --------------------------------------------------------------
def test_chat_creates_session(client):
    response = client.post("/chat", json={"message": "Bonjour"})
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert body["response"] == "réponse de test"
    assert body["tools_used"] == []
    assert body["errors"] == []


def test_chat_with_explicit_session(client):
    response = client.post("/chat", json={"message": "Salut", "session_id": "sess-42"})
    body = response.json()
    assert body["session_id"] == "sess-42"

    # The session is now listed with its two messages.
    sessions = client.get("/sessions").json()
    assert any(item["id"] == "sess-42" for item in sessions)
    detail = client.get("/sessions/sess-42").json()
    assert detail["session"]["id"] == "sess-42"
    assert [message["role"] for message in detail["messages"]] == ["human", "ai"]
    assert detail["messages"][0]["content"] == "Salut"


def test_chat_reports_tools_used(settings):
    settings.ensure_directories()
    llm = FakeLLM(scripted=[_tool_call_reply(), AIMessage(content="Cela fait 4.")])
    service = NockiService(settings, llm=llm, db=Database(":memory:"))
    app = create_app(settings)
    app.state.service = service
    with TestClient(app) as api:
        body = api.post("/chat", json={"message": "combien 2+2 ?"}).json()
    assert body["tools_used"] == ["calculator"]
    assert body["response"] == "Cela fait 4."
    assert body["errors"] == []


def test_chat_validation_rejects_empty_message(client):
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_llm_offline_returns_clean_errors(settings):
    settings.ensure_directories()
    service = NockiService(settings, llm=FakeLLM(available=False), db=Database(":memory:"))
    app = create_app(settings)
    app.state.service = service
    with TestClient(app) as api:
        body = api.post("/chat", json={"message": "hello"}).json()
    assert body["errors"]
    assert "réponse" in body["response"] or body["response"]  # graceful fallback text


# -- sessions CRUD -------------------------------------------------------------------
def test_sessions_unknown_404(client):
    assert client.get("/sessions/does-not-exist").status_code == 404
    assert client.delete("/sessions/does-not-exist").status_code == 404


def test_session_delete(client):
    client.post("/chat", json={"message": "Bonjour", "session_id": "doomed"})
    assert client.get("/sessions/doomed").status_code == 200

    deleted = client.delete("/sessions/doomed")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "session_id": "doomed"}

    assert client.get("/sessions/doomed").status_code == 404
    assert client.delete("/sessions/doomed").status_code == 404


def test_sessions_list_ordering(client):
    client.post("/chat", json={"message": "premier", "session_id": "older"})
    client.post("/chat", json={"message": "deuxième", "session_id": "newer"})
    sessions = client.get("/sessions").json()
    assert sessions[0]["id"] == "newer"  # most recently updated first
    assert sessions[1]["message_count"] == 2


# -- SSE streaming ----------------------------------------------------------------------
def _parse_sse(text: str) -> list[dict[str, str]]:
    """Parse an SSE body into (event, data) pairs."""
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("event:"):
            current["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current["data"] = line.split(":", 1)[1].strip()
        elif not line.strip() and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


def test_sse_stream_simple(client):
    with client.stream(
        "POST", "/chat/stream", json={"message": "Bonjour", "session_id": "sse-1"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    events = _parse_sse(body)
    names = [event["event"] for event in events]
    assert names[0] == "session"
    assert json.loads(events[0]["data"])["session_id"] == "sse-1"
    assert "final" in names
    assert names[-1] == "done"

    final = next(event for event in events if event["event"] == "final")
    assert json.loads(final["data"])["text"] == "réponse de test"


def test_sse_stream_with_tool_calls(settings):
    settings.ensure_directories()
    llm = FakeLLM(scripted=[_tool_call_reply(), AIMessage(content="Résultat: 4.")])
    service = NockiService(settings, llm=llm, db=Database(":memory:"))
    app = create_app(settings)
    app.state.service = service
    with TestClient(app) as api:
        with api.stream(
            "POST", "/chat/stream", json={"message": "2+2 ?", "session_id": "sse-2"}
        ) as response:
            body = "".join(response.iter_text())

    events = _parse_sse(body)
    names = [event["event"] for event in events]
    assert names == ["session", "tool_call", "tool_result", "final", "done"]

    tool_call = events[1]
    assert json.loads(tool_call["data"])["tool"] == "calculator"
    tool_result = events[2]
    assert "4" in json.loads(tool_result["data"])["text"]


def test_sse_persists_session(client):
    with client.stream(
        "POST", "/chat/stream", json={"message": "Bonjour", "session_id": "sse-3"}
    ) as response:
        body = "".join(response.iter_text())
    assert "done" in body

    detail = client.get("/sessions/sse-3").json()
    assert [message["role"] for message in detail["messages"]] == ["human", "ai"]


# -- service layer (shared core) -------------------------------------------------------------
def test_service_chat_and_set_model(settings):
    settings.ensure_directories()
    service = NockiService(settings, llm=FakeLLM(reply="ok"), db=Database(":memory:"))
    try:
        result = service.chat("hello", session_id="svc-1")
        assert result["response"] == "ok"
        assert result["session_id"] == "svc-1"

        service.set_model("qwen3")
        assert service.llm.model == "qwen3"
        assert service.health()["model"] == "qwen3"
    finally:
        service.close()


def test_service_tool_specs(settings):
    settings.ensure_directories()
    service = NockiService(settings, llm=FakeLLM(), db=Database(":memory:"))
    try:
        specs = service.tool_specs()
        assert {spec["name"] for spec in specs} == set(service.registry.names)
        assert all({"name", "permission", "description"} <= set(spec) for spec in specs)
    finally:
        service.close()


# -- web UI, MCP, uploads --------------------------------------------------------
def test_root_serves_chat_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Nocki Agent" in response.text
    assert "/static/app.js" in response.text


def test_static_assets_served(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_health_reports_provider(client):
    body = client.get("/health").json()
    assert body["provider"] == "ollama"
    assert body["llm_available"] is True
    assert body["mcp_servers"] == 0


def test_mcp_endpoint_empty_without_config(client):
    assert client.get("/mcp").json() == []


def test_upload_then_list_then_chat_can_read_it(client):
    response = client.post(
        "/files", files={"file": ("cours.md", b"# Cours\nLe sujet est X.", "text/markdown")}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["path"] == "uploads/cours.md"
    listed = client.get("/files").json()
    assert [f["name"] for f in listed] == ["cours.md"]


def test_upload_refused_with_400(client):
    files = {"file": ("virus.exe", b"MZ", "application/octet-stream")}
    response = client.post("/files", files=files)
    assert response.status_code == 400
    assert "non autorisé" in response.json()["detail"]


def test_upload_size_limit_enforced(client, settings):
    too_big = b"x" * (settings.upload_max_bytes + 10)
    response = client.post("/files", files={"file": ("big.txt", too_big, "text/plain")})
    assert response.status_code == 400
    assert "volumineux" in response.json()["detail"]
