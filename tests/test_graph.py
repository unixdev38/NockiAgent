from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agent.graph import AgentRuntime, build_graph
from app.agent.nodes import NodeContext, should_continue
from app.agent.state import AgentState


def test_reason_produces_answer(fake_llm, settings):
    runtime = AgentRuntime(fake_llm, settings)
    result = runtime.invoke("Bonjour", session_id="s1")
    assert result.get("final_answer") == "Bonjour depuis Nocki"
    msgs = result.get("messages", [])
    assert any(isinstance(m, AIMessage) for m in msgs)


def test_llm_unavailable_is_handled(settings):
    from tests.conftest import FakeLLM

    runtime = AgentRuntime(FakeLLM(available=False), settings)
    result = runtime.invoke("Bonjour", session_id="s2")
    # No crash; error captured and a fallback answer returned.
    assert result.get("errors")
    assert result.get("final_answer")


def test_should_continue_stops_at_max_iterations():
    state = AgentState(iteration=3, max_iterations=3)
    assert should_continue(state) == "finish"


def test_should_continue_finishes_with_answer():
    state = AgentState(iteration=1, max_iterations=8, final_answer="done")
    assert should_continue(state) == "finish"


def test_graph_compiles(fake_llm, settings):
    ctx = NodeContext(llm=fake_llm, settings=settings)
    graph = build_graph(ctx)
    assert graph is not None


def test_streaming_yields_text(fake_llm, settings):
    runtime = AgentRuntime(fake_llm, settings)
    chunks = list(runtime.stream("Salut", session_id="s3"))
    assert "".join(chunks).strip() == "Bonjour depuis Nocki"
