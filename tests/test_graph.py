"""Graph/runtime tests: tool loop, permissions, bounded loop, events, errors."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.graph import AgentRuntime, EventType, build_graph
from app.agent.nodes import NodeContext, should_continue
from app.agent.state import AgentState
from app.tools.registry import build_default_registry
from tests.conftest import FakeLLM


def _tool_call(name: str, arguments: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": call_id, "type": "tool_call"}],
    )


# --- agent without tools ------------------------------------------------------
def test_reason_produces_answer(fake_llm, settings):
    runtime = AgentRuntime(fake_llm, settings)
    result = runtime.invoke("Bonjour", session_id="s1")
    assert result.get("final_answer") == "Bonjour depuis Nocki"
    assert any(isinstance(m, AIMessage) for m in result.get("messages", []))


def test_llm_unavailable_is_handled(settings):
    runtime = AgentRuntime(FakeLLM(available=False), settings)
    result = runtime.invoke("Bonjour", session_id="s2")
    assert result.get("errors")
    assert result.get("final_answer")


def test_agent_without_registry_binds_no_tools(fake_llm, settings):
    runtime = AgentRuntime(fake_llm, settings)
    runtime.invoke("Bonjour", session_id="s3")
    assert fake_llm.bound_tools[0] is None


# --- agent with tools (scripted LLM) -------------------------------------------
def test_agent_executes_tool_and_finishes(settings):
    settings.ensure_directories()
    registry = build_default_registry(settings)
    llm = FakeLLM(
        scripted=[
            _tool_call("calculator", {"expression": "21 * 2"}, "call_1"),
            AIMessage(content="Le résultat est 42."),
        ]
    )
    runtime = AgentRuntime(llm, settings, registry=registry)
    result = runtime.invoke("calcule 21*2", session_id="s4")

    assert result["final_answer"] == "Le résultat est 42."
    assert [tc.name for tc in result["tool_calls"]] == ["calculator"]
    assert len(result["observations"]) == 1
    assert result["observations"][0].ok
    assert "42" in result["observations"][0].content
    assert not result.get("errors")


def test_agent_binds_tools_to_llm(settings):
    settings.ensure_directories()
    registry = build_default_registry(settings)
    llm = FakeLLM(scripted=[AIMessage(content="ok")])
    runtime = AgentRuntime(llm, settings, registry=registry)
    runtime.invoke("hello", session_id="s5")
    bound = llm.bound_tools[0]
    assert bound is not None
    assert {tool.name for tool in bound} == set(registry.names)


def test_execute_permission_refused_by_default(settings):
    settings.ensure_directories()
    registry = build_default_registry(settings)
    llm = FakeLLM(
        scripted=[
            _tool_call("run_python", {"code": "print('pwned')"}, "call_1"),
            AIMessage(content="D'accord, l'exécution a été refusée."),
        ]
    )
    runtime = AgentRuntime(llm, settings, registry=registry)
    result = runtime.invoke("exécute ce code", session_id="s6")

    observation = result["observations"][0]
    assert not observation.ok
    assert "désactivée" in observation.error
    assert "pwned" not in observation.content
    assert result["final_answer"] == "D'accord, l'exécution a été refusée."


def test_unknown_tool_reports_clean_error(settings):
    settings.ensure_directories()
    registry = build_default_registry(settings)
    llm = FakeLLM(
        scripted=[
            _tool_call("missile_launch", {}, "call_1"),
            AIMessage(content="Cet outil n'existe pas."),
        ]
    )
    runtime = AgentRuntime(llm, settings, registry=registry)
    result = runtime.invoke("feu", session_id="s7")
    assert not result["observations"][0].ok
    assert "Outil inconnu" in result["observations"][0].error


def test_invalid_tool_arguments_reported(settings):
    settings.ensure_directories()
    registry = build_default_registry(settings)
    llm = FakeLLM(
        scripted=[
            _tool_call("calculator", {"expression": 12345}, "call_1"),
            AIMessage(content="Arguments invalides."),
        ]
    )
    runtime = AgentRuntime(llm, settings, registry=registry)
    result = runtime.invoke("calcule", session_id="s8")
    assert not result["observations"][0].ok
    assert "Arguments invalides" in result["observations"][0].error


# --- bounded loop ------------------------------------------------------------------
def test_max_iterations_stops_gracefully(settings):
    settings.ensure_directories()
    settings.max_iterations = 1
    registry = build_default_registry(settings)
    llm = FakeLLM(scripted=[_tool_call("calculator", {"expression": "1+1"}, "call_1")])
    runtime = AgentRuntime(llm, settings, registry=registry)
    result = runtime.invoke("loop forever", session_id="s9")

    assert "limite" in result["final_answer"]
    assert any("max iterations" in error for error in result["errors"])


def test_should_continue_routing():
    with_tool_call = AgentState(
        iteration=1,
        max_iterations=5,
        messages=[_tool_call("calculator", {"expression": "1"}, "c")],
    )
    assert should_continue(with_tool_call) == "tools"

    without = AgentState(iteration=1, max_iterations=5, messages=[AIMessage(content="done")])
    assert should_continue(without) == "finish"

    exhausted = AgentState(
        iteration=3,
        max_iterations=3,
        messages=[_tool_call("calculator", {"expression": "1"}, "c")],
    )
    assert should_continue(exhausted) == "finish"

    empty = AgentState(iteration=3, max_iterations=3)
    assert should_continue(empty) == "finish"


def test_should_continue_ignores_tool_messages():
    state = AgentState(
        iteration=1,
        max_iterations=5,
        messages=[
            _tool_call("calculator", {"expression": "1"}, "c"),
        ],
    )
    from langchain_core.messages import ToolMessage

    state["messages"].append(ToolMessage(content="1", tool_call_id="c", name="calculator"))
    assert should_continue(state) == "finish"


# --- event stream ---------------------------------------------------------------------
def test_stream_events_full_scenario(settings):
    settings.ensure_directories()
    registry = build_default_registry(settings)
    llm = FakeLLM(
        scripted=[
            _tool_call("calculator", {"expression": "6 * 7"}, "call_1"),
            AIMessage(content="La réponse est 42."),
        ]
    )
    runtime = AgentRuntime(llm, settings, registry=registry)
    events = list(runtime.stream_events("calcule", session_id="s10"))

    types = [event.type for event in events]
    assert types == [
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.FINAL,
    ]
    assert events[0].tool == "calculator"
    assert "42" in events[1].text
    assert events[2].text == "La réponse est 42."


def test_stream_events_llm_unavailable(settings):
    runtime = AgentRuntime(FakeLLM(available=False), settings)
    events = list(runtime.stream_events("hello", session_id="s11"))
    types = [event.type for event in events]
    assert EventType.ERROR in types
    assert EventType.FINAL in types


# --- graph shape -------------------------------------------------------------------------
def test_graph_compiles(fake_llm, settings):
    ctx = NodeContext(llm=fake_llm, settings=settings)
    graph = build_graph(ctx)
    assert graph is not None


def test_system_prompt_includes_tools_and_identity(settings):
    settings.ensure_directories()
    registry = build_default_registry(settings)
    llm = FakeLLM(scripted=[AIMessage(content="ok")])
    runtime = AgentRuntime(llm, settings, registry=registry)
    runtime.invoke("hello", session_id="s12")

    messages = llm.calls[0]
    assert isinstance(messages[0], SystemMessage)
    prompt = messages[0].content
    assert "Nocki Agent" in prompt
    for name in registry.names:
        assert name in prompt
    assert isinstance(messages[-1], HumanMessage)
