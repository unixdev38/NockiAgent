from __future__ import annotations

from app.agent.state import Observation, Plan, ToolCall, new_state


def test_new_state_defaults():
    st = new_state("do something", max_iterations=8)
    assert st["task"] == "do something"
    assert st["iteration"] == 0
    assert st["max_iterations"] == 8
    assert st["tool_calls"] == []
    assert st["observations"] == []
    assert st["final_answer"] is None


def test_tool_call_model():
    tc = ToolCall(name="calculator", arguments={"expr": "2+2"})
    assert tc.name == "calculator"
    assert tc.arguments["expr"] == "2+2"


def test_observation_model():
    obs = Observation(tool="calculator", ok=True, content="4")
    assert obs.ok
    assert obs.content == "4"


def test_plan_model():
    plan = Plan(steps=["a", "b"], rationale="because")
    assert plan.steps == ["a", "b"]
