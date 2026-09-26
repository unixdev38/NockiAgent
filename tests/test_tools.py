"""Tool layer tests: calculator, filesystem, search, git, python runner, registry."""

from __future__ import annotations

import subprocess

import pytest

from app.security.permissions import Permission
from app.security.validation import WorkspaceValidator
from app.tools.base import ToolContext, ToolResult
from app.tools.calculator import CalculatorTool, SafeEvalError, safe_eval
from app.tools.filesystem import ListDirectoryTool, ReadFileTool
from app.tools.git import GitDiffTool, GitLogTool, GitStatusTool
from app.tools.python_runner import RunPythonTool
from app.tools.registry import ToolRegistry, build_default_registry
from app.tools.search import SearchCodeTool


@pytest.fixture
def ctx(settings):
    settings.ensure_directories()
    validator = WorkspaceValidator(settings.workspace)
    return ToolContext(settings=settings, validator=validator)


# --- calculator ----------------------------------------------------------------
def test_safe_eval_basic():
    assert safe_eval("2 + 2") == 4
    assert safe_eval("2 ** 10") == 1024
    assert safe_eval("7 // 2") == 3
    assert safe_eval("10 % 3") == 1


def test_safe_eval_math_functions():
    assert safe_eval("sqrt(16)") == 4.0
    assert safe_eval("abs(-5)") == 5
    assert safe_eval("max(1, 2, 3)") == 3
    assert safe_eval("round(2.7)") == 3
    assert safe_eval("pi") > 3.14


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "open('/etc/passwd')",
        "().__class__",
        "x = 5",
        "unknown_name",
        "'a' + 'b'",
    ],
)
def test_safe_eval_rejects_dangerous_syntax(expression):
    with pytest.raises(SafeEvalError):
        safe_eval(expression)


def test_calculator_tool(ctx):
    result = CalculatorTool(ctx).run(expression="2 + 2")
    assert result.ok
    assert "4" in result.output


def test_calculator_tool_error(ctx):
    result = CalculatorTool(ctx).run(expression="1 /")
    assert not result.ok
    assert result.error


# --- filesystem ----------------------------------------------------------------
def test_read_file_tool(ctx, settings):
    target = settings.workspace / "notes.txt"
    target.write_text("contenu secret", encoding="utf-8")
    tool = ReadFileTool(ctx)
    result = tool.run(path="notes.txt")
    assert result.ok
    assert result.output == "contenu secret"


def test_read_file_tool_missing(ctx):
    assert not ReadFileTool(ctx).run(path="missing.txt").ok


def test_read_file_tool_traversal(ctx):
    result = ReadFileTool(ctx).run(path="../../etc/passwd")
    assert not result.ok
    assert "workspace" in result.error


def test_read_file_tool_too_large(ctx, settings):
    (settings.workspace / "big.txt").write_text("x" * 100, encoding="utf-8")
    settings.max_file_read_bytes = 10
    result = ReadFileTool(ctx).run(path="big.txt")
    assert not result.ok


def test_list_directory_tool(ctx, settings):
    (settings.workspace / "alpha.txt").write_text("a", encoding="utf-8")
    (settings.workspace / "zeta").mkdir()
    result = ListDirectoryTool(ctx).run(path=".")
    assert result.ok
    assert "alpha.txt" in result.output
    assert "zeta/" in result.output


def test_list_directory_depth(ctx, settings):
    (settings.workspace / "lvl1" / "lvl2").mkdir(parents=True)
    (settings.workspace / "lvl1" / "lvl2" / "deep.txt").write_text("d", encoding="utf-8")
    result = ListDirectoryTool(ctx).run(path=".", depth=3)
    assert result.ok
    assert "deep.txt" in result.output


# --- search -----------------------------------------------------------------------
def _make_project(settings):
    src = settings.workspace / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "a.py").write_text("def nocki_magic():\n    return 1\n", encoding="utf-8")
    (src / "b.py").write_text("# nothing here\nvalue = 42\n", encoding="utf-8")
    (settings.workspace / "binary.bin").write_bytes(b"\x00\x01\x02nocki_magic\xff")
    return src


def test_search_finds_matches(ctx, settings):
    _make_project(settings)
    result = SearchCodeTool(ctx).run(query="nocki_magic")
    assert result.ok
    assert "a.py:1:" in result.output
    assert "binary.bin" not in result.output  # binary files are skipped


def test_search_case_insensitive_by_default(ctx, settings):
    _make_project(settings)
    result = SearchCodeTool(ctx).run(query="NOCKI_MAGIC")
    assert result.ok
    assert "a.py:1:" in result.output


def test_search_no_results(ctx, settings):
    _make_project(settings)
    result = SearchCodeTool(ctx).run(query="zebra_not_found")
    assert result.ok
    assert "Aucun résultat" in result.output


def test_search_respects_max_results(ctx, settings):
    settings.search_max_results = 1
    _make_project(settings)
    (settings.workspace / "src" / "c.py").write_text("nocki_magic\nnocki_magic\n", encoding="utf-8")
    result = SearchCodeTool(ctx).run(query="nocki_magic")
    assert result.ok
    assert result.output.count(":") >= 1
    lines = [line for line in result.output.splitlines() if ":" in line]
    assert len(lines) == 1


# --- git tools ----------------------------------------------------------------------
def _init_repo(settings) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=str(settings.workspace),
            capture_output=True,
            check=True,
        )

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (settings.workspace / "README.md").write_text("# test\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "initial commit")


def test_git_status_tool(ctx, settings):
    _init_repo(settings)
    result = GitStatusTool(ctx).run()
    assert result.ok
    assert "##" in result.output or "main" in result.output or "master" in result.output


def test_git_status_tool_not_a_repo(ctx, settings):
    settings.ensure_directories()
    result = GitStatusTool(ctx).run()
    assert not result.ok
    assert "dépôt git" in result.error


def test_git_log_and_diff_tools(ctx, settings):
    _init_repo(settings)
    log = GitLogTool(ctx).run(limit=5)
    assert log.ok
    assert "initial commit" in log.output

    (settings.workspace / "README.md").write_text("# test modified\n", encoding="utf-8")
    diff = GitDiffTool(ctx).run()
    assert diff.ok
    assert "test modified" in diff.output


# --- python runner ----------------------------------------------------------------------
def test_run_python_disabled_by_default(ctx):
    result = RunPythonTool(ctx).run(code="print('nope')")
    assert not result.ok
    assert "désactivée" in result.error


def test_run_python_enabled(settings, ctx):
    settings.allow_python_exec = True
    settings.require_confirmation = False
    result = RunPythonTool(ctx).run(code="print(6 * 7)")
    assert result.ok
    assert "42" in result.output


def test_run_python_requires_confirmation(settings):
    # CONFIRM is enforced by the tool NODE (graph layer), not the tool itself;
    # here we verify the node path end-to-end.
    from langchain_core.messages import AIMessage

    from app.agent.graph import AgentRuntime
    from tests.conftest import FakeLLM

    settings.ensure_directories()
    settings.allow_python_exec = True
    settings.require_confirmation = True
    registry = build_default_registry(settings)
    llm = FakeLLM(
        scripted=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "run_python", "args": {"code": "print('nope')"}, "id": "c1"}
                ],
            ),
            AIMessage(content="refusé"),
        ]
    )
    runtime = AgentRuntime(llm, settings, registry=registry)
    result = runtime.invoke("exécute", session_id="s-confirm")
    observation = result["observations"][0]
    assert not observation.ok
    assert "confirmation" in observation.error
    assert "nope" not in observation.content


# --- registry -----------------------------------------------------------------------------
def test_default_registry_contains_all_tools(settings):
    settings.ensure_directories()
    registry = build_default_registry(settings)
    expected = {
        "calculator",
        "read_file",
        "list_directory",
        "search_code",
        "git_status",
        "git_diff",
        "git_log",
        "run_python",
    }
    assert set(registry.names) == expected


def test_registry_run_validates_arguments(settings):
    settings.ensure_directories()
    registry = build_default_registry(settings)
    result = registry.run("calculator", {"wrong_field": True})
    assert isinstance(result, ToolResult)
    assert not result.ok
    assert "Arguments invalides" in result.error


def test_registry_run_unknown_tool(settings):
    settings.ensure_directories()
    registry = build_default_registry(settings)
    result = registry.run("does_not_exist", {})
    assert not result.ok
    assert "Outil inconnu" in result.error


def test_registry_never_raises(settings):
    settings.ensure_directories()
    registry = build_default_registry(settings)
    # Even nonsense input yields a ToolResult, never an exception.
    result = registry.run("read_file", {"path": 12345})
    assert isinstance(result, ToolResult)


def test_registry_as_langchain(settings):
    settings.ensure_directories()
    registry = build_default_registry(settings)
    tools = registry.as_langchain()
    assert len(tools) == len(registry)
    assert registry.as_langchain() is tools  # cached
    by_name = {tool.name: tool for tool in tools}
    assert by_name["calculator"].invoke({"expression": "1 + 1"}) == "2"


def test_registry_rejects_duplicates(ctx):
    registry = ToolRegistry([ReadFileTool(ctx)])
    assert len(registry) == 1
    assert "read_file" in registry
    with pytest.raises(ValueError, match="en double"):
        ToolRegistry([ReadFileTool(ctx), ReadFileTool(ctx)])


def test_tool_permissions_declared(settings):
    settings.ensure_directories()
    registry = build_default_registry(settings)
    execute_tools = [
        tool.name for tool in registry if tool.permission is Permission.EXECUTE
    ]
    assert execute_tools == ["run_python"]
