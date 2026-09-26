from __future__ import annotations

from pathlib import Path

from app.config import LogLevel, Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.ollama_model  # non-empty default
    assert s.max_iterations > 0
    assert s.agent_timeout > 0
    assert isinstance(s.workspace_path, Path)
    assert s.workspace_path.is_absolute()


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3")
    monkeypatch.setenv("MAX_ITERATIONS", "5")
    s = Settings(_env_file=None)
    assert s.ollama_model == "qwen3"
    assert s.max_iterations == 5


def test_log_level_enum():
    s = Settings(_env_file=None, log_level="DEBUG")
    assert s.log_level is LogLevel.DEBUG


def test_ensure_directories(tmp_path):
    s = Settings(
        _env_file=None,
        workspace_path=tmp_path / "ws",
        database_path=tmp_path / "d" / "db.sqlite",
    )
    s.ensure_directories()
    assert (tmp_path / "ws").exists()
    assert (tmp_path / "d").exists()


def test_security_defaults_are_safe():
    s = Settings(_env_file=None)
    assert s.allow_shell is False
    assert s.allow_file_write is False
    assert s.allow_python_exec is False
    assert s.require_confirmation is True


def test_tool_limits_defaults():
    s = Settings(_env_file=None)
    assert s.max_file_read_bytes == 200_000
    assert s.search_max_results == 50
    assert s.search_max_files == 2_000
    assert s.python_exec_timeout == 10


def test_security_env_overrides(monkeypatch):
    monkeypatch.setenv("ALLOW_PYTHON_EXEC", "true")
    monkeypatch.setenv("PYTHON_EXEC_TIMEOUT", "30")
    monkeypatch.setenv("MAX_FILE_READ_BYTES", "1000")
    s = Settings(_env_file=None)
    assert s.allow_python_exec is True
    assert s.python_exec_timeout == 30
    assert s.max_file_read_bytes == 1000


def test_invalid_values_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, python_exec_timeout=-1)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_iterations=0)
