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
