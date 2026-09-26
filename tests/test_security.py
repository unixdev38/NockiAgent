"""Security layer tests: workspace validation, size limits, permissions."""

from __future__ import annotations

import os
import sys

import pytest

from app.config import Settings
from app.security.permissions import Decision, Permission, PermissionManager
from app.security.sandbox import run_python_code
from app.security.validation import (
    FileTooLargeError,
    PathTraversalError,
    WorkspaceValidator,
)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "a.txt").write_text("hello", encoding="utf-8")
    (ws / "sub").mkdir()
    (ws / "sub" / "b.txt").write_text("world", encoding="utf-8")
    return ws


@pytest.fixture
def validator(workspace):
    return WorkspaceValidator(workspace)


# --- validation ---------------------------------------------------------------
def test_resolve_relative_inside(workspace, validator):
    resolved = validator.resolve("a.txt")
    assert resolved == (workspace / "a.txt").resolve()


def test_resolve_dotdot_escape_rejected(workspace, validator):
    with pytest.raises(PathTraversalError):
        validator.resolve("../outside.txt")


def test_resolve_absolute_outside_rejected(tmp_path, validator):
    with pytest.raises(PathTraversalError):
        validator.resolve(str(tmp_path / "elsewhere" / "x.txt"))


def test_resolve_absolute_inside_accepted(workspace, validator):
    resolved = validator.resolve(str(workspace / "a.txt"))
    assert resolved.name == "a.txt"


def test_resolve_workspace_root_itself_accepted(workspace, validator):
    assert validator.resolve(".") == workspace


@pytest.mark.skipif(os.name == "nt", reason="symlink semantics differ on Windows")
def test_resolve_symlink_escape_rejected(tmp_path, workspace, validator):
    target = tmp_path / "outside.txt"
    target.write_text("secret", encoding="utf-8")
    link = workspace / "link.txt"
    link.symlink_to(target)
    with pytest.raises(PathTraversalError):
        validator.resolve("link.txt")


def test_read_text_ok(validator):
    assert validator.read_text("a.txt", max_bytes=100) == "hello"


def test_read_text_missing(validator):
    with pytest.raises(FileNotFoundError):
        validator.read_text("nope.txt", max_bytes=100)


def test_read_text_directory(validator):
    with pytest.raises(IsADirectoryError):
        validator.read_text("sub", max_bytes=100)


def test_read_text_too_large(validator):
    with pytest.raises(FileTooLargeError):
        validator.read_text("a.txt", max_bytes=2)


def test_relative_display(validator):
    assert validator.relative("sub/b.txt") == os.path.join("sub", "b.txt")


# --- permissions -----------------------------------------------------------------
def test_read_always_allowed():
    manager = PermissionManager(Settings(_env_file=None))
    result = manager.check(Permission.READ)
    assert result.allowed and not result.needs_confirmation and not result.denied


def test_write_denied_by_default():
    manager = PermissionManager(Settings(_env_file=None))
    result = manager.check(Permission.WRITE)
    assert result.denied
    assert "ALLOW_FILE_WRITE" in result.reason


def test_write_confirm_then_allow():
    confirm = PermissionManager(
        Settings(_env_file=None, allow_file_write=True, require_confirmation=True)
    )
    assert confirm.check(Permission.WRITE).needs_confirmation
    allow = PermissionManager(
        Settings(_env_file=None, allow_file_write=True, require_confirmation=False)
    )
    assert allow.check(Permission.WRITE).allowed


def test_execute_denied_by_default():
    manager = PermissionManager(Settings(_env_file=None))
    result = manager.check(Permission.EXECUTE)
    assert result.denied


def test_execute_confirm_when_shell_allowed():
    manager = PermissionManager(
        Settings(_env_file=None, allow_shell=True, require_confirmation=True)
    )
    assert manager.check(Permission.EXECUTE).needs_confirmation


def test_execute_allowed_when_python_exec_and_no_confirmation():
    manager = PermissionManager(
        Settings(_env_file=None, allow_python_exec=True, require_confirmation=False)
    )
    assert manager.check(Permission.EXECUTE).decision is Decision.ALLOW


# --- sandbox ---------------------------------------------------------------------
def test_sandbox_runs_code_and_captures_output(tmp_path):
    result = run_python_code("print(21 * 2)", workspace=tmp_path, timeout=10)
    assert result.ok
    assert result.stdout.strip() == "42"
    assert not result.timed_out


def test_sandbox_timeout(tmp_path):
    result = run_python_code("import time\ntime.sleep(30)", workspace=tmp_path, timeout=1)
    assert not result.ok
    assert result.timed_out


def test_sandbox_reports_errors(tmp_path):
    result = run_python_code("raise SystemExit(3)", workspace=tmp_path, timeout=10)
    assert not result.ok
    assert result.returncode == 3


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits")
def test_sandbox_memory_limit(tmp_path):
    code = "x = 'a' * (512 * 1024 * 1024)\nprint('allocated')"
    result = run_python_code(code, workspace=tmp_path, timeout=10)
    assert not result.ok  # rlimit RLIMIT_AS must kill the allocation
    assert "allocated" not in result.stdout
