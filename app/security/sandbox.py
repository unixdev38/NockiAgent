"""Best-effort sandbox for running LLM-requested Python code.

Honest security posture (documented on purpose):
- The code runs in a *separate* interpreter (`python -I`, isolated mode) with
  its own process, a wall-clock timeout, a restricted current directory (the
  workspace) and — on POSIX — rlimits (CPU seconds, address space, file size).
- On Windows the `resource` module does not exist, so only the process
  isolation and the timeout apply.
- This is a *containment convenience*, NOT an adversary-proof sandbox: a
  determined program can still attempt network calls or fork. Real security
  comes from the permission layer (EXECUTE is denied by default) and from the
  workspace validator. Never enable execution for a model you do not trust.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

#: Default address-space limit for sandboxed runs (256 MiB).
DEFAULT_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
#: Max size of any single file written by sandboxed code (16 MiB).
DEFAULT_FILE_SIZE_LIMIT_BYTES = 16 * 1024 * 1024


class SandboxResult(BaseModel):
    """Outcome of a sandboxed execution."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False


def _posix_limits(
    memory_bytes: int,
    file_size_bytes: int,
    cpu_seconds: int,
):  # noqa: ANN202 — closure signature for preexec_fn
    import resource

    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_bytes, file_size_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

    return _apply


def run_python_code(
    code: str,
    workspace: Path,
    timeout: int = 10,
    memory_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
) -> SandboxResult:
    """Run `code` in an isolated subprocess rooted at `workspace`."""
    cmd = [sys.executable, "-I", "-c", code]

    preexec_fn = None
    if sys.platform != "win32":
        try:
            preexec_fn = _posix_limits(
                memory_bytes=memory_bytes,
                file_size_bytes=DEFAULT_FILE_SIZE_LIMIT_BYTES,
                cpu_seconds=max(int(timeout), 1),
            )
        except Exception:  # noqa: BLE001 — rlimits are best effort
            preexec_fn = None

    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, isolated interpreter
            cmd,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=preexec_fn,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        def _text(value: object) -> str:
            if value is None:
                return ""
            return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)

        return SandboxResult(
            ok=False,
            stdout=_text(exc.stdout),
            stderr=_text(exc.stderr),
            returncode=None,
            timed_out=True,
        )

    return SandboxResult(
        ok=proc.returncode == 0,
        stdout=proc.stdout,
        stderr=proc.stderr,
        returncode=proc.returncode,
        timed_out=False,
    )
