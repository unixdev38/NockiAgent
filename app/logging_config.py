"""Structured, safe logging setup.

We never log secrets (tokens, passwords, API keys). Log records describe the
agent lifecycle (task received, node entered, iteration, completion) so runs
are observable without leaking sensitive data.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, with a Rich handler for readable output."""
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
    )
    # Quiet down noisy third-party libraries.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
