"""Centralized configuration for Nocki Agent.

All tunables live here and are loaded from environment variables / a `.env`
file (see `.env.example`). Nothing about the model, workspace or security
policy is hardcoded elsewhere in the code base — that is a hard project rule.

Usage::

    from app.config import get_settings
    settings = get_settings()
    print(settings.ollama_model)
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Values come from (in priority order): explicit init kwargs, environment
    variables, then the `.env` file. This keeps configuration declarative and
    makes the LLM/model/workspace fully interchangeable without code changes.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM / Ollama -------------------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server.",
    )
    ollama_model: str = Field(
        default="gemma3:1b",
        description="Ollama model name. Interchangeable (gemma3:1b, qwen3, llama3, ...).",
    )
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_request_timeout: int = Field(default=120, gt=0)

    # --- Agent workflow -----------------------------------------------------
    max_iterations: int = Field(default=8, gt=0, le=100)
    agent_timeout: int = Field(default=120, gt=0)

    # --- Workspace ----------------------------------------------------------
    workspace_path: Path = Field(default=Path("./workspace"))

    # --- Security -----------------------------------------------------------
    allow_shell: bool = Field(default=False)
    allow_file_write: bool = Field(default=False)
    allow_python_exec: bool = Field(default=False)
    require_confirmation: bool = Field(default=True)

    # --- Tools --------------------------------------------------------------
    max_file_read_bytes: int = Field(default=200_000, gt=0)
    search_max_results: int = Field(default=50, gt=0)
    search_max_files: int = Field(default=2_000, gt=0)
    python_exec_timeout: int = Field(default=10, gt=0, le=600)

    # --- Memory -------------------------------------------------------------
    database_path: Path = Field(default=Path("./data/nocki.db"))

    # --- Observability ------------------------------------------------------
    log_level: LogLevel = Field(default=LogLevel.INFO)

    @field_validator("workspace_path", "database_path", mode="after")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        # Expand ~ and make absolute so all downstream checks are unambiguous.
        return value.expanduser().resolve()

    @property
    def workspace(self) -> Path:
        """Absolute, resolved workspace directory."""
        return self.workspace_path

    def ensure_directories(self) -> None:
        """Create workspace and data directories if they do not exist."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (single source of truth)."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and reload settings (useful for tests / CLI /model)."""
    get_settings.cache_clear()
    return get_settings()
