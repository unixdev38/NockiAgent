"""Nocki Agent — a fully local, modular and secure AI agent.

Package layout (grows across phases):

    app.config      Centralized, env-driven configuration (Pydantic Settings).
    app.llm         LLM providers behind a common interface (Ollama today).
    app.agent       LangGraph state, prompts, nodes and workflow.
    app.tools       Extensible, LLM-independent tools (Phase 2+).
    app.memory      Persistent memory over SQLite (Phase 3+).
    app.security    Permissions, sandboxing and validation (Phase 2+).
    app.main        CLI entrypoint.
"""

__version__ = "0.1.0"
