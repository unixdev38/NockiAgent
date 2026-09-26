"""Persistent memory over SQLite (Phase 3).

Layers: `Database` (connection + schema) → repositories (the only SQL) →
`MemoryManager` (the façade used by the runtime/CLI/API). Sessions store full
conversations; the `memory` table stores short-term (session-scoped, pruned)
and long-term facts; only *relevant* context is ever sent to the LLM.
"""

from app.memory.database import Database
from app.memory.manager import MemoryManager
from app.memory.models import (
    MemoryItem,
    MemoryKind,
    MessageRecord,
    Role,
    SessionInfo,
)
from app.memory.repository import (
    MemoryRepository,
    MessageRepository,
    SessionRepository,
)

__all__ = [
    "Database",
    "MemoryItem",
    "MemoryKind",
    "MemoryManager",
    "MemoryRepository",
    "MessageRecord",
    "MessageRepository",
    "Role",
    "SessionInfo",
    "SessionRepository",
]
