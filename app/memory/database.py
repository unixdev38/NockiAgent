"""SQLite access layer.

A tiny, dependency-free data-access layer: one `Database` class managing the
connection (thread-safe via a lock, `row_factory=Row`, `PRAGMA foreign_keys=ON`)
and the schema. Repositories are the only place SQL is written.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    content     TEXT NOT NULL,
    importance  REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory(kind);
CREATE INDEX IF NOT EXISTS idx_memory_session ON memory(session_id);
"""


class Database:
    """Thread-safe wrapper around a sqlite3 connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path == ":memory:":
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self.cursor() as cur:
            cur.executescript(_SCHEMA)

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        """Yield a cursor inside a transaction (commit on success, rollback on error)."""
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
            except Exception:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
