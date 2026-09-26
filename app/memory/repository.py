"""Repositories — the only place that speaks SQL.

Three focused repositories: sessions, messages (conversation turns) and
memory items (short/long-term facts). All return typed Pydantic models.
"""

from __future__ import annotations

from app.memory.database import Database
from app.memory.models import (
    MemoryItem,
    MemoryKind,
    MessageRecord,
    Role,
    SessionInfo,
    from_iso,
    to_iso,
    utc_now,
)


def _session_from_row(row) -> SessionInfo:  # noqa: ANN001 — sqlite3.Row
    return SessionInfo(
        id=row["id"],
        created_at=from_iso(row["created_at"]),
        updated_at=from_iso(row["updated_at"]),
        message_count=row["message_count"] if "message_count" in row.keys() else 0,
    )


class SessionRepository:
    """CRUD for conversation sessions."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def ensure(self, session_id: str) -> SessionInfo:
        """Create the session if missing; return its current state."""
        now = to_iso(utc_now())
        with self._db.cursor() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO sessions (id, created_at, updated_at)"
                " VALUES (?, ?, ?)",
                (session_id, now, now),
            )
            cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cur.fetchone()
        assert row is not None
        return _session_from_row(row)

    def touch(self, session_id: str) -> None:
        with self._db.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (to_iso(utc_now()), session_id),
            )

    def get(self, session_id: str) -> SessionInfo | None:
        with self._db.cursor() as cur:
            cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cur.fetchone()
        return _session_from_row(row) if row else None

    def list(self) -> list[SessionInfo]:
        """All sessions, most recently updated first, with message counts."""
        with self._db.cursor() as cur:
            cur.execute(
                """
                SELECT s.*,
                       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id)
                           AS message_count
                FROM sessions s
                ORDER BY s.updated_at DESC, s.id DESC
                """
            )
            rows = cur.fetchall()
        return [_session_from_row(row) for row in rows]

    def delete(self, session_id: str) -> bool:
        with self._db.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0


class MessageRepository:
    """Conversation turns of a session."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, session_id: str, role: Role, content: str) -> MessageRecord:
        record = MessageRecord(
            session_id=session_id, role=role, content=content, created_at=utc_now()
        )
        with self._db.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (session_id, role, content, created_at)"
                " VALUES (?, ?, ?, ?)",
                (session_id, role.value, content, to_iso(record.created_at)),
            )
            record.id = cur.lastrowid
        return record

    def list(self, session_id: str, limit: int | None = None) -> list[MessageRecord]:
        """Messages in chronological order; when `limit` is given, the LAST `limit`."""
        with self._db.cursor() as cur:
            if limit is None:
                cur.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                )
            else:
                cur.execute(
                    "SELECT * FROM messages WHERE session_id = ?"
                    " ORDER BY id DESC LIMIT ?",
                    (session_id, limit),
                )
            rows = cur.fetchall()
        if limit is not None:
            rows = list(reversed(rows))
        return [
            MessageRecord(
                id=row["id"],
                session_id=row["session_id"],
                role=Role(row["role"]),
                content=row["content"],
                created_at=from_iso(row["created_at"]),
            )
            for row in rows
        ]

    def delete_for_session(self, session_id: str) -> None:
        with self._db.cursor() as cur:
            cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


class MemoryRepository:
    """Short/long-term memory items."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, item: MemoryItem) -> MemoryItem:
        stored = item.model_copy(deep=True)
        with self._db.cursor() as cur:
            cur.execute(
                "INSERT INTO memory (session_id, kind, content, importance, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    stored.session_id,
                    stored.kind.value,
                    stored.content,
                    stored.importance,
                    to_iso(stored.created_at),
                ),
            )
            stored.id = cur.lastrowid
        return stored

    def list(
        self,
        kind: MemoryKind | None = None,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        clauses: list[str] = []
        params: list[object] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        with self._db.cursor() as cur:
            cur.execute(
                f"SELECT * FROM memory {where} ORDER BY id DESC {limit_clause}",  # noqa: S608
                params,
            )
            rows = cur.fetchall()
        return [self._from_row(row) for row in rows]

    def prune_short_term(self, keep: int, session_id: str | None = None) -> int:
        """Keep only the `keep` most recent short-term items. Returns deleted count."""
        if session_id is not None:
            scope = "session_id = ?"
            params: list[object] = [
                MemoryKind.SHORT_TERM.value,
                session_id,
                MemoryKind.SHORT_TERM.value,
                session_id,
                keep,
            ]
        else:
            scope = "session_id IS NULL"
            params = [
                MemoryKind.SHORT_TERM.value,
                MemoryKind.SHORT_TERM.value,
                keep,
            ]
        with self._db.cursor() as cur:
            cur.execute(
                f"""
                DELETE FROM memory
                WHERE kind = ? AND {scope}
                  AND id NOT IN (
                      SELECT id FROM memory WHERE kind = ? AND {scope}
                      ORDER BY id DESC LIMIT ?
                  )
                """,  # noqa: S608
                params,
            )
            return cur.rowcount

    @staticmethod
    def _from_row(row) -> MemoryItem:  # noqa: ANN001 — sqlite3.Row
        return MemoryItem(
            id=row["id"],
            session_id=row["session_id"],
            kind=MemoryKind(row["kind"]),
            content=row["content"],
            importance=row["importance"],
            created_at=from_iso(row["created_at"]),
        )
