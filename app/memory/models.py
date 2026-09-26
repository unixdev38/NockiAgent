"""Typed models for the memory layer.

All timestamps are timezone-aware (`datetime.now(UTC)`); `datetime.utcnow()`
is deprecated and never used.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Current timezone-aware UTC time."""
    return datetime.now(UTC)


def to_iso(moment: datetime) -> str:
    """Normalize a datetime to a UTC ISO-8601 string for storage."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat()


def from_iso(value: str) -> datetime:
    """Parse an ISO-8601 string back to a timezone-aware datetime."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


class Role(StrEnum):
    """Author of a stored message."""

    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"
    TOOL = "tool"


class MemoryKind(StrEnum):
    """Lifetime category of a memory item."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class SessionInfo(BaseModel):
    """A conversation session (no content — see MessageRecord)."""

    id: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class MessageRecord(BaseModel):
    """One stored message of a session."""

    id: int | None = None
    session_id: str
    role: Role
    content: str
    created_at: datetime = Field(default_factory=utc_now)


class MemoryItem(BaseModel):
    """A remembered fact, either short-term (session-scoped, pruned) or long-term."""

    id: int | None = None
    session_id: str | None = None
    kind: MemoryKind = MemoryKind.LONG_TERM
    content: str
    importance: float = Field(default=1.0, ge=0.0, le=10.0)
    created_at: datetime = Field(default_factory=utc_now)
