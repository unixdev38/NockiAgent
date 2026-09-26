"""MemoryManager — the façade the rest of the app uses for memory.

Design rules:
- The manager never sends the *whole* memory to the LLM: `relevant_context()`
  scores items (keyword overlap × importance) and returns only the top few.
- Short-term memory is session-scoped and pruned to the last N items;
  long-term memory is kept.
- `build_context_block()` produces the SystemMessage injected before the
  user's question (relevant memories + RAG chunks when a retriever is set).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.logging_config import get_logger
from app.memory.database import Database
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

if TYPE_CHECKING:  # Phase 6 module — kept out of Phase 3 runtime imports.
    from app.rag.base import Retriever

logger = get_logger("nocki.memory")

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


class MemoryManager:
    """Persistent conversation memory + relevant-context selection."""

    def __init__(
        self,
        database: Database,
        history_window: int = 20,
        short_term_keep: int = 20,
        retriever: Retriever | None = None,
    ) -> None:
        self.database = database
        self.sessions = SessionRepository(database)
        self.messages = MessageRepository(database)
        self.memories = MemoryRepository(database)
        self.history_window = history_window
        self.short_term_keep = short_term_keep
        self.retriever = retriever

    # -- sessions --------------------------------------------------------------
    def ensure_session(self, session_id: str) -> SessionInfo:
        return self.sessions.ensure(session_id)

    def list_sessions(self) -> list[SessionInfo]:
        return self.sessions.list()

    def get_session(self, session_id: str) -> SessionInfo | None:
        return self.sessions.get(session_id)

    def get_messages(self, session_id: str) -> list[MessageRecord]:
        return self.messages.list(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self.sessions.delete(session_id)

    # -- conversation ------------------------------------------------------------
    def load_conversation(self, session_id: str) -> list[BaseMessage]:
        """Return the last `history_window` turns as Human/AI messages."""
        records = self.messages.list(session_id, limit=self.history_window)
        conversation: list[BaseMessage] = []
        for record in records:
            if record.role is Role.HUMAN:
                conversation.append(HumanMessage(content=record.content))
            elif record.role is Role.AI:
                conversation.append(AIMessage(content=record.content))
        return conversation

    def append_turn(self, session_id: str, user: str, assistant: str) -> None:
        """Persist one completed turn (user question + assistant answer)."""
        self.ensure_session(session_id)
        self.messages.add(session_id, Role.HUMAN, user)
        self.messages.add(session_id, Role.AI, assistant)
        self.sessions.touch(session_id)

    # -- memory items ---------------------------------------------------------------
    def remember(
        self,
        content: str,
        kind: MemoryKind = MemoryKind.LONG_TERM,
        importance: float = 1.0,
        session_id: str | None = None,
    ) -> MemoryItem:
        """Store a fact; short-term items are pruned to `short_term_keep`."""
        if session_id is not None:
            self.ensure_session(session_id)  # FK: memory rows reference sessions
        item = self.memories.add(
            MemoryItem(
                session_id=session_id if kind is MemoryKind.SHORT_TERM else None,
                kind=kind,
                content=content,
                importance=importance,
            )
        )
        if kind is MemoryKind.SHORT_TERM:
            deleted = self.memories.prune_short_term(self.short_term_keep, session_id)
            if deleted:
                logger.debug("Pruned %d short-term memory items", deleted)
        return item

    def list_memory(
        self,
        kind: MemoryKind | None = None,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        return self.memories.list(kind=kind, session_id=session_id, limit=limit)

    # -- context selection --------------------------------------------------------------
    def relevant_context(
        self, query: str, max_items: int = 5, session_id: str | None = None
    ) -> list[MemoryItem]:
        """Score memory items by keyword overlap × importance; top `max_items`.

        Long-term memory is always considered; short-term (session-scoped)
        items only when `session_id` matches. Unrelated items are never sent.
        """
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        candidates = self.memories.list(limit=1000)
        scored: list[tuple[float, MemoryItem]] = []
        for item in candidates:
            if item.kind is MemoryKind.SHORT_TERM:
                if session_id is None or item.session_id != session_id:
                    continue
            overlap = len(query_tokens & _tokens(item.content)) / len(query_tokens)
            if overlap <= 0:
                continue
            scored.append((overlap * item.importance, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:max_items]]

    def build_context_block(
        self, query: str, max_items: int = 5, session_id: str | None = None
    ) -> str:
        """Render the SystemMessage context block (memory + RAG chunks)."""
        sections: list[str] = []

        memories = self.relevant_context(query, max_items=max_items, session_id=session_id)
        if memories:
            lines = [f"- [{item.kind.value}] {item.content}" for item in memories]
            sections.append("Mémoire pertinente:\n" + "\n".join(lines))

        if self.retriever is not None:
            try:
                chunks = self.retriever.retrieve(query, k=3)
            except Exception as exc:  # noqa: BLE001 — retrieval is optional context
                logger.warning("Retriever failed: %s", exc)
                chunks = []
            if chunks:
                lines = [f"- {chunk.document.content.strip()[:500]}" for chunk in chunks]
                sections.append("Contexte documentaire:\n" + "\n".join(lines))

        if not sections:
            return ""
        return "Contexte pertinent pour la question:\n\n" + "\n\n".join(sections)
