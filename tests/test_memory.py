"""Memory layer tests: database, repositories, manager, persistence, cascades."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.memory.database import Database
from app.memory.manager import MemoryManager
from app.memory.models import MemoryItem, MemoryKind, Role, SessionInfo, from_iso, to_iso
from app.memory.repository import (
    MemoryRepository,
    MessageRepository,
    SessionRepository,
)


# --- database -------------------------------------------------------------------
def test_database_in_memory():
    db = Database(":memory:")
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM sessions")
        assert cur.fetchone()[0] == 0
    db.close()


def test_database_file_created(tmp_path):
    path = tmp_path / "data" / "nocki.db"
    db = Database(path)
    db.close()
    assert path.exists()


def test_database_rollback_on_error():
    db = Database(":memory:")
    sessions = SessionRepository(db)
    sessions.ensure("s1")
    try:
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (id, created_at, updated_at) VALUES ('s2', 'x', 'y')"
            )
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert sessions.get("s2") is None  # rolled back


# --- models ---------------------------------------------------------------------
def test_timestamps_are_timezone_aware():
    iso = to_iso(datetime.now(UTC))
    parsed = from_iso(iso)
    assert parsed.tzinfo is not None
    assert abs(parsed - datetime.now(UTC)) < timedelta(seconds=5)


def test_memory_item_validation():
    item = MemoryItem(content="fact", importance=5.0)
    assert item.kind is MemoryKind.LONG_TERM
    assert item.session_id is None


# --- session repository ---------------------------------------------------------
def test_session_ensure_is_idempotent():
    db = Database(":memory:")
    sessions = SessionRepository(db)
    first = sessions.ensure("s1")
    second = sessions.ensure("s1")
    assert first.id == second.id == "s1"


def test_session_list_counts_messages():
    db = Database(":memory:")
    sessions = SessionRepository(db)
    messages = MessageRepository(db)
    sessions.ensure("a")
    sessions.ensure("b")
    messages.add("a", Role.HUMAN, "hi")
    messages.add("a", Role.AI, "hello")
    sessions.touch("a")

    listed = sessions.list()
    assert [info.id for info in listed] == ["a", "b"]  # most recent first
    by_id = {info.id: info.message_count for info in listed}
    assert by_id["a"] == 2
    assert by_id["b"] == 0


def test_session_delete_cascades():
    db = Database(":memory:")
    sessions = SessionRepository(db)
    messages = MessageRepository(db)
    memories = MemoryRepository(db)
    sessions.ensure("s1")
    messages.add("s1", Role.HUMAN, "hi")
    memories.add(MemoryItem(session_id="s1", kind=MemoryKind.SHORT_TERM, content="note"))

    assert sessions.delete("s1") is True
    assert sessions.delete("s1") is False  # already gone
    assert messages.list("s1") == []
    assert memories.list(session_id="s1") == []


# --- message repository ------------------------------------------------------------
def test_message_window_returns_last_n_chronological():
    db = Database(":memory:")
    sessions = SessionRepository(db)
    messages = MessageRepository(db)
    sessions.ensure("s")
    for i in range(10):
        messages.add("s", Role.HUMAN, f"m{i}")

    window = messages.list("s", limit=3)
    assert [m.content for m in window] == ["m7", "m8", "m9"]

    full = messages.list("s")
    assert len(full) == 10
    assert full[0].content == "m0"


# --- memory repository ----------------------------------------------------------------
def test_memory_prune_short_term_keeps_recent():
    db = Database(":memory:")
    sessions = SessionRepository(db)
    memories = MemoryRepository(db)
    sessions.ensure("s")
    for i in range(5):
        memories.add(
            MemoryItem(session_id="s", kind=MemoryKind.SHORT_TERM, content=f"n{i}")
        )

    deleted = memories.prune_short_term(keep=2, session_id="s")
    assert deleted == 3
    remaining = memories.list(kind=MemoryKind.SHORT_TERM, session_id="s")
    assert [item.content for item in remaining] == ["n4", "n3"]  # newest kept


def test_memory_list_filters():
    db = Database(":memory:")
    sessions = SessionRepository(db)
    memories = MemoryRepository(db)
    sessions.ensure("s")
    memories.add(MemoryItem(kind=MemoryKind.LONG_TERM, content="global"))
    memories.add(MemoryItem(session_id="s", kind=MemoryKind.SHORT_TERM, content="local"))

    assert len(memories.list()) == 2
    assert len(memories.list(kind=MemoryKind.LONG_TERM)) == 1
    assert memories.list(kind=MemoryKind.LONG_TERM)[0].content == "global"


# --- manager -------------------------------------------------------------------------
def test_manager_conversation_roundtrip():
    db = Database(":memory:")
    manager = MemoryManager(db, history_window=10)
    manager.ensure_session("s1")

    manager.append_turn("s1", "Bonjour", "Salut !")
    manager.append_turn("s1", "Ça va ?", "Très bien.")

    conversation = manager.load_conversation("s1")
    assert [type(m).__name__ for m in conversation] == [
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
        "AIMessage",
    ]
    assert conversation[0].content == "Bonjour"
    assert conversation[3].content == "Très bien."


def test_manager_respects_history_window():
    db = Database(":memory:")
    manager = MemoryManager(db, history_window=4)
    for i in range(10):
        manager.append_turn("s", f"q{i}", f"a{i}")
    assert len(manager.load_conversation("s")) == 4  # 2 turns = 4 messages


def test_manager_remember_prunes_short_term():
    db = Database(":memory:")
    manager = MemoryManager(db, short_term_keep=3)
    for i in range(6):
        manager.remember(f"note {i}", kind=MemoryKind.SHORT_TERM, session_id="s")
    items = manager.list_memory(kind=MemoryKind.SHORT_TERM)
    assert len(items) == 3
    assert items[0].content == "note 5"  # newest first


def test_manager_long_term_not_pruned():
    db = Database(":memory:")
    manager = MemoryManager(db, short_term_keep=2)
    for i in range(6):
        manager.remember(f"fact {i}", kind=MemoryKind.LONG_TERM)
    assert len(manager.list_memory(kind=MemoryKind.LONG_TERM)) == 6


def test_manager_relevant_context_scores_and_limits():
    db = Database(":memory:")
    manager = MemoryManager(db)
    manager.remember("L'utilisateur préfère Python pour l'analyse de code", importance=5.0)
    manager.remember("Le chat s'appelle Mistigri", importance=1.0)
    manager.remember("Fact totalement sans rapport avec la question", importance=9.0)

    items = manager.relevant_context("Quel langage pour analyser du code ?", max_items=2)
    contents = [item.content for item in items]
    assert "L'utilisateur préfère Python pour l'analyse de code" in contents
    assert "Le chat s'appelle Mistigri" not in contents  # no keyword overlap
    assert len(items) <= 2


def test_manager_never_sends_everything():
    db = Database(":memory:")
    manager = MemoryManager(db)
    for i in range(20):
        manager.remember(f"python astuce numéro {i}", importance=1.0)
    items = manager.relevant_context("python", max_items=5)
    assert len(items) == 5


def test_manager_short_term_scoped_to_session():
    db = Database(":memory:")
    manager = MemoryManager(db)
    manager.remember("note de session A", kind=MemoryKind.SHORT_TERM, session_id="a")
    manager.remember("note long terme", kind=MemoryKind.LONG_TERM)

    from_a = manager.relevant_context("note", session_id="a")
    from_b = manager.relevant_context("note", session_id="b")
    assert any("session A" in item.content for item in from_a)
    assert not any("session A" in item.content for item in from_b)


def test_manager_build_context_block():
    db = Database(":memory:")
    manager = MemoryManager(db)
    assert manager.build_context_block("n'importe quoi") == ""

    manager.remember("L'utilisateur aime les réponses concises", importance=3.0)
    block = manager.build_context_block("réponses courtes ?", session_id="s")
    assert "Mémoire pertinente" in block
    assert "réponses concises" in block


def test_manager_sessions_crud():
    db = Database(":memory:")
    manager = MemoryManager(db)
    manager.append_turn("s1", "q", "a")

    assert isinstance(manager.get_session("s1"), SessionInfo)
    assert manager.get_session("missing") is None
    assert [info.id for info in manager.list_sessions()] == ["s1"]
    assert manager.delete_session("s1") is True
    assert manager.list_sessions() == []


# --- persistence after reopen -----------------------------------------------------
def test_persistence_across_reopen(tmp_path):
    path = tmp_path / "nocki.db"
    db = Database(path)
    manager = MemoryManager(db)
    manager.append_turn("s1", "question", "réponse")
    manager.remember("fait important", kind=MemoryKind.LONG_TERM, importance=4.0)
    db.close()

    reopened = Database(path)
    manager2 = MemoryManager(reopened)
    conversation = manager2.load_conversation("s1")
    assert [m.content for m in conversation] == ["question", "réponse"]
    assert manager2.list_memory(kind=MemoryKind.LONG_TERM)[0].content == "fait important"
    reopened.close()


# --- runtime + memory integration ---------------------------------------------------
def test_runtime_persists_and_rehydrates(settings):
    from app.agent.graph import AgentRuntime
    from tests.conftest import FakeLLM

    settings.ensure_directories()
    db = Database(settings.database_path)
    memory = MemoryManager(db)
    llm = FakeLLM(reply="première réponse")
    runtime = AgentRuntime(llm, settings, memory=memory)

    runtime.invoke("Bonjour Nocki", session_id="chat1")
    stored = memory.load_conversation("chat1")
    assert [m.content for m in stored] == ["Bonjour Nocki", "première réponse"]

    # A NEW runtime on the same database rehydrates the conversation.
    llm2 = FakeLLM(reply="deuxième réponse")
    runtime2 = AgentRuntime(llm2, settings, memory=MemoryManager(Database(settings.database_path)))
    runtime2.invoke("Suite de la discussion", session_id="chat1")

    history = llm2.calls[0]
    human_messages = [m for m in history if isinstance(m, HumanMessage)]
    assert len(human_messages) == 2  # prior turn + new question
    assert human_messages[0].content == "Bonjour Nocki"
    assert human_messages[1].content == "Suite de la discussion"
    assert any(isinstance(m, AIMessage) and m.content == "première réponse" for m in history)


def test_runtime_stream_events_persists(settings):
    from app.agent.graph import AgentRuntime
    from tests.conftest import FakeLLM

    settings.ensure_directories()
    memory = MemoryManager(Database(settings.database_path))
    runtime = AgentRuntime(FakeLLM(reply="streamed"), settings, memory=memory)

    events = list(runtime.stream_events("question", session_id="chat2"))
    assert events  # sanity
    stored = memory.load_conversation("chat2")
    assert [m.content for m in stored] == ["question", "streamed"]


def test_runtime_memory_context_block_injected(settings):
    from app.agent.graph import AgentRuntime
    from tests.conftest import FakeLLM

    settings.ensure_directories()
    memory = MemoryManager(Database(settings.database_path))
    memory.remember("L'utilisateur travaille sur le projet Nocki", importance=5.0)

    llm = FakeLLM(reply="ok")
    runtime = AgentRuntime(llm, settings, memory=memory)
    runtime.invoke("parle-moi du projet", session_id="chat3")

    system_messages = [
        m for m in llm.calls[0] if isinstance(m, SystemMessage)
    ]
    # First system message is the prompt; a context block may follow it.
    assert system_messages, "expected at least the main system prompt"
    context_blocks = [m for m in system_messages if "Contexte pertinent" in m.content]
    assert context_blocks and "projet Nocki" in context_blocks[0].content
