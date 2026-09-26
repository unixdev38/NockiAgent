"""RAG tests: chunking, keyword retriever, memory integration."""

from __future__ import annotations

import pytest

from app.memory.database import Database
from app.memory.manager import MemoryManager
from app.rag.base import (
    Chunker,
    Document,
    EmbeddingProvider,
    RetrievedChunk,
    Retriever,
)
from app.rag.chunking import SimpleChunker
from app.rag.keyword import KeywordRetriever, tokenize


# --- chunking --------------------------------------------------------------------
def test_chunker_overlapping_windows():
    chunker = SimpleChunker(chunk_size=100, overlap=20)
    text = "x" * 250
    chunks = chunker.split(text, metadata={"id": "doc1"})
    assert len(chunks) == 3  # [0:100], [80:180], [160:250]
    assert all(len(chunk.content) <= 100 for chunk in chunks)
    # Overlap: chunk 1 starts where the previous window advanced (step=80).
    assert chunks[1].metadata["start"] == 80
    assert chunks[0].content[20:] == chunks[1].content[:80]


def test_chunker_short_text_single_chunk():
    chunker = SimpleChunker()
    chunks = chunker.split("court texte")
    assert len(chunks) == 1
    assert chunks[0].content == "court texte"


def test_chunker_empty_text():
    assert SimpleChunker().split("") == []
    assert SimpleChunker().split("   \n  ") == []


def test_chunker_validates_params():
    with pytest.raises(ValueError, match="strictement inférieur"):
        SimpleChunker(chunk_size=100, overlap=100)
    with pytest.raises(ValueError, match="strictement inférieur"):
        SimpleChunker(chunk_size=100, overlap=150)
    with pytest.raises(ValueError):
        SimpleChunker(chunk_size=0)
    with pytest.raises(ValueError):
        SimpleChunker(chunk_size=100, overlap=-1)


def test_chunker_propagates_metadata_and_ids():
    chunker = SimpleChunker(chunk_size=10, overlap=2)
    chunks = chunker.split("abcdefghij" * 3, metadata={"id": "readme", "path": "README.md"})
    assert len(chunks) > 1
    assert chunks[0].id.startswith("readme::chunk-")
    assert all(chunk.metadata["path"] == "README.md" for chunk in chunks)
    assert [chunk.metadata["index"] for chunk in chunks] == list(range(len(chunks)))


# --- keyword retriever ---------------------------------------------------------------
def _index_corpus() -> KeywordRetriever:
    retriever = KeywordRetriever()
    retriever.add_documents(
        [
            Document(id="py", content="Python est un langage de programmation polyvalent."),
            Document(id="git", content="Git est un système de contrôle de version distribué."),
            Document(
                id="sql", content="SQLite est une base de données embarquée sans serveur."
            ),
        ]
    )
    return retriever


def test_keyword_retriever_ranks_relevant_first():
    retriever = _index_corpus()
    results = retriever.retrieve("langage de programmation", k=2)
    assert results
    assert results[0].document.id == "py"
    assert results[0].score > 0
    assert len(results) <= 2


def test_keyword_retriever_no_match_returns_empty():
    retriever = _index_corpus()
    assert retriever.retrieve("kubernetes") == []


def test_keyword_retriever_case_insensitive():
    retriever = _index_corpus()
    results = retriever.retrieve("PYTHON")
    assert results and results[0].document.id == "py"


def test_keyword_retriever_respects_k():
    retriever = _index_corpus()
    assert len(retriever.retrieve("de", k=1)) == 1
    assert retriever.retrieve("de", k=0) == []


def test_keyword_retriever_clear():
    retriever = _index_corpus()
    assert len(retriever) == 3
    retriever.clear()
    assert len(retriever) == 0
    assert retriever.retrieve("python") == []


def test_keyword_retriever_empty_query():
    assert _index_corpus().retrieve("?!") == []


def test_tokenize():
    assert tokenize("Bonjour Le Monde") == ["bonjour", "le", "monde"]
    assert tokenize("snake_case et accents: école") == ["snake_case", "et", "accents", "école"]


# --- ABCs ---------------------------------------------------------------------------------
def test_abstract_interfaces_cannot_be_instantiated():
    for abstract in (Chunker, EmbeddingProvider, Retriever):
        with pytest.raises(TypeError):
            abstract()  # type: ignore[abstract]


def test_retriever_interface_contract():
    """Any Retriever implementation honors add/retrieve/clear semantics."""

    class UpperRetriever(Retriever):
        def __init__(self) -> None:
            self.docs: list[Document] = []

        def add_documents(self, documents: list[Document]) -> None:
            self.docs.extend(documents)

        def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
            hits = [d for d in self.docs if query.lower() in d.content.lower()]
            return [RetrievedChunk(document=d, score=1.0) for d in hits[:k]]

        def clear(self) -> None:
            self.docs.clear()

    retriever = UpperRetriever()
    retriever.add_documents([Document(id="a", content="Hello World")])
    assert retriever.retrieve("hello", k=2)[0].document.id == "a"
    retriever.clear()
    assert retriever.retrieve("hello") == []


# --- memory + RAG integration -------------------------------------------------------------
def test_memory_manager_injects_rag_chunks():
    retriever = _index_corpus()
    manager = MemoryManager(Database(":memory:"), retriever=retriever)

    block = manager.build_context_block("comment gérer les versions de mon code ?")
    assert "Contexte documentaire" in block
    assert "Git est un système de contrôle de version" in block


def test_memory_manager_rag_and_memory_together():
    retriever = _index_corpus()
    manager = MemoryManager(Database(":memory:"), retriever=retriever)
    manager.remember("L'utilisateur écrit ses notes en Python", importance=5.0)

    block = manager.build_context_block("python")
    assert "Mémoire pertinente" in block
    assert "Contexte documentaire" in block
    assert "notes en Python" in block
    assert "langage de programmation" in block


def test_memory_manager_without_rag_match_only_memory():
    retriever = _index_corpus()
    manager = MemoryManager(Database(":memory:"), retriever=retriever)
    manager.remember("Un fait sur Python", importance=2.0)

    block = manager.build_context_block("kubernetes")
    assert block == ""  # no memory overlap, no RAG match → nothing injected


def test_memory_manager_rag_failure_is_graceful():
    class ExplodingRetriever(Retriever):
        def add_documents(self, documents: list[Document]) -> None: ...

        def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
            raise RuntimeError("boom")

        def clear(self) -> None: ...

    manager = MemoryManager(Database(":memory:"), retriever=ExplodingRetriever())
    manager.remember("Fait important python", importance=3.0)
    block = manager.build_context_block("python")
    # The memory section is still there; the retriever failure was swallowed.
    assert "Mémoire pertinente" in block
    assert "Contexte documentaire" not in block


def test_chunker_feeds_retriever_end_to_end():
    chunker = SimpleChunker(chunk_size=120, overlap=30)
    long_doc = (
        "Nocki Agent est un agent IA local. " * 10
        + "Il utilise LangGraph pour l'orchestration et Ollama pour l'inférence. " * 5
    )
    chunks = chunker.split(long_doc, metadata={"id": "about", "source": "README"})
    retriever = KeywordRetriever()
    retriever.add_documents(chunks)
    manager = MemoryManager(Database(":memory:"), retriever=retriever)

    block = manager.build_context_block("orchestration LangGraph")
    assert "Contexte documentaire" in block
    assert "LangGraph" in block
