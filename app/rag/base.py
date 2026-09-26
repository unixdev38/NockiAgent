"""RAG primitives — interfaces and data models.

Phase 6 ships the *architecture* without heavy dependencies: a reference
`KeywordRetriever` (TF-IDF, pure Python) implements the same `Retriever`
interface a future embeddings-based retriever (Ollama embeddings + Chroma /
FAISS, see the optional `rag` dependency group) will implement. Swapping the
implementation requires no change anywhere else — that is the whole point of
these ABCs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A unit of indexable content."""

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A document returned by a retriever, with its relevance score."""

    document: Document
    score: float


class Chunker(ABC):
    """Splits raw text into Documents (with metadata propagation)."""

    @abstractmethod
    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        """Split `text` into chunk Documents."""


class EmbeddingProvider(ABC):
    """Turns texts into vectors (future: Ollama embeddings)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text."""


class Retriever(ABC):
    """Stores documents and returns the most relevant ones for a query."""

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """Index documents for retrieval."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        """Return the top-k most relevant chunks for `query`."""

    @abstractmethod
    def clear(self) -> None:
        """Drop all indexed documents."""
