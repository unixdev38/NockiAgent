"""RAG layer (Phase 6): architecture-ready, dependency-free reference implementation.

- `base.py`: Document / RetrievedChunk models + Chunker / EmbeddingProvider /
  Retriever ABCs. A future embeddings retriever (Ollama + Chroma/FAISS, see
  the optional `rag` dependency group) implements the same interfaces.
- `chunking.py`: SimpleChunker (overlapping windows).
- `keyword.py`: KeywordRetriever (TF-IDF, no dependency).

Integration: `MemoryManager(retriever=...)` injects retrieved chunks into
`build_context_block()`, next to relevant memories.
"""

from app.rag.base import (
    Chunker,
    Document,
    EmbeddingProvider,
    RetrievedChunk,
    Retriever,
)
from app.rag.chunking import SimpleChunker
from app.rag.keyword import KeywordRetriever, tokenize

__all__ = [
    "Chunker",
    "Document",
    "EmbeddingProvider",
    "KeywordRetriever",
    "RetrievedChunk",
    "Retriever",
    "SimpleChunker",
    "tokenize",
]
