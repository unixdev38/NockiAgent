"""KeywordRetriever — a dependency-free TF-IDF reference implementation.

Scoring: for every query token present in a document we add
`tf(token, doc) / len(doc) * idf(token)`, where
`idf = ln((1 + N) / (1 + df)) + 1` (smoothed, never negative). Documents
sharing no token with the query are never returned.

This is deliberately simple: it gives correct *ranking* semantics for small
local corpora. A future embeddings-based retriever (Ollama embeddings +
Chroma/FAISS) implements the same `Retriever` interface and can replace it
without touching the rest of the code base.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.rag.base import Document, RetrievedChunk, Retriever

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class KeywordRetriever(Retriever):
    """In-memory TF-IDF retriever over added documents."""

    def __init__(self) -> None:
        self._documents: list[Document] = []

    def add_documents(self, documents: list[Document]) -> None:
        self._documents.extend(documents)

    def clear(self) -> None:
        self._documents.clear()

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        if k <= 0 or not self._documents:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        doc_tokens = [tokenize(document.content) for document in self._documents]
        total = len(self._documents)

        document_frequency: Counter[str] = Counter()
        for tokens in doc_tokens:
            document_frequency.update(set(tokens))

        chunks: list[RetrievedChunk] = []
        for document, tokens in zip(self._documents, doc_tokens, strict=True):
            if not tokens:
                continue
            term_frequency = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                count = term_frequency.get(token)
                if not count:
                    continue
                idf = math.log((1 + total) / (1 + document_frequency[token])) + 1
                score += (count / len(tokens)) * idf
            if score > 0:
                chunks.append(RetrievedChunk(document=document, score=round(score, 6)))

        chunks.sort(key=lambda chunk: chunk.score, reverse=True)
        return chunks[:k]

    def __len__(self) -> int:
        return len(self._documents)
