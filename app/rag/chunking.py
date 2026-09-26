"""Text chunking with overlapping windows.

`SimpleChunker` walks the text with a fixed window (`chunk_size`) advanced by
`chunk_size - overlap` characters, so consecutive chunks share `overlap`
characters of context. Chunk ids are derived from the source id in metadata,
making them stable and traceable back to their document.
"""

from __future__ import annotations

from typing import Any

from app.rag.base import Chunker, Document


class SimpleChunker(Chunker):
    """Character-window chunker with overlap (no external dependency)."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size doit être > 0 (reçu: {chunk_size})")
        if overlap < 0:
            raise ValueError(f"overlap doit être >= 0 (reçu: {overlap})")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) doit être strictement inférieur à chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[Document]:
        source_metadata = dict(metadata or {})
        base_id = str(source_metadata.get("id") or source_metadata.get("source") or "doc")

        text = text or ""
        if not text.strip():
            return []

        step = self.chunk_size - self.overlap
        documents: list[Document] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end]
            if piece.strip():
                documents.append(
                    Document(
                        id=f"{base_id}::chunk-{index}",
                        content=piece,
                        metadata={
                            **source_metadata,
                            "index": index,
                            "start": start,
                            "end": end,
                        },
                    )
                )
            if end >= len(text):
                break
            start += step
            index += 1
        return documents
