"""Readable sliding-window character chunking."""

from __future__ import annotations

import hashlib
import logging

from src.models.chunk import DocumentChunk
from src.models.document import Document


logger = logging.getLogger(__name__)


class DocumentChunker:
    """Split documents into bounded overlapping passages."""

    def __init__(
        self,
        chunk_size: int = 1_200,
        chunk_overlap: int = 200,
        min_chunk_length: int = 100,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size 必须大于 0")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")
        if not 1 <= min_chunk_length <= chunk_size:
            raise ValueError("min_chunk_length 必须在 1 到 chunk_size 之间")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    def chunk(self, documents: list[Document]) -> list[DocumentChunk]:
        """Create chunks while preserving URL, title, and stable indices."""
        chunks: list[DocumentChunk] = []
        for document in documents:
            chunks.extend(self._chunk_document(document))
        logger.info(
            "Generated %d chunks from %d documents", len(chunks), len(documents)
        )
        return chunks

    def _chunk_document(self, document: Document) -> list[DocumentChunk]:
        text = document.text.strip()
        step = self.chunk_size - self.chunk_overlap
        document_id = hashlib.sha256(str(document.url).encode("utf-8")).hexdigest()[:12]
        chunks: list[DocumentChunk] = []

        # Chunking makes relevant middle/end passages retrievable. The overlap
        # repeats boundary text so a sentence split between windows is not lost.
        for start in range(0, len(text), step):
            passage = text[start : start + self.chunk_size].strip()
            if len(passage) < self.min_chunk_length:
                continue
            chunk_index = len(chunks)
            chunks.append(
                DocumentChunk(
                    id=f"{document_id}:{chunk_index}",
                    text=passage,
                    url=str(document.url),
                    title=document.title,
                    chunk_index=chunk_index,
                )
            )
            if start + self.chunk_size >= len(text):
                break
        return chunks
