"""A bounded passage cut from one extracted web document."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Chunk text and the source metadata needed after retrieval."""

    id: str
    text: str
    url: str
    title: str | None
    chunk_index: int
