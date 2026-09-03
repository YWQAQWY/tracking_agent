"""Retrieval scoring models passed toward final answer generation."""

from dataclasses import dataclass

from src.models.chunk import DocumentChunk


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk selected by fast dense embedding retrieval."""

    chunk: DocumentChunk
    embedding_score: float


@dataclass(frozen=True, slots=True)
class Evidence:
    """A final passage selected for the local answer model."""

    text: str
    url: str
    title: str | None
    chunk_index: int
    embedding_score: float
    rerank_score: float | None

    @classmethod
    def from_scored_chunk(
        cls,
        candidate: ScoredChunk,
        rerank_score: float | None,
    ) -> "Evidence":
        chunk = candidate.chunk
        return cls(
            text=chunk.text,
            url=chunk.url,
            title=chunk.title,
            chunk_index=chunk.chunk_index,
            embedding_score=candidate.embedding_score,
            rerank_score=rerank_score,
        )

    def to_scored_chunk(self) -> ScoredChunk:
        """Recreate a reranker candidate for final cross-round selection."""
        chunk = DocumentChunk(
            id=f"{self.url}#chunk-{self.chunk_index}",
            text=self.text,
            url=self.url,
            title=self.title,
            chunk_index=self.chunk_index,
        )
        return ScoredChunk(chunk=chunk, embedding_score=self.embedding_score)
