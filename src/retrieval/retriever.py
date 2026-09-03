"""In-memory cosine-similarity retrieval over current-run chunks."""

from __future__ import annotations

import logging

import numpy as np

from src.models.chunk import DocumentChunk
from src.models.evidence import ScoredChunk
from src.retrieval.embedder import Embedder, EmbeddingError


logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """Raised when semantic retrieval cannot produce valid candidates."""


class SemanticRetriever:
    """Use a bi-encoder to select broad candidates for slower reranking."""

    def __init__(self, embedder: Embedder, top_k: int = 20) -> None:
        if top_k < 1:
            raise ValueError("embedding top_k 必须大于 0")
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        chunks: list[DocumentChunk],
    ) -> list[ScoredChunk]:
        clean_question = question.strip()
        if not clean_question:
            raise RetrievalError("原始用户问题不能为空。")
        if not chunks:
            return []

        try:
            # SearchPlanner queries discover pages; the original question is
            # the authoritative information need for semantic relevance.
            query_vector = self.embedder.encode([clean_question])
            chunk_vectors = self.embedder.encode([chunk.text for chunk in chunks])
            scores = self.cosine_scores(query_vector, chunk_vectors)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise RetrievalError("Embedding similarity 计算失败。") from exc

        order = np.argsort(-scores, kind="stable")[: self.top_k]
        candidates = [
            ScoredChunk(chunk=chunks[int(index)], embedding_score=float(scores[index]))
            for index in order
        ]
        logger.info(
            "Semantic retrieval selected top %d candidates from %d chunks",
            len(candidates),
            len(chunks),
        )
        return candidates

    def offload(self) -> None:
        """Release the embedder's CUDA allocation between pipeline stages."""
        offload = getattr(self.embedder, "offload", None)
        if callable(offload):
            offload()

    @staticmethod
    def cosine_scores(
        query_vectors: np.ndarray,
        document_vectors: np.ndarray,
    ) -> np.ndarray:
        query = np.asarray(query_vectors, dtype=np.float32)
        documents = np.asarray(document_vectors, dtype=np.float32)
        if query.ndim != 2 or query.shape[0] != 1:
            raise RetrievalError("Query embedding 必须包含一个二维向量。")
        if documents.ndim != 2 or documents.shape[0] < 1:
            raise RetrievalError("Chunk embeddings 必须是非空二维矩阵。")
        if query.shape[1] != documents.shape[1]:
            raise RetrievalError("Query 与 Chunk embedding 维度不一致。")

        query_norm = np.linalg.norm(query[0])
        document_norms = np.linalg.norm(documents, axis=1)
        if query_norm == 0:
            raise RetrievalError("Query embedding 是零向量。")
        safe_norms = np.where(document_norms == 0, 1.0, document_norms)
        normalized_query = query[0] / query_norm
        normalized_documents = documents / safe_norms[:, None]
        return normalized_documents @ normalized_query
