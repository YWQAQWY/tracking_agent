import numpy as np

from src.models.chunk import DocumentChunk
from src.models.evidence import ScoredChunk
from src.retrieval.reranker import BGEReranker, EmbeddingOnlyReranker


def candidate(url: str, index: int, embedding_score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=DocumentChunk(
            id=f"{url}:{index}",
            text=f"chunk {index}",
            url=url,
            title="title",
            chunk_index=index,
        ),
        embedding_score=embedding_score,
    )


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.arguments = None

    def predict(self, pairs, **kwargs):
        self.arguments = (pairs, kwargs)
        return np.asarray(self.scores, dtype=np.float32)


def test_reranker_can_reverse_embedding_ranking_and_apply_top_k() -> None:
    candidates = [
        candidate("https://a.com", 0, 0.9),
        candidate("https://b.com", 0, 0.8),
        candidate("https://c.com", 0, 0.7),
    ]
    model = FakeCrossEncoder([2.0, 1.0, 5.0])
    evidence = BGEReranker(
        model=model,
        device="test",
        top_k=2,
        max_chunks_per_document=2,
    ).rerank("question", candidates)

    assert [item.url for item in evidence] == ["https://c.com", "https://a.com"]
    assert [item.rerank_score for item in evidence] == [5.0, 2.0]
    assert len(model.arguments[0]) == 3


def test_reranker_limits_chunks_per_document() -> None:
    candidates = [
        candidate("https://a.com", 0, 0.9),
        candidate("https://a.com", 1, 0.8),
        candidate("https://a.com", 2, 0.7),
        candidate("https://a.com", 3, 0.6),
        candidate("https://b.com", 0, 0.5),
    ]
    evidence = BGEReranker(
        model=FakeCrossEncoder([5.0, 4.0, 3.0, 2.0, 1.0]),
        device="test",
        top_k=5,
        max_chunks_per_document=2,
    ).rerank("question", candidates)

    assert sum(item.url == "https://a.com" for item in evidence) == 2
    assert any(item.url == "https://b.com" for item in evidence)


def test_reranker_reduces_twenty_candidates_to_six_evidence() -> None:
    candidates = [
        candidate(f"https://example.com/{index}", 0, 1.0 - index / 100)
        for index in range(20)
    ]
    evidence = BGEReranker(
        model=FakeCrossEncoder([float(index) for index in range(20)]),
        device="test",
        top_k=6,
    ).rerank("question", candidates)

    assert len(evidence) == 6
    assert [item.url for item in evidence] == [
        f"https://example.com/{index}" for index in range(19, 13, -1)
    ]


def test_no_reranker_mode_uses_embedding_order_and_diversity() -> None:
    candidates = [
        candidate("https://a.com", 0, 0.9),
        candidate("https://a.com", 1, 0.8),
        candidate("https://b.com", 0, 0.7),
    ]
    evidence = EmbeddingOnlyReranker(
        top_k=2,
        max_chunks_per_document=1,
    ).rerank("question", candidates)

    assert [item.url for item in evidence] == ["https://a.com", "https://b.com"]
    assert all(item.rerank_score is None for item in evidence)


def test_reranker_offload_preserves_model_for_reuse() -> None:
    class MovableCrossEncoder(FakeCrossEncoder):
        def __init__(self) -> None:
            super().__init__([1.0])
            self.model = self
            self.moves: list[str] = []

        def to(self, device: str):
            self.moves.append(device)
            return self

    model = MovableCrossEncoder()
    reranker = BGEReranker(model=model, device="cuda", top_k=1)
    candidates = [candidate("https://a.com", 0, 0.9)]

    reranker.rerank("question", candidates)
    reranker.offload()
    reranker.rerank("question", candidates)

    assert model.moves == ["cpu", "cuda"]
