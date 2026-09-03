import numpy as np
import pytest

from src.models.chunk import DocumentChunk
from src.retrieval.embedder import BGEEmbedder, EmbeddingError
from src.retrieval.retriever import SemanticRetriever


def chunk(index: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=f"doc:{index}",
        text=text,
        url=f"https://example.com/{index}",
        title=f"Chunk {index}",
        chunk_index=index,
    )


class FakeEmbedder:
    model_name = "fake"
    device = "cpu"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def test_semantic_retriever_ranks_by_cosine_similarity() -> None:
    embedder = FakeEmbedder(
        {
            "question": [1.0, 0.0],
            "A": [1.0, 0.0],
            "B": [0.8, 0.2],
            "C": [0.0, 1.0],
        }
    )
    candidates = SemanticRetriever(embedder, top_k=3).retrieve(
        "question", [chunk(0, "A"), chunk(1, "B"), chunk(2, "C")]
    )

    assert [candidate.chunk.text for candidate in candidates] == ["A", "B", "C"]
    assert candidates[0].embedding_score > candidates[1].embedding_score
    assert candidates[1].embedding_score > candidates[2].embedding_score
    assert embedder.calls == [["question"], ["A", "B", "C"]]


def test_semantic_retriever_applies_top_k() -> None:
    vectors = {"question": [1.0, 0.0]}
    chunks = []
    for index in range(10):
        text = f"chunk-{index}"
        vectors[text] = [float(10 - index), 1.0]
        chunks.append(chunk(index, text))

    candidates = SemanticRetriever(FakeEmbedder(vectors), top_k=3).retrieve(
        "question", chunks
    )
    assert len(candidates) == 3


def test_bge_embedder_uses_one_batched_normalized_encode_call() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.arguments = None

        def encode(self, texts, **kwargs):
            self.arguments = (texts, kwargs)
            return [[1.0, 0.0], [0.0, 1.0]]

    model = FakeModel()
    vectors = BGEEmbedder(model=model, device="test", batch_size=7).encode(["a", "b"])

    assert vectors.shape == (2, 2)
    assert model.arguments[0] == ["a", "b"]
    assert model.arguments[1]["batch_size"] == 7
    assert model.arguments[1]["normalize_embeddings"] is True


def test_embedding_model_error_is_clear() -> None:
    class BrokenModel:
        def encode(self, texts, **kwargs):
            raise RuntimeError("out of memory")

    embedder = BGEEmbedder(model=BrokenModel(), device="test")
    with pytest.raises(EmbeddingError, match="显存/内存"):
        embedder.encode(["text"])


def test_bge_embedder_offload_preserves_model_for_reuse() -> None:
    class MovableModel:
        def __init__(self) -> None:
            self.moves: list[str] = []

        def encode(self, texts, **kwargs):
            return [[1.0, 0.0] for _ in texts]

        def to(self, device: str):
            self.moves.append(device)
            return self

    model = MovableModel()
    embedder = BGEEmbedder(model=model, device="cuda")

    embedder.encode(["first"])
    embedder.offload()
    embedder.encode(["second"])

    assert model.moves == ["cpu", "cuda"]
