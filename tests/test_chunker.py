from src.models.document import Document
from src.retrieval.chunker import DocumentChunker


def test_chunker_creates_bounded_overlapping_chunks_with_metadata() -> None:
    text = "".join(str(index % 10) for index in range(350))
    document = Document(url="https://example.com/article", title="Article", text=text)

    chunks = DocumentChunker(
        chunk_size=120,
        chunk_overlap=20,
        min_chunk_length=30,
    ).chunk([document])

    assert len(chunks) == 4
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert chunks[0].text[-20:] == chunks[1].text[:20]
    assert chunks[0].url == "https://example.com/article"
    assert chunks[0].title == "Article"
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert chunks[0].id.endswith(":0")


def test_chunker_filters_short_final_chunk() -> None:
    document = Document(url="https://example.com", text="x" * 130)
    chunks = DocumentChunker(
        chunk_size=100,
        chunk_overlap=10,
        min_chunk_length=50,
    ).chunk([document])
    assert len(chunks) == 1


def test_chunker_validates_overlap() -> None:
    try:
        DocumentChunker(chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("invalid overlap should fail")
