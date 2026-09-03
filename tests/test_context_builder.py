from src.context.context_builder import ContextBuilder
from src.models.evidence import Evidence


def make_evidence(url: str, title: str, text: str) -> Evidence:
    return Evidence(
        url=url,
        title=title,
        text=text,
        chunk_index=0,
        embedding_score=0.8,
        rerank_score=1.2,
    )


def test_context_builder_includes_source_metadata_and_content() -> None:
    evidence = [
        make_evidence("https://a.com", "A", "Article A"),
        make_evidence("https://b.com", "B", "Article B"),
    ]

    context = ContextBuilder(100, 1_000).build(evidence)

    assert "[Source 1]" in context
    assert "Title: A" in context
    assert "URL: https://a.com" in context
    assert "Chunk: 0" in context
    assert "Content:\nArticle A" in context
    assert "[Source 2]" in context
    assert "Title: B" in context


def test_context_builder_limits_each_evidence_text() -> None:
    evidence = make_evidence("https://a.com", "A", "x" * 500)
    context = ContextBuilder(50, 1_000).build([evidence])
    content = context.split("Content:\n", maxsplit=1)[1]
    assert content == "x" * 50


def test_context_builder_respects_total_character_limit() -> None:
    evidence = [
        make_evidence("https://a.com", "A", "a" * 500),
        make_evidence("https://b.com", "B", "b" * 500),
    ]
    context = ContextBuilder(200, 180).build(evidence)
    assert len(context) <= 180
