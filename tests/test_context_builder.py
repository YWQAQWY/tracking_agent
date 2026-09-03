from src.context.context_builder import ContextBuilder
from src.models.document import Document


def make_document(url: str, title: str, text: str) -> Document:
    return Document(url=url, title=title, text=text)


def test_context_builder_includes_source_metadata_and_content() -> None:
    documents = [
        make_document("https://a.com", "A", "Article A"),
        make_document("https://b.com", "B", "Article B"),
    ]

    context = ContextBuilder(100, 1_000).build(documents)

    assert "[Source 1]" in context
    assert "Title: A" in context
    assert "URL: https://a.com/" in context
    assert "Content:\nArticle A" in context
    assert "[Source 2]" in context
    assert "Title: B" in context


def test_context_builder_limits_each_document_text() -> None:
    document = make_document("https://a.com", "A", "x" * 500)
    context = ContextBuilder(50, 1_000).build([document])
    content = context.split("Content:\n", maxsplit=1)[1]
    assert content == "x" * 50


def test_context_builder_respects_total_character_limit() -> None:
    documents = [
        make_document("https://a.com", "A", "a" * 500),
        make_document("https://b.com", "B", "b" * 500),
    ]
    context = ContextBuilder(200, 180).build(documents)
    assert len(context) <= 180

