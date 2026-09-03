from src.models.document import Document
from src.retrieval.deduplicator import ContentDeduplicator


def document(url: str, text: str) -> Document:
    return Document(url=url, title="title", text=text)


def test_content_deduplicator_normalizes_case_and_whitespace() -> None:
    unique = ContentDeduplicator().deduplicate(
        [
            document("https://a.com", "Hello world\n\nfrom Tracker"),
            document("https://b.com", " hello   WORLD from tracker "),
        ]
    )
    assert len(unique) == 1
    assert str(unique[0].url) == "https://a.com/"


def test_content_deduplicator_preserves_different_articles() -> None:
    documents = [
        document("https://a.com", "Article A has distinct information."),
        document("https://b.com", "Article B has different information."),
    ]
    assert ContentDeduplicator().deduplicate(documents) == documents
