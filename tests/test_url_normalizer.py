from src.models.search_result import SearchResult
from src.search.url_normalizer import URLDeduplicator, URLNormalizer


def result(url: str) -> SearchResult:
    return SearchResult(
        title="title",
        url=url,
        snippet="snippet",
        provider="test",
        query="query",
    )


def test_url_normalizer_removes_fragment_tracking_and_trailing_slash() -> None:
    normalized = URLNormalizer().normalize(
        "HTTPS://Example.COM/page/?utm_source=google&id=123#section"
    )
    assert normalized == "https://example.com/page?id=123"


def test_url_normalizer_removes_common_tracking_parameters() -> None:
    normalized = URLNormalizer().normalize(
        "https://example.com/page?utm_medium=x&utm_campaign=y&gclid=1&fbclid=2&id=3"
    )
    assert normalized == "https://example.com/page?id=3"


def test_url_normalizer_sorts_query_parameters_and_preserves_root() -> None:
    normalizer = URLNormalizer()
    assert normalizer.normalize("https://example.com/?b=2&a=1") == (
        "https://example.com/?a=1&b=2"
    )
    assert normalizer.normalize("https://example.com?a=1&b=2") == (
        "https://example.com/?a=1&b=2"
    )


def test_url_deduplicator_keeps_one_normalized_result() -> None:
    unique = URLDeduplicator().deduplicate(
        [
            result("https://example.com/article"),
            result("https://example.com/article/"),
            result("https://example.com/article?utm_source=x"),
        ]
    )
    assert len(unique) == 1
    assert str(unique[0].url) == "https://example.com/article"
