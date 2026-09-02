import pytest

from src.models.search_result import SearchResult
from src.search.base import SearchError, SearchProvider
from src.search.ddgs_provider import DDGSSearchProvider


def test_provider_implements_interface() -> None:
    assert isinstance(DDGSSearchProvider(search_func=lambda *args, **kwargs: []), SearchProvider)


def test_provider_normalizes_ddgs_fields_and_skips_invalid_rows() -> None:
    def fake_search(query: str, max_results: int):
        assert query == "qwen3"
        assert max_results == 2
        return [
            {"title": " Result One ", "href": "https://example.com/1", "body": " One "},
            {"title": "", "href": "not-a-url", "body": "invalid"},
            {"title": "Result Two", "url": "https://example.com/2", "snippet": "Two"},
        ]

    results = DDGSSearchProvider(search_func=fake_search).search(" qwen3 ", 2)
    assert results == [
        SearchResult(title="Result One", url="https://example.com/1", snippet="One"),
        SearchResult(title="Result Two", url="https://example.com/2", snippet="Two"),
    ]


def test_provider_returns_empty_list_for_no_results() -> None:
    provider = DDGSSearchProvider(search_func=lambda *args, **kwargs: [])
    assert provider.search("unusual query", 5) == []


def test_provider_wraps_network_failure() -> None:
    def failing_search(*args, **kwargs):
        raise OSError("network down")

    provider = DDGSSearchProvider(search_func=failing_search)
    with pytest.raises(SearchError, match="网络"):
        provider.search("qwen3", 5)


@pytest.mark.parametrize(("query", "max_results"), [("", 5), ("qwen3", 0), ("qwen3", 11)])
def test_provider_validates_input(query: str, max_results: int) -> None:
    provider = DDGSSearchProvider(search_func=lambda *args, **kwargs: [])
    with pytest.raises(SearchError):
        provider.search(query, max_results)

