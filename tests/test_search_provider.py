import asyncio

import httpx
import pytest

from src.models.search_result import SearchResult
from src.search.base import SearchError, SearchProvider
from src.search.ddgs_provider import DDGSSearchProvider
from src.search.wikipedia_provider import WikipediaSearchProvider


def test_provider_implements_interface() -> None:
    assert isinstance(
        DDGSSearchProvider(search_func=lambda *args, **kwargs: []), SearchProvider
    )
    assert issubclass(WikipediaSearchProvider, SearchProvider)


def test_provider_normalizes_ddgs_fields_and_adds_metadata() -> None:
    def fake_search(query: str, max_results: int):
        assert query == "qwen3"
        assert max_results == 2
        return [
            {"title": " Result One ", "href": "https://example.com/1", "body": " One "},
            {"title": "", "href": "not-a-url", "body": "invalid"},
            {"title": "Result Two", "url": "https://example.com/2", "snippet": "Two"},
        ]

    results = asyncio.run(
        DDGSSearchProvider(search_func=fake_search).search(" qwen3 ", 2)
    )
    assert results == [
        SearchResult(
            title="Result One",
            url="https://example.com/1",
            snippet="One",
            provider="ddgs",
            query="qwen3",
        ),
        SearchResult(
            title="Result Two",
            url="https://example.com/2",
            snippet="Two",
            provider="ddgs",
            query="qwen3",
        ),
    ]


def test_provider_returns_empty_list_for_no_results() -> None:
    provider = DDGSSearchProvider(search_func=lambda *args, **kwargs: [])
    assert asyncio.run(provider.search("unusual query", 5)) == []


def test_provider_wraps_network_failure() -> None:
    def failing_search(*args, **kwargs):
        raise OSError("network down")

    provider = DDGSSearchProvider(search_func=failing_search)
    with pytest.raises(SearchError, match="网络"):
        asyncio.run(provider.search("qwen3", 5))


@pytest.mark.parametrize(("query", "max_results"), [("", 5), ("qwen3", 0), ("qwen3", 11)])
def test_provider_validates_input(query: str, max_results: int) -> None:
    provider = DDGSSearchProvider(search_func=lambda *args, **kwargs: [])
    with pytest.raises(SearchError):
        asyncio.run(provider.search(query, max_results))


def test_wikipedia_provider_normalizes_api_response() -> None:
    async def scenario() -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            assert request.url.params["srsearch"] == "retrieval augmented generation"
            return httpx.Response(
                200,
                json={
                    "query": {
                        "search": [
                            {
                                "title": "Retrieval-augmented generation",
                                "snippet": "A <span>retrieval</span> technique",
                            }
                        ]
                    }
                },
                request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            provider = WikipediaSearchProvider(client=client)
            results = await provider.search("retrieval augmented generation", 2)

        assert len(results) == 1
        assert results[0].provider == "wikipedia"
        assert results[0].query == "retrieval augmented generation"
        assert "<span>" not in results[0].snippet
        assert str(results[0].url).startswith("https://en.wikipedia.org/wiki/")

    asyncio.run(scenario())
