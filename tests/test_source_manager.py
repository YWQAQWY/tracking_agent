import asyncio

from src.models.search_result import SearchResult
from src.search.source_manager import SourceManager


def make_result(url: str, provider: str, query: str) -> SearchResult:
    return SearchResult(
        title=f"{provider} result",
        url=url,
        snippet="snippet",
        provider=provider,
        query=query,
    )


def test_source_manager_runs_multi_query_multi_source_concurrently() -> None:
    async def scenario() -> None:
        calls: list[tuple[str, str, int]] = []
        active = 0
        peak = 0

        class FakeProvider:
            def __init__(self, name: str) -> None:
                self.name = name

            async def search(self, query: str, limit: int) -> list[SearchResult]:
                nonlocal active, peak
                calls.append((self.name, query, limit))
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0)
                active -= 1
                return [make_result(f"https://{self.name}.com/{query}", self.name, query)]

        manager = SourceManager(
            [FakeProvider("a"), FakeProvider("b")],
            results_per_task=4,
            max_concurrency=3,
        )
        batch = await manager.search(["q1", "q2", "q3"])

        assert len(calls) == 6
        assert set(calls) == {
            (provider, query, 4)
            for query in ("q1", "q2", "q3")
            for provider in ("a", "b")
        }
        assert peak == 3
        assert len(batch.coverage) == 6
        assert all(item.succeeded for item in batch.coverage)
        assert {result.provider for result in batch.results} == {"a", "b"}
        assert {result.query for result in batch.results} == {"q1", "q2", "q3"}

    asyncio.run(scenario())


def test_source_manager_isolates_provider_failure() -> None:
    async def scenario() -> None:
        class WorkingProvider:
            name = "working"

            async def search(self, query: str, limit: int) -> list[SearchResult]:
                return [make_result("https://example.com/good", self.name, query)]

        class BrokenProvider:
            name = "broken"

            async def search(self, query: str, limit: int) -> list[SearchResult]:
                raise TimeoutError("provider timed out")

        batch = await SourceManager(
            [WorkingProvider(), BrokenProvider()]
        ).search(["q1"])

        assert [str(result.url) for result in batch.results] == [
            "https://example.com/good"
        ]
        assert batch.coverage[0].succeeded
        assert not batch.coverage[1].succeeded
        assert "TimeoutError" in (batch.coverage[1].error or "")

    asyncio.run(scenario())


def test_source_manager_removes_exact_url_duplicates_and_round_robins() -> None:
    async def scenario() -> None:
        class FakeProvider:
            def __init__(self, name: str, urls: list[str]) -> None:
                self.name = name
                self.urls = urls

            async def search(self, query: str, limit: int) -> list[SearchResult]:
                return [make_result(url, self.name, query) for url in self.urls]

        batch = await SourceManager(
            [
                FakeProvider(
                    "a", ["https://same.com/article", "https://a.com/second"]
                ),
                FakeProvider(
                    "b", ["https://same.com/article", "https://b.com/second"]
                ),
            ]
        ).search(["q1"])

        assert batch.raw_result_count == 4
        assert batch.duplicate_count == 1
        assert [str(result.url) for result in batch.results] == [
            "https://same.com/article",
            "https://a.com/second",
            "https://b.com/second",
        ]

    asyncio.run(scenario())
