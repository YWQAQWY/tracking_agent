"""Concurrent orchestration across search queries and providers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from itertools import zip_longest

from src.models.search_result import SearchResult
from src.runtime.context import is_retryable_exception, run_async_operation
from src.runtime.errors import BudgetExceededError
from src.search.base import SearchError, SearchProvider


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchCoverage:
    """Outcome of one query/provider search operation."""

    query_index: int
    query: str
    provider: str
    result_count: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class SearchBatch:
    """Combined search results plus an inspectable execution trace."""

    results: tuple[SearchResult, ...]
    coverage: tuple[SearchCoverage, ...]
    providers: tuple[str, ...]
    raw_result_count: int
    duplicate_count: int


class SourceManager:
    """Run queries across providers without coupling providers to each other."""

    def __init__(
        self,
        providers: list[SearchProvider],
        results_per_task: int = 5,
        max_combined_results: int = 20,
        max_concurrency: int = 5,
    ) -> None:
        if not providers:
            raise ValueError("至少需要一个 SearchProvider")
        if results_per_task < 1 or max_combined_results < 1 or max_concurrency < 1:
            raise ValueError("搜索数量和并发配置必须大于 0")
        names = [provider.name for provider in providers]
        if len(names) != len(set(names)):
            raise ValueError("SearchProvider name 必须唯一")

        self.providers = tuple(providers)
        self.results_per_task = results_per_task
        self.max_combined_results = max_combined_results
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def search(self, queries: list[str]) -> SearchBatch:
        """Execute every query/provider pair concurrently and isolate failures."""
        clean_queries = [query.strip() for query in queries if query.strip()]
        if not clean_queries:
            raise ValueError("queries 不能为空")

        # A natural-language question has several useful search expressions,
        # and each provider has different index coverage. SourceManager owns the
        # cross-product so no concrete provider needs to know about another.
        tasks: list[tuple[int, str, SearchProvider]] = [
            (query_index, query, provider)
            for query_index, query in enumerate(clean_queries, start=1)
            for provider in self.providers
        ]
        logger.info(
            "Enabled search providers: %s",
            ", ".join(provider.name for provider in self.providers),
        )
        logger.info("Starting %d concurrent search tasks", len(tasks))

        # Network requests spend most of their time waiting. asyncio lets other
        # searches make progress during that wait, while the semaphore prevents
        # an unbounded burst of outbound requests.
        outcomes = await asyncio.gather(
            *(
                self._search_one(provider, query)
                for _, query, provider in tasks
            ),
            return_exceptions=True,
        )

        groups: list[list[SearchResult]] = []
        coverage: list[SearchCoverage] = []
        raw_result_count = 0
        for (query_index, query, provider), outcome in zip(tasks, outcomes):
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, BudgetExceededError):
                raise outcome
            if isinstance(outcome, Exception):
                message = f"{outcome.__class__.__name__}: {outcome}"
                logger.warning(
                    "%s failed for Query[%d]: %s",
                    provider.name,
                    query_index,
                    message,
                )
                groups.append([])
                coverage.append(
                    SearchCoverage(query_index, query, provider.name, 0, message)
                )
                continue

            # The manager stamps provenance at its orchestration boundary so a
            # provider cannot accidentally attach metadata for the wrong task.
            results = [
                result.model_copy(
                    update={"provider": provider.name, "query": query}
                )
                for result in outcome
            ]
            groups.append(results)
            raw_result_count += len(results)
            coverage.append(
                SearchCoverage(query_index, query, provider.name, len(results))
            )
            logger.info(
                "%s returned %d results for Query[%d]",
                provider.name,
                len(results),
                query_index,
            )

        combined, duplicate_count = self._round_robin_unique(groups)
        logger.info("Combined %d raw search results", raw_result_count)
        logger.info(
            "Retained %d results after %d exact URL duplicates and result cap",
            len(combined),
            duplicate_count,
        )
        return SearchBatch(
            results=tuple(combined),
            coverage=tuple(coverage),
            providers=tuple(provider.name for provider in self.providers),
            raw_result_count=raw_result_count,
            duplicate_count=duplicate_count,
        )

    async def _search_one(
        self, provider: SearchProvider, query: str
    ) -> list[SearchResult]:
        async with self._semaphore:
            return await run_async_operation(
                "search",
                f"{provider.name}:{query}",
                lambda: provider.search(query, self.results_per_task),
                retryable=lambda error: isinstance(error, SearchError)
                or is_retryable_exception(error),
            )

    def _round_robin_unique(
        self, groups: list[list[SearchResult]]
    ) -> tuple[list[SearchResult], int]:
        """Interleave task groups and remove exact URL duplicates only."""
        combined: list[SearchResult] = []
        seen_urls: set[str] = set()
        duplicate_count = 0
        sentinel = object()
        for row in zip_longest(*groups, fillvalue=sentinel):
            for result in row:
                if result is sentinel:
                    continue
                url = str(result.url)
                if url in seen_urls:
                    duplicate_count += 1
                    continue
                seen_urls.add(url)
                combined.append(result)
                if len(combined) >= self.max_combined_results:
                    return combined, duplicate_count
        return combined, duplicate_count
