from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx

from src.models.search_result import SearchResult
from src.runtime.context import is_retryable_exception, run_async_operation
from src.runtime.errors import NonRetryableError
from src.runtime.models import RunRequest, RunStatus, RuntimeBudget, RuntimeConfig
from src.runtime.retry import RetryPolicy
from src.runtime.runtime import AgentRuntime
from src.search.source_manager import SourceManager


def config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        overall_timeout_seconds=1,
        search_timeout_seconds=0.01,
        crawl_timeout_seconds=0.01,
        budget=RuntimeBudget(2, 5, 5, 5),
        network_retry=RetryPolicy(2, 0, 0, 0),
        llm_retry=RetryPolicy(2, 0, 0, 0),
        checkpoint_dir=str(tmp_path),
    )


def successful_result():
    return SimpleNamespace(
        answer="ok",
        research_trace=SimpleNamespace(stop_reason="sufficient", timings={}),
        grounding_trace=None,
    )


def result(url: str, provider: str, query: str) -> SearchResult:
    return SearchResult(
        title="title", url=url, snippet="snippet", provider=provider, query=query
    )


def test_partial_provider_failure_preserves_results_and_counts_calls(tmp_path) -> None:
    class Working:
        name = "working"

        async def search(self, query: str, limit: int):
            return [result("https://example.com/good", self.name, query)]

    class Broken:
        name = "broken"

        async def search(self, query: str, limit: int):
            raise ValueError("bad provider response")

    class Agent:
        async def run(self, question, resume_state=None, checkpoint_callback=None):
            batch = await SourceManager([Working(), Broken()]).search(["q"])
            assert len(batch.results) == 1
            assert not batch.coverage[1].succeeded
            return successful_result()

    outcome = asyncio.run(
        AgentRuntime(config(tmp_path)).run(RunRequest("q"), Agent())
    )
    assert outcome.status is RunStatus.SUCCEEDED
    assert outcome.trace.counters.search_requests == 2


def test_all_provider_failures_escalate_as_controlled_runtime_failure(tmp_path) -> None:
    class Broken:
        def __init__(self, name: str) -> None:
            self.name = name

        async def search(self, query: str, limit: int):
            raise ValueError("bad provider response")

    class Agent:
        async def run(self, question, resume_state=None, checkpoint_callback=None):
            batch = await SourceManager([Broken("a"), Broken("b")]).search(["q"])
            if not batch.results:
                raise NonRetryableError("all providers failed")

    outcome = asyncio.run(
        AgentRuntime(config(tmp_path)).run(RunRequest("q"), Agent())
    )
    assert outcome.status is RunStatus.FAILED
    assert outcome.error is not None
    assert "all providers failed" in outcome.error.message
    assert outcome.trace.counters.search_requests == 2


def test_crawl_operation_timeout_has_its_own_counter_and_error(tmp_path) -> None:
    class Agent:
        async def run(self, question, resume_state=None, checkpoint_callback=None):
            async def slow():
                await asyncio.sleep(1)

            await run_async_operation("crawl", "slow-page", slow)

    outcome = asyncio.run(
        AgentRuntime(config(tmp_path)).run(RunRequest("q"), Agent())
    )
    assert outcome.status is RunStatus.FAILED
    assert outcome.error is not None
    assert outcome.error.error_type == "CrawlerTimeoutError"
    assert outcome.trace.counters.crawl_requests == 2
    assert outcome.trace.counters.timeouts == 2


def test_http_retry_classification_accepts_temporary_status_only() -> None:
    request = httpx.Request("GET", "https://example.com")
    temporary = httpx.HTTPStatusError(
        "temporary", request=request, response=httpx.Response(429, request=request)
    )
    permanent = httpx.HTTPStatusError(
        "missing", request=request, response=httpx.Response(404, request=request)
    )
    assert is_retryable_exception(temporary)
    assert not is_retryable_exception(permanent)
