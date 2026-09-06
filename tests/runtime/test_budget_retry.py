import asyncio

import pytest

from src.runtime.budgets import BudgetTracker
from src.runtime.errors import BudgetExceededError, NonRetryableError, RetryableError
from src.runtime.models import RuntimeBudget
from src.runtime.retry import RetryPolicy, retry_async, retry_sync


def test_budget_is_consumed_before_call_and_stops_at_limit() -> None:
    tracker = BudgetTracker(
        RuntimeBudget(
            max_research_rounds=1,
            max_search_requests=3,
            max_crawl_requests=1,
            max_llm_calls=1,
        )
    )
    tracker.consume("search", 3)
    with pytest.raises(BudgetExceededError, match="limit=3"):
        tracker.consume("search")
    assert tracker.counters.search_requests == 3


def test_llm_budget_is_independent_from_search_budget() -> None:
    tracker = BudgetTracker(RuntimeBudget(max_llm_calls=1))
    tracker.consume("llm")
    with pytest.raises(BudgetExceededError):
        tracker.consume("llm")
    assert tracker.counters.search_requests == 0


def test_retryable_error_succeeds_on_third_attempt_with_backoff() -> None:
    async def scenario() -> None:
        attempts = 0
        delays: list[float] = []

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RetryableError("temporary")
            return "ok"

        async def sleeper(delay: float) -> None:
            delays.append(delay)

        result = await retry_async(
            operation,
            RetryPolicy(3, 0.5, 2.0, 0.0),
            lambda error: isinstance(error, RetryableError),
            sleeper=sleeper,
        )
        assert result == "ok"
        assert attempts == 3
        assert delays == [0.5, 1.0]

    asyncio.run(scenario())


def test_retry_limit_is_exact() -> None:
    async def scenario() -> None:
        attempts = 0

        async def operation() -> None:
            nonlocal attempts
            attempts += 1
            raise RetryableError("temporary")

        with pytest.raises(RetryableError):
            await retry_async(
                operation,
                RetryPolicy(2, 0.0, 0.0, 0.0),
                lambda error: isinstance(error, RetryableError),
            )
        assert attempts == 2

    asyncio.run(scenario())


def test_non_retryable_error_is_not_repeated() -> None:
    async def scenario() -> None:
        attempts = 0

        async def operation() -> None:
            nonlocal attempts
            attempts += 1
            raise NonRetryableError("bad config")

        with pytest.raises(NonRetryableError):
            await retry_async(
                operation,
                RetryPolicy(3, 0.0, 0.0, 0.0),
                lambda error: isinstance(error, RetryableError),
            )
        assert attempts == 1

    asyncio.run(scenario())


def test_cancelled_retry_sleep_propagates_immediately() -> None:
    async def scenario() -> None:
        started = asyncio.Event()

        async def operation() -> None:
            started.set()
            raise RetryableError("temporary")

        task = asyncio.create_task(
            retry_async(
                operation,
                RetryPolicy(3, 10.0, 10.0, 0.0),
                lambda error: isinstance(error, RetryableError),
            )
        )
        await started.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_sync_retry_uses_same_attempt_limit() -> None:
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RetryableError("temporary")
        return "ok"

    assert retry_sync(
        operation,
        RetryPolicy(2, 0.0, 0.0, 0.0),
        lambda error: isinstance(error, RetryableError),
    ) == "ok"
    assert attempts == 2


@pytest.mark.parametrize("resource", ["research_round", "search", "crawl", "llm"])
def test_each_budget_type_is_enforced(resource: str) -> None:
    tracker = BudgetTracker(RuntimeBudget(1, 1, 1, 1))
    tracker.consume(resource)
    with pytest.raises(BudgetExceededError):
        tracker.consume(resource)


def test_unknown_budget_resource_is_rejected() -> None:
    with pytest.raises(ValueError, match="未知"):
        BudgetTracker(RuntimeBudget()).consume("browser")
