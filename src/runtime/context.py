"""Per-run context shared by existing tools without duplicating runtime plumbing."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TypeVar

import httpx

from src.runtime.budgets import BudgetTracker
from src.runtime.errors import (
    CrawlerTimeoutError,
    ModelTimeoutError,
    RetryableError,
    SearchTimeoutError,
)
from src.runtime.models import RunStage
from src.runtime.retry import RetryPolicy, retry_async, retry_sync


ResultT = TypeVar("ResultT")
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeRecorder:
    tracker: BudgetTracker

    def retry(self) -> None:
        self.tracker.counters.retries += 1

    def timeout(self) -> None:
        self.tracker.counters.timeouts += 1


class RuntimeExecutionContext:
    """Budget, timeout, retry, and correlated logging for one active run."""

    def __init__(
        self,
        run_id: str,
        tracker: BudgetTracker,
        network_retry: RetryPolicy,
        llm_retry: RetryPolicy,
        search_timeout: float,
        crawl_timeout: float,
        overall_deadline: float | None = None,
        stage_callback: Callable[[RunStage], None] | None = None,
    ) -> None:
        self.run_id = run_id
        self.tracker = tracker
        self.recorder = RuntimeRecorder(tracker)
        self.network_retry = network_retry
        self.llm_retry = llm_retry
        self.search_timeout = search_timeout
        self.crawl_timeout = crawl_timeout
        self.overall_deadline = overall_deadline
        self.stage_callback = stage_callback
        self.stage_durations: dict[str, float] = {}

    def set_stage(self, stage: RunStage) -> None:
        self._current_stage = stage
        if self.stage_callback:
            self.stage_callback(stage)

    def consume(self, resource: str, count: int = 1) -> None:
        self.tracker.consume(resource, count)

    async def run_async(
        self,
        resource: str,
        operation_name: str,
        operation: Callable[[], Awaitable[ResultT]],
        retryable: Callable[[BaseException], bool] | None = None,
    ) -> ResultT:
        stage = RunStage.SEARCH if resource == "search" else RunStage.READING
        self.set_stage(stage)
        timeout = (
            self.search_timeout if resource == "search" else self.crawl_timeout
        )

        async def timed_operation() -> ResultT:
            started = time.perf_counter()
            try:
                async with asyncio.timeout(timeout):
                    return await operation()
            except asyncio.CancelledError:
                raise
            except TimeoutError as exc:
                self.recorder.timeout()
                timeout_error = (
                    SearchTimeoutError
                    if resource == "search"
                    else CrawlerTimeoutError
                )
                logger.warning(
                    "operation=%s operation_timeout=%.3gs",
                    operation_name,
                    timeout,
                    extra={"run_id": self.run_id, "stage": stage.value},
                )
                raise timeout_error(
                    f"{resource} operation timed out after {timeout:.3g}s"
                ) from exc
            finally:
                self._record_duration(stage, time.perf_counter() - started)

        return await retry_async(
            timed_operation,
            self.network_retry,
            retryable or is_retryable_exception,
            before_attempt=lambda: self.consume(resource),
            on_retry=lambda attempt, exc, delay: self._log_retry(
                resource, operation_name, attempt, self.network_retry, exc, delay
            ),
        )

    def run_sync(
        self,
        resource: str,
        operation_name: str,
        operation: Callable[[], ResultT],
        retryable: Callable[[BaseException], bool] | None = None,
    ) -> ResultT:
        started = time.perf_counter()

        def tracked_operation() -> ResultT:
            try:
                return operation()
            except BaseException as exc:
                if is_timeout_exception(exc):
                    self.recorder.timeout()
                    logger.warning(
                        "operation=%s model_timeout=true",
                        operation_name,
                        extra={
                            "run_id": self.run_id,
                            "stage": getattr(
                                self, "_current_stage", RunStage.RUNTIME
                            ).value,
                        },
                    )
                    raise ModelTimeoutError(
                        f"LLM operation timed out: {operation_name}"
                    ) from exc
                raise

        classifier = retryable or is_retryable_exception

        def can_retry(error: BaseException) -> bool:
            has_time = (
                self.overall_deadline is None
                or time.perf_counter() < self.overall_deadline
            )
            return has_time and classifier(error)

        def before_attempt() -> None:
            if (
                self.overall_deadline is not None
                and time.perf_counter() >= self.overall_deadline
            ):
                raise TimeoutError("overall run deadline reached before LLM retry")
            self.consume(resource)

        def sleep_with_deadline(delay: float) -> None:
            if self.overall_deadline is None:
                time.sleep(delay)
                return
            remaining = max(0.0, self.overall_deadline - time.perf_counter())
            time.sleep(min(delay, remaining))

        try:
            return retry_sync(
                tracked_operation,
                self.llm_retry,
                can_retry,
                before_attempt=before_attempt,
                on_retry=lambda attempt, exc, delay: self._log_retry(
                    getattr(self, "_current_stage", RunStage.RUNTIME).value,
                    operation_name,
                    attempt,
                    self.llm_retry,
                    exc,
                    delay,
                ),
                sleeper=sleep_with_deadline,
            )
        finally:
            # The orchestrator announces whether an LLM call belongs to planning,
            # evaluation, grounding, or rendering.
            stage = getattr(self, "_current_stage", RunStage.RUNTIME)
            self._record_duration(stage, time.perf_counter() - started)

    def _record_duration(self, stage: RunStage, duration: float) -> None:
        key = f"runtime.{stage.value}"
        self.stage_durations[key] = self.stage_durations.get(key, 0.0) + duration

    def _log_retry(
        self,
        stage: str,
        operation: str,
        attempt: int,
        policy: RetryPolicy,
        error: BaseException,
        delay: float,
    ) -> None:
        self.recorder.retry()
        logger.warning(
            "operation=%s attempt=%d/%d error=%s "
            "retry_delay=%.3fs",
            operation,
            attempt + 1,
            policy.max_attempts,
            f"{error.__class__.__name__}: {error}",
            delay,
            extra={"run_id": self.run_id, "stage": stage},
        )


_CURRENT_RUNTIME: ContextVar[RuntimeExecutionContext | None] = ContextVar(
    "tracker_runtime", default=None
)


def activate_runtime(context: RuntimeExecutionContext) -> Token:
    return _CURRENT_RUNTIME.set(context)


def deactivate_runtime(token: Token) -> None:
    _CURRENT_RUNTIME.reset(token)


def current_runtime() -> RuntimeExecutionContext | None:
    return _CURRENT_RUNTIME.get()


def effective_runtime_timeout(configured_timeout: float) -> float:
    """Clamp a blocking transport timeout to the active run's remaining time."""
    context = current_runtime()
    if context is None or context.overall_deadline is None:
        return configured_timeout
    remaining = context.overall_deadline - time.perf_counter()
    return max(0.001, min(configured_timeout, remaining))


def consume_runtime_budget(resource: str, count: int = 1) -> None:
    context = current_runtime()
    if context:
        context.consume(resource, count)


def set_runtime_stage(stage: RunStage) -> None:
    context = current_runtime()
    if context:
        context.set_stage(stage)


class RuntimeLogFilter(logging.Filter):
    """Inject correlation fields without changing every existing log call."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = current_runtime()
        if not hasattr(record, "run_id"):
            record.run_id = context.run_id if context else "-"
        if not hasattr(record, "stage"):
            stage = getattr(context, "_current_stage", RunStage.RUNTIME)
            record.stage = stage.value
        return True


async def run_async_operation(
    resource: str,
    operation_name: str,
    operation: Callable[[], Awaitable[ResultT]],
    retryable: Callable[[BaseException], bool] | None = None,
) -> ResultT:
    context = current_runtime()
    if context is None:
        return await operation()
    return await context.run_async(resource, operation_name, operation, retryable)


def run_sync_operation(
    resource: str,
    operation_name: str,
    operation: Callable[[], ResultT],
    retryable: Callable[[BaseException], bool] | None = None,
) -> ResultT:
    context = current_runtime()
    if context is None:
        return operation()
    return context.run_sync(resource, operation_name, operation, retryable)


def is_retryable_exception(error: BaseException) -> bool:
    """Classify temporary transport/rate-limit failures, including wrapped causes."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (
                RetryableError,
                TimeoutError,
                httpx.TimeoutException,
                httpx.TransportError,
            ),
        ):
            return True
        if isinstance(current, httpx.HTTPStatusError):
            status = current.response.status_code
            return status == 429 or status in {502, 503, 504}
        status = getattr(current, "status_code", None)
        if status == 429 or status in {502, 503, 504}:
            return True
        current = current.__cause__ or current.__context__
    return False


def is_timeout_exception(error: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, httpx.TimeoutException)):
            return True
        current = current.__cause__ or current.__context__
    return False
