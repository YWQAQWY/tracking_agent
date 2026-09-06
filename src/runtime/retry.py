"""Bounded retry helpers with exponential backoff and optional jitter."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar


ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 2
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 2.0
    jitter_seconds: float = 0.1

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须大于 0")
        if min(
            self.base_delay_seconds,
            self.max_delay_seconds,
            self.jitter_seconds,
        ) < 0:
            raise ValueError("retry delay 不能小于 0")

    def delay(self, failed_attempt: int, random_value: float | None = None) -> float:
        base = min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2 ** (failed_attempt - 1)),
        )
        jitter = self.jitter_seconds * (
            random.random() if random_value is None else random_value
        )
        return min(self.max_delay_seconds, base + jitter)


async def retry_async(
    operation: Callable[[], Awaitable[ResultT]],
    policy: RetryPolicy,
    is_retryable: Callable[[BaseException], bool],
    before_attempt: Callable[[], None] | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ResultT:
    for attempt in range(1, policy.max_attempts + 1):
        if before_attempt:
            before_attempt()
        try:
            return await operation()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise
            delay = policy.delay(attempt)
            if on_retry:
                on_retry(attempt, exc, delay)
            await sleeper(delay)
    raise AssertionError("retry loop exhausted unexpectedly")


def retry_sync(
    operation: Callable[[], ResultT],
    policy: RetryPolicy,
    is_retryable: Callable[[BaseException], bool],
    before_attempt: Callable[[], None] | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> ResultT:
    for attempt in range(1, policy.max_attempts + 1):
        if before_attempt:
            before_attempt()
        try:
            return operation()
        except BaseException as exc:
            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise
            delay = policy.delay(attempt)
            if on_retry:
                on_retry(attempt, exc, delay)
            sleeper(delay)
    raise AssertionError("retry loop exhausted unexpectedly")
