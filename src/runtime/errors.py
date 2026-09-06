"""Small runtime error taxonomy used for retry and terminal status decisions."""

from __future__ import annotations


class TrackerRuntimeError(RuntimeError):
    """Base class for expected runtime failures."""


class RetryableError(TrackerRuntimeError):
    """A temporary failure for which repeating the same operation may help."""


class NonRetryableError(TrackerRuntimeError):
    """A deterministic failure that must not be retried."""


class OperationTimeoutError(RetryableError):
    """One external operation exceeded its timeout."""


class SearchTimeoutError(OperationTimeoutError):
    """One search request exceeded its operation timeout."""


class CrawlerTimeoutError(OperationTimeoutError):
    """One page fetch exceeded its operation timeout."""


class ModelTimeoutError(OperationTimeoutError):
    """One local-model request exceeded its transport timeout."""


class BudgetExceededError(NonRetryableError):
    """A tool-call budget would be exceeded before the next operation."""

    def __init__(self, resource: str, limit: int) -> None:
        self.resource = resource
        self.limit = limit
        super().__init__(f"{resource} budget exceeded (limit={limit})")


class CheckpointError(NonRetryableError):
    """A checkpoint cannot be saved or loaded safely."""


class CheckpointCorruptedError(CheckpointError):
    """A checkpoint exists but is malformed or fails validation."""
