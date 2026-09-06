"""Reliable single-machine execution runtime for Tracker."""

from src.runtime.budgets import BudgetTracker
from src.runtime.checkpoint import CheckpointStore
from src.runtime.errors import (
    BudgetExceededError,
    CheckpointCorruptedError,
    CheckpointError,
    CrawlerTimeoutError,
    ModelTimeoutError,
    NonRetryableError,
    OperationTimeoutError,
    RetryableError,
    SearchTimeoutError,
    TrackerRuntimeError,
)
from src.runtime.models import (
    RunRequest,
    RunStage,
    RunStatus,
    RuntimeBudget,
    RuntimeConfig,
    RuntimeCounters,
    RuntimeResult,
    RuntimeState,
    RuntimeTrace,
)
from src.runtime.retry import RetryPolicy, retry_async, retry_sync

__all__ = [
    "AgentRuntime",
    "BudgetExceededError",
    "BudgetTracker",
    "CheckpointCorruptedError",
    "CheckpointError",
    "CheckpointStore",
    "CrawlerTimeoutError",
    "ModelTimeoutError",
    "NonRetryableError",
    "OperationTimeoutError",
    "RetryPolicy",
    "RetryableError",
    "RunRequest",
    "RunStage",
    "RunStatus",
    "SearchTimeoutError",
    "RuntimeBudget",
    "RuntimeConfig",
    "RuntimeCounters",
    "RuntimeResult",
    "RuntimeState",
    "RuntimeTrace",
    "TrackerRuntimeError",
    "retry_async",
    "retry_sync",
]


def __getattr__(name: str):
    # Runtime context is imported by low-level tools. Delay the orchestrator
    # import so LLMClient → runtime.context cannot loop back through ResearchAgent.
    if name == "AgentRuntime":
        from src.runtime.runtime import AgentRuntime

        return AgentRuntime
    raise AttributeError(name)
