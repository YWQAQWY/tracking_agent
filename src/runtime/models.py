"""Lifecycle, budget, trace, and structured result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from src.runtime.retry import RetryPolicy

if TYPE_CHECKING:
    from src.agent.models import ResearchResult


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    BUDGET_EXCEEDED = "budget_exceeded"


class RunStage(str, Enum):
    RUNTIME = "runtime"
    PLANNING = "planning"
    SEARCH = "search"
    READING = "reading"
    RETRIEVAL = "retrieval"
    EVALUATION = "evaluation"
    GROUNDING = "grounding"
    RENDERING = "rendering"


@dataclass(frozen=True, slots=True)
class RunRequest:
    question: str
    run_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimeBudget:
    max_research_rounds: int = 3
    max_search_requests: int = 30
    max_crawl_requests: int = 30
    max_llm_calls: int = 40

    def __post_init__(self) -> None:
        if min(
            self.max_research_rounds,
            self.max_search_requests,
            self.max_crawl_requests,
            self.max_llm_calls,
        ) < 1:
            raise ValueError("runtime budgets 必须大于 0")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    overall_timeout_seconds: float = 900.0
    search_timeout_seconds: float = 15.0
    crawl_timeout_seconds: float = 15.0
    budget: RuntimeBudget = field(default_factory=RuntimeBudget)
    network_retry: RetryPolicy = field(default_factory=RetryPolicy)
    llm_retry: RetryPolicy = field(
        default_factory=lambda: RetryPolicy(
            max_attempts=2,
            base_delay_seconds=1.0,
            max_delay_seconds=4.0,
            jitter_seconds=0.1,
        )
    )
    checkpoint_enabled: bool = True
    checkpoint_dir: str = ".runtime/runs"

    def __post_init__(self) -> None:
        if min(
            self.overall_timeout_seconds,
            self.search_timeout_seconds,
            self.crawl_timeout_seconds,
        ) <= 0:
            raise ValueError("runtime timeouts 必须大于 0")


@dataclass(slots=True)
class RuntimeCounters:
    research_rounds: int = 0
    search_requests: int = 0
    crawl_requests: int = 0
    llm_calls: int = 0
    retries: int = 0
    timeouts: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "research_rounds": self.research_rounds,
            "search_requests": self.search_requests,
            "crawl_requests": self.crawl_requests,
            "llm_calls": self.llm_calls,
            "retries": self.retries,
            "timeouts": self.timeouts,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "RuntimeCounters":
        fields = cls.__dataclass_fields__
        return cls(**{name: int(value.get(name, 0)) for name in fields})


@dataclass(slots=True)
class RuntimeState:
    run_id: str
    question: str
    status: RunStatus = RunStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    termination_reason: str | None = None
    current_stage: RunStage = RunStage.RUNTIME
    current_research_round: int = 0
    counters: RuntimeCounters = field(default_factory=RuntimeCounters)
    status_history: list[RunStatus] = field(
        default_factory=lambda: [RunStatus.PENDING]
    )

    def transition(self, status: RunStatus) -> None:
        self.status = status
        if not self.status_history or self.status_history[-1] != status:
            self.status_history.append(status)


@dataclass(frozen=True, slots=True)
class RuntimeErrorInfo:
    error_type: str
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class RuntimeTrace:
    run_id: str
    question: str
    status: RunStatus
    started_at: datetime
    finished_at: datetime
    total_duration: float
    counters: RuntimeCounters
    termination_reason: str
    budget_limits: dict[str, int] = field(default_factory=dict)
    stage_durations: dict[str, float] = field(default_factory=dict)
    status_history: tuple[RunStatus, ...] = ()
    checkpoint_path: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    run_id: str
    status: RunStatus
    research_result: ResearchResult | None
    error: RuntimeErrorInfo | None
    trace: RuntimeTrace
    termination_reason: str

    @property
    def answer(self) -> str | None:
        return self.research_result.answer if self.research_result else None

    @property
    def succeeded(self) -> bool:
        return self.status is RunStatus.SUCCEEDED
