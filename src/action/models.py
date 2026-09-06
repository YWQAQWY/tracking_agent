"""Explicit actions and stop decisions emitted by the agent policy."""

from dataclasses import dataclass
from typing import Literal


StopReason = Literal[
    "sufficient",
    "max_rounds",
    "no_follow_up_queries",
    "duplicate_queries",
    "no_new_evidence",
    "critic_failure",
]
STOP_REASONS: frozenset[str] = frozenset(
    {
        "sufficient",
        "max_rounds",
        "no_follow_up_queries",
        "duplicate_queries",
        "no_new_evidence",
        "critic_failure",
    }
)
ActionKind = Literal["search", "finish"]


@dataclass(frozen=True, slots=True)
class ResearchAction:
    """One inspectable next action selected after evidence evaluation."""

    kind: ActionKind
    queries: tuple[str, ...] = ()
    stop_reason: StopReason | None = None

    def __post_init__(self) -> None:
        if self.kind == "search" and (not self.queries or self.stop_reason is not None):
            raise ValueError("search action 必须包含 queries 且不能包含 stop_reason")
        if self.kind == "finish" and (self.queries or self.stop_reason is None):
            raise ValueError("finish action 必须包含 stop_reason 且不能包含 queries")
