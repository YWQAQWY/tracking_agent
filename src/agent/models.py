"""State, decisions, and inspectable traces for the research agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.agent.evidence_pool import EvidencePool
from src.models.evidence import Evidence
from src.models.search_plan import SearchPlan

if TYPE_CHECKING:
    from src.agent.research_round import ResearchRoundResult


class CriticResult(BaseModel):
    """Structured evidence-sufficiency decision returned by the local LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sufficient: bool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    missing_aspects: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)
    reason: str

    @field_validator("missing_aspects", "follow_up_queries")
    @classmethod
    def normalize_lists(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = value.strip()
            if text and text not in seen:
                seen.add(text)
                normalized.append(text)
        return normalized

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Critic reason 不能为空")
        return text


@dataclass(slots=True)
class ResearchState:
    """Mutable task state that makes decisions aware of earlier rounds."""

    question: str
    round_index: int = 0
    evidence_pool: EvidencePool = field(default_factory=EvidencePool)
    executed_queries: list[str] = field(default_factory=list)


StopReason = Literal[
    "sufficient",
    "max_rounds",
    "no_follow_up_queries",
    "duplicate_queries",
    "no_new_evidence",
    "critic_failure",
]


@dataclass(frozen=True, slots=True)
class ResearchRoundTrace:
    """One observe/evaluate/decide cycle in an agent run."""

    round_index: int
    queries: tuple[str, ...]
    search_result_count: int
    new_evidence_count: int
    total_evidence_count: int
    critic_sufficient: bool | None
    missing_aspects: tuple[str, ...] = ()
    follow_up_queries: tuple[str, ...] = ()
    critic_reason: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResearchTrace:
    """Why the agent searched, continued, and eventually stopped."""

    rounds: tuple[ResearchRoundTrace, ...]
    stop_reason: StopReason
    timings: dict[str, float] = field(default_factory=dict)
    critic_error: str | None = None


@dataclass(frozen=True, slots=True)
class ResearchResult:
    """Final answer plus complete evidence and agent execution trace."""

    plan: SearchPlan
    rounds: tuple[ResearchRoundResult, ...]
    pooled_evidence: tuple[Evidence, ...]
    evidence: tuple[Evidence, ...]
    research_trace: ResearchTrace
    answer: str
