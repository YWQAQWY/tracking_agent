"""Mutable and resumable memory for one research task."""

from dataclasses import dataclass, field

from src.action.models import StopReason
from src.memory.evidence_pool import EvidencePool
from src.models.evidence import Evidence
from src.plan.models import SearchPlan


@dataclass(slots=True)
class ResearchState:
    """Working memory that makes each round aware of earlier rounds."""

    question: str
    round_index: int = 0
    evidence_pool: EvidencePool = field(default_factory=EvidencePool)
    executed_queries: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ResearchResumeState:
    """Serializable stage boundary used to resume a research run."""

    question: str
    plan: SearchPlan
    round_index: int
    executed_queries: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    next_queries: tuple[str, ...]
    research_complete: bool = False
    stop_reason: StopReason | None = None
