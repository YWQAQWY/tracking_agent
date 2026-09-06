"""Evidence-driven adaptive research loop."""

from src.agent.critic import CriticError, EvidenceCritic
from src.agent.models import (
    CriticResult,
    ResearchResult,
    ResearchRoundTrace,
    ResearchTrace,
)
from src.agent.research_agent import ResearchAgent, ResearchAgentError
from src.memory import EvidencePool, ResearchResumeState, ResearchState
from src.tools import ResearchTool, ResearchToolResult

ResearchRound = ResearchTool
ResearchRoundResult = ResearchToolResult

__all__ = [
    "CriticError",
    "CriticResult",
    "EvidenceCritic",
    "EvidencePool",
    "ResearchAgent",
    "ResearchAgentError",
    "ResearchResult",
    "ResearchResumeState",
    "ResearchRound",
    "ResearchRoundResult",
    "ResearchRoundTrace",
    "ResearchState",
    "ResearchTrace",
    "ResearchTool",
    "ResearchToolResult",
]
