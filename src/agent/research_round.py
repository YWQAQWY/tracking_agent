"""Compatibility aliases for the canonical Research Tool."""

from src.tools.research import ResearchTool, ResearchToolResult

ResearchRound = ResearchTool
ResearchRoundResult = ResearchToolResult

__all__ = ["ResearchRound", "ResearchRoundResult"]
