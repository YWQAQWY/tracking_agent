"""Compatibility package; use :mod:`src.plan` for new code."""

from src.plan.search_planner import PlannerError, SearchPlanner

__all__ = ["PlannerError", "SearchPlanner"]
