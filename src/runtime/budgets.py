"""Pre-call budget enforcement for bounded agent execution."""

from src.runtime.errors import BudgetExceededError
from src.runtime.models import RuntimeBudget, RuntimeCounters


class BudgetTracker:
    """Consume counters before work starts so over-budget calls never launch."""

    _FIELDS = {
        "research_round": ("research_rounds", "max_research_rounds"),
        "search": ("search_requests", "max_search_requests"),
        "crawl": ("crawl_requests", "max_crawl_requests"),
        "llm": ("llm_calls", "max_llm_calls"),
    }

    def __init__(
        self,
        budget: RuntimeBudget,
        counters: RuntimeCounters | None = None,
    ) -> None:
        self.budget = budget
        self.counters = counters or RuntimeCounters()

    def consume(self, resource: str, count: int = 1) -> None:
        if count < 1:
            raise ValueError("budget count 必须大于 0")
        try:
            counter_name, limit_name = self._FIELDS[resource]
        except KeyError as exc:
            raise ValueError(f"未知 runtime budget resource: {resource}") from exc
        current = getattr(self.counters, counter_name)
        limit = getattr(self.budget, limit_name)
        if current + count > limit:
            raise BudgetExceededError(resource, limit)
        setattr(self.counters, counter_name, current + count)
