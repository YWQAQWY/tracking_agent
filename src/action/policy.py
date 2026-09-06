"""Deterministic policy translating Critic output into the next action."""

from src.action.models import ResearchAction


def sanitize_queries(
    queries: list[str], executed_queries: list[str], limit: int
) -> list[str]:
    """Strip, exact-deduplicate, remove executed queries, and cap output."""
    if limit < 1:
        raise ValueError("query limit 必须大于 0")
    executed = set(executed_queries)
    accepted: list[str] = []
    seen: set[str] = set()
    for query in queries:
        clean = query.strip()
        if not clean or clean in executed or clean in seen:
            continue
        seen.add(clean)
        accepted.append(clean)
        if len(accepted) >= limit:
            break
    return accepted


class ResearchActionPolicy:
    """Select Search or Finish without performing either operation."""

    def __init__(self, max_rounds: int, max_followup_queries: int) -> None:
        if min(max_rounds, max_followup_queries) < 1:
            raise ValueError("action policy limits 必须大于 0")
        self.max_rounds = max_rounds
        self.max_followup_queries = max_followup_queries

    def decide(
        self,
        *,
        sufficient: bool,
        follow_up_queries: list[str],
        executed_queries: list[str],
        round_index: int,
    ) -> ResearchAction:
        """Apply the bounded-loop stop rules in a stable priority order."""
        if sufficient:
            return ResearchAction("finish", stop_reason="sufficient")
        if round_index >= self.max_rounds:
            return ResearchAction("finish", stop_reason="max_rounds")
        if not follow_up_queries:
            return ResearchAction("finish", stop_reason="no_follow_up_queries")

        queries = sanitize_queries(
            follow_up_queries,
            executed_queries,
            self.max_followup_queries,
        )
        if not queries:
            return ResearchAction("finish", stop_reason="duplicate_queries")
        return ResearchAction("search", queries=tuple(queries))
