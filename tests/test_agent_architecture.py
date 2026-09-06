from src.action import ResearchActionPolicy, sanitize_queries
from src.agent.evidence_pool import EvidencePool as LegacyEvidencePool
from src.agent.research_round import ResearchRound, ResearchRoundResult
from src.memory import EvidencePool, ResearchResumeState, ResearchState
from src.models.search_plan import SearchPlan as LegacySearchPlan
from src.plan import SearchPlan, SearchPlanner
from src.planner.search_planner import SearchPlanner as LegacySearchPlanner
from src.tools import ResearchTool, ResearchToolResult


def test_canonical_layers_do_not_duplicate_legacy_implementations() -> None:
    assert LegacyEvidencePool is EvidencePool
    assert LegacySearchPlan is SearchPlan
    assert LegacySearchPlanner is SearchPlanner
    assert ResearchRound is ResearchTool
    assert ResearchRoundResult is ResearchToolResult


def test_action_policy_selects_targeted_search() -> None:
    policy = ResearchActionPolicy(max_rounds=3, max_followup_queries=2)

    action = policy.decide(
        sufficient=False,
        follow_up_queries=[" done ", "next", "next", "last"],
        executed_queries=["done"],
        round_index=1,
    )

    assert action.kind == "search"
    assert action.queries == ("next", "last")
    assert action.stop_reason is None


def test_action_policy_applies_stop_conditions() -> None:
    policy = ResearchActionPolicy(max_rounds=2, max_followup_queries=2)

    assert policy.decide(
        sufficient=True,
        follow_up_queries=["unused"],
        executed_queries=[],
        round_index=1,
    ).stop_reason == "sufficient"
    assert policy.decide(
        sufficient=False,
        follow_up_queries=["unused"],
        executed_queries=[],
        round_index=2,
    ).stop_reason == "max_rounds"
    assert policy.decide(
        sufficient=False,
        follow_up_queries=[],
        executed_queries=[],
        round_index=1,
    ).stop_reason == "no_follow_up_queries"
    assert policy.decide(
        sufficient=False,
        follow_up_queries=["done"],
        executed_queries=["done"],
        round_index=1,
    ).stop_reason == "duplicate_queries"


def test_memory_and_action_public_api_is_importable() -> None:
    state = ResearchState(question="q")
    resume = ResearchResumeState(
        question="q",
        plan=SearchPlan(queries=["query"]),
        round_index=0,
        executed_queries=(),
        evidence=(),
        next_queries=("query",),
    )

    assert isinstance(state.evidence_pool, EvidencePool)
    assert resume.next_queries == ("query",)
    assert sanitize_queries([" q ", "q", "x"], [], 2) == ["q", "x"]
