import asyncio
from collections.abc import Sequence

import pytest

from src.agent.models import CriticResult, ResearchResumeState
from src.agent.research_agent import ResearchAgent
from src.agent.research_round import ResearchRoundResult
from src.grounding.generator import GroundedGenerationError
from src.grounding.models import CitationSource, GroundedAnswer, GroundingTrace
from src.models.evidence import Evidence, ScoredChunk
from src.models.search_plan import SearchPlan
from src.retrieval.trace import RetrievalTrace
from src.search.source_manager import SearchBatch


QUESTION = "Original research question"


def make_evidence(name: str, url: str | None = None) -> Evidence:
    return Evidence(
        text=f"Evidence {name}",
        url=url or f"https://{name.lower()}.example",
        title=name,
        chunk_index=0,
        embedding_score=0.8,
        rerank_score=0.9,
    )


def round_result(queries: list[str], items: list[Evidence]) -> ResearchRoundResult:
    batch = SearchBatch(
        results=(), coverage=(), providers=("fake",), raw_result_count=len(items),
        duplicate_count=0,
    )
    trace = RetrievalTrace(
        raw_search_results=len(items), combined_search_results=len(items),
        domain_filtered_results=len(items), unique_urls=len(items),
        documents=len(items), unique_documents=len(items), chunks=len(items),
        embedding_candidates=len(items), final_evidence=len(items),
        evidence_source_count=len({item.url for item in items}),
        embedding_model="fake-embedder", reranker_model="fake-reranker",
        device="cpu", timings={"search": 0.1, "rerank": 0.1, "total": 0.2},
    )
    return ResearchRoundResult(
        queries=tuple(queries), search_batch=batch, search_results=(), documents=(),
        embedding_candidates=(), evidence=tuple(items), retrieval_trace=trace,
    )


class FakePlanner:
    def plan(self, question: str) -> SearchPlan:
        assert question == QUESTION
        return SearchPlan(queries=["query a"])


class FakeRound:
    def __init__(self, rounds: Sequence[list[Evidence]]) -> None:
        self.rounds = list(rounds)
        self.calls: list[tuple[str, list[str]]] = []

    async def run(self, question: str, queries: list[str]) -> ResearchRoundResult:
        self.calls.append((question, list(queries)))
        return round_result(queries, self.rounds.pop(0))


class FakeCritic:
    def __init__(self, outcomes: Sequence[CriticResult | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, list[Evidence], list[str]]] = []

    def evaluate(
        self, question: str, evidence: list[Evidence], queries: list[str]
    ) -> CriticResult:
        self.calls.append((question, evidence, queries))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeFinalReranker:
    model_name = "fake-final"

    def __init__(self, returned: int = 5) -> None:
        self.returned = returned
        self.calls: list[tuple[str, list[ScoredChunk], int | None]] = []

    def rerank(
        self, question: str, candidates: list[ScoredChunk], top_k: int | None = None
    ) -> list[Evidence]:
        self.calls.append((question, candidates, top_k))
        return [
            Evidence.from_scored_chunk(candidate, 1.0)
            for candidate in candidates[: min(self.returned, top_k or self.returned)]
        ]


class FakeContext:
    def __init__(self) -> None:
        self.received: list[Evidence] = []

    def build(self, evidence: list[Evidence]) -> str:
        self.received = list(evidence)
        return "\n".join(item.text for item in evidence)


class FakeAnswer:
    def __init__(self) -> None:
        self.question = ""

    def generate(self, question: str, context: str) -> str:
        self.question = question
        assert context
        return "final answer"


def critique(
    sufficient: bool,
    queries: list[str] | None = None,
    missing: list[str] | None = None,
) -> CriticResult:
    return CriticResult(
        sufficient=sufficient,
        missing_aspects=missing or [],
        follow_up_queries=queries or [],
        reason="covered" if sufficient else "more evidence needed",
    )


def build_agent(
    rounds: Sequence[list[Evidence]],
    outcomes: Sequence[CriticResult | Exception],
    *,
    max_rounds: int = 3,
    final_count: int = 5,
    max_followups: int = 3,
) -> tuple[ResearchAgent, FakeRound, FakeCritic, FakeFinalReranker, FakeContext, FakeAnswer]:
    round_runner = FakeRound(rounds)
    critic = FakeCritic(outcomes)
    reranker = FakeFinalReranker(final_count)
    context = FakeContext()
    answer = FakeAnswer()
    agent = ResearchAgent(
        planner=FakePlanner(), research_round=round_runner, critic=critic,
        final_reranker=reranker, context_builder=context,
        answer_generator=answer, max_rounds=max_rounds,
        max_followup_queries_per_round=max_followups, final_evidence_top_k=5,
    )
    return agent, round_runner, critic, reranker, context, answer


def test_two_round_agent_accumulates_evidence_and_answers_original_question() -> None:
    agent, rounds, critic, reranker, context, answer = build_agent(
        [[make_evidence("A"), make_evidence("B")], [make_evidence("C")]],
        [
            critique(False, ["query b"], ["limitations"]),
            critique(True),
        ],
    )
    result = asyncio.run(agent.run(QUESTION))

    assert len(rounds.calls) == 2
    assert rounds.calls[1] == (QUESTION, ["query b"])
    assert [item.title for item in result.pooled_evidence] == ["A", "B", "C"]
    assert [len(call[1]) for call in critic.calls] == [2, 3]
    assert critic.calls[0][2] == ["query a"]
    assert critic.calls[1][2] == ["query a", "query b"]
    assert reranker.calls[0][0] == QUESTION
    assert answer.question == QUESTION
    assert context.received == list(result.evidence)
    assert result.research_trace.stop_reason == "sufficient"


def test_max_rounds_stops_an_always_insufficient_critic() -> None:
    agent, rounds, _, _, _, _ = build_agent(
        [[make_evidence("A")], [make_evidence("B")]],
        [critique(False, ["query b"]), critique(False, ["query c"])],
        max_rounds=2,
    )
    result = asyncio.run(agent.run(QUESTION))
    assert len(rounds.calls) == 2
    assert result.research_trace.stop_reason == "max_rounds"


def test_no_follow_up_queries_stops_after_first_round() -> None:
    agent, rounds, _, _, _, _ = build_agent(
        [[make_evidence("A")]], [critique(False)]
    )
    result = asyncio.run(agent.run(QUESTION))
    assert len(rounds.calls) == 1
    assert result.research_trace.stop_reason == "no_follow_up_queries"


def test_all_follow_up_queries_already_executed_stops() -> None:
    agent, rounds, _, _, _, _ = build_agent(
        [[make_evidence("A")]], [critique(False, ["query a", "query a"])]
    )
    result = asyncio.run(agent.run(QUESTION))
    assert len(rounds.calls) == 1
    assert result.research_trace.stop_reason == "duplicate_queries"


def test_no_new_evidence_stops_before_second_critique() -> None:
    agent, rounds, critic, _, _, _ = build_agent(
        [[make_evidence("A")], []], [critique(False, ["query b"])]
    )
    result = asyncio.run(agent.run(QUESTION))
    assert len(rounds.calls) == 2
    assert len(critic.calls) == 1
    assert result.research_trace.stop_reason == "no_new_evidence"
    assert result.research_trace.rounds[1].new_evidence_count == 0


@pytest.mark.parametrize("error", [RuntimeError("critic down"), ValueError("parse")])
def test_critic_failure_gracefully_uses_existing_evidence(error: Exception) -> None:
    agent, _, _, _, context, _ = build_agent(
        [[make_evidence("A")]], [error]
    )
    result = asyncio.run(agent.run(QUESTION))
    assert result.answer == "final answer"
    assert context.received
    assert result.research_trace.stop_reason == "critic_failure"
    assert result.research_trace.critic_error is not None


def test_query_sanitization_removes_duplicates_executed_and_caps() -> None:
    assert ResearchAgent.sanitize_queries(
        [" query a ", "query b", "query b", "query c", "query d"],
        ["query a"],
        2,
    ) == ["query b", "query c"]


def test_duplicate_evidence_across_rounds_is_not_accumulated() -> None:
    duplicate = make_evidence("A")
    agent, _, _, _, _, _ = build_agent(
        [[duplicate], [duplicate, make_evidence("B")]],
        [critique(False, ["query b"]), critique(True)],
    )
    result = asyncio.run(agent.run(QUESTION))
    assert [item.title for item in result.pooled_evidence] == ["A", "B"]
    assert result.research_trace.rounds[1].new_evidence_count == 1


def test_final_reranker_limits_context_to_selected_evidence() -> None:
    items = [make_evidence(str(index)) for index in range(10)]
    agent, _, _, reranker, context, _ = build_agent(
        [items], [critique(True)], final_count=5
    )
    result = asyncio.run(agent.run(QUESTION))
    assert len(result.pooled_evidence) == 10
    assert len(result.evidence) == 5
    assert len(context.received) == 5
    assert reranker.calls[0][0] == QUESTION
    assert reranker.calls[0][2] == 5


def test_trace_records_queries_gaps_counts_and_stop_reason() -> None:
    agent, _, _, _, _, _ = build_agent(
        [[make_evidence("A")]],
        [critique(True, missing=[])],
    )
    trace = asyncio.run(agent.run(QUESTION)).research_trace
    assert trace.stop_reason == "sufficient"
    assert trace.rounds[0].queries == ("query a",)
    assert trace.rounds[0].new_evidence_count == 1
    assert trace.rounds[0].total_evidence_count == 1
    assert trace.rounds[0].action == "finish"
    assert trace.rounds[0].action_stop_reason == "sufficient"
    assert set(trace.timings) == {
        "planning", "search", "crawl_extract", "retrieval", "critic",
        "final_rerank", "context", "generation", "total",
    }


class FakeGroundedGenerator:
    def __init__(self, response: GroundedAnswer | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, list[Evidence]]] = []

    def generate(self, question: str, evidence: list[Evidence]) -> GroundedAnswer:
        self.calls.append((question, evidence))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def empty_grounding_trace() -> GroundingTrace:
    return GroundingTrace(
        draft_claim_count=1,
        initially_supported_claim_count=1,
        unsupported_claim_count=0,
        rewritten_claim_count=0,
        rewrite_passed_claim_count=0,
        dropped_claim_count=0,
        verified_claim_count=1,
        cited_source_count=1,
    )


def test_research_agent_uses_grounded_generator_and_skips_legacy_context() -> None:
    agent, _, _, _, context, legacy = build_agent(
        [[make_evidence("A")]], [critique(True)]
    )
    grounded = FakeGroundedGenerator(
        GroundedAnswer(
            text="grounded answer [1]",
            sources=(CitationSource(1, "https://a.example", "A", ("E1",), (0,)),),
            grounding_trace=empty_grounding_trace(),
        )
    )
    agent.grounded_answer_generator = grounded
    result = asyncio.run(agent.run(QUESTION))
    assert result.answer == "grounded answer [1]"
    assert result.grounding_verified is True
    assert len(result.sources) == 1
    assert context.received == []
    assert legacy.question == ""
    assert grounded.calls[0][0] == QUESTION


def test_grounding_failure_falls_back_to_v05_answer_and_marks_result() -> None:
    agent, _, _, _, context, _ = build_agent(
        [[make_evidence("A")]], [critique(True)]
    )
    agent.grounded_answer_generator = FakeGroundedGenerator(
        GroundedGenerationError("planner parse failed")
    )
    result = asyncio.run(agent.run(QUESTION))
    assert result.answer == "final answer"
    assert context.received
    assert result.grounding_verified is False
    assert result.grounding_trace is not None
    assert "planner parse failed" in (result.grounding_trace.fallback_reason or "")


def test_resume_continues_with_saved_queries_and_evidence_without_replanning() -> None:
    saved = make_evidence("A")
    agent, rounds, critic, _, _, _ = build_agent(
        [[saved, make_evidence("B")]], [critique(True)]
    )

    class PlannerMustNotRun:
        def plan(self, question: str) -> SearchPlan:
            raise AssertionError("resume must not plan again")

    agent.planner = PlannerMustNotRun()
    checkpoints: list[ResearchResumeState] = []
    result = asyncio.run(
        agent.run(
            QUESTION,
            resume_state=ResearchResumeState(
                question=QUESTION,
                plan=SearchPlan(queries=["query a"]),
                round_index=1,
                executed_queries=("query a",),
                evidence=(saved,),
                next_queries=("query b",),
            ),
            checkpoint_callback=checkpoints.append,
        )
    )

    assert rounds.calls == [(QUESTION, ["query b"])]
    assert [item.title for item in result.pooled_evidence] == ["A", "B"]
    assert critic.calls[0][2] == ["query a", "query b"]
    assert checkpoints[-1].research_complete is True
    assert checkpoints[-1].round_index == 2
    assert checkpoints[-1].executed_queries == ("query a", "query b")
    assert checkpoints[-1].stop_reason == "sufficient"


def test_resume_rejects_checkpoint_for_another_question() -> None:
    agent, *_ = build_agent([[make_evidence("A")]], [critique(True)])
    checkpoint = ResearchResumeState(
        question="another question",
        plan=SearchPlan(queries=["query a"]),
        round_index=1,
        executed_queries=("query a",),
        evidence=(make_evidence("A"),),
        next_queries=("query b",),
    )
    with pytest.raises(Exception, match="不一致"):
        asyncio.run(agent.run(QUESTION, resume_state=checkpoint))


def test_completed_retrieval_is_checkpointed_before_critic() -> None:
    agent, *_ = build_agent([[make_evidence("A")]], [critique(True)])
    checkpoints: list[ResearchResumeState] = []
    asyncio.run(agent.run(QUESTION, checkpoint_callback=checkpoints.append))

    before_critic = checkpoints[0]
    assert before_critic.round_index == 1
    assert before_critic.executed_queries == ("query a",)
    assert [item.title for item in before_critic.evidence] == ["A"]
    assert before_critic.research_complete is True
    assert before_critic.stop_reason == "critic_failure"
    assert checkpoints[-1].stop_reason == "sufficient"
