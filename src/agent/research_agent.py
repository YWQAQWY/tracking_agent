"""Single-agent Search → Critic → targeted Search control loop."""

from __future__ import annotations

import logging
import time

from src.agent.critic import EvidenceCritic
from src.agent.models import (
    ResearchResult,
    ResearchRoundTrace,
    ResearchState,
    ResearchTrace,
    StopReason,
)
from src.agent.research_round import ResearchRound, ResearchRoundResult
from src.answer.answer_generator import AnswerGenerator
from src.context.context_builder import ContextBuilder
from src.models.evidence import Evidence
from src.planner.search_planner import SearchPlanner
from src.retrieval.reranker import Reranker


logger = logging.getLogger(__name__)


class ResearchAgentError(RuntimeError):
    """Raised when no usable evidence or final answer can be produced."""


class ResearchAgent:
    """Coordinate one bounded evidence-driven loop around V0.4 retrieval."""

    def __init__(
        self,
        planner: SearchPlanner,
        research_round: ResearchRound,
        critic: EvidenceCritic,
        final_reranker: Reranker,
        context_builder: ContextBuilder,
        answer_generator: AnswerGenerator,
        max_rounds: int = 3,
        max_followup_queries_per_round: int = 3,
        min_new_evidence_to_continue: int = 1,
        final_evidence_top_k: int = 8,
    ) -> None:
        if min(
            max_rounds,
            max_followup_queries_per_round,
            min_new_evidence_to_continue,
            final_evidence_top_k,
        ) < 1:
            raise ValueError("Research Agent budgets 必须大于 0")
        self.planner = planner
        self.research_round = research_round
        self.critic = critic
        self.final_reranker = final_reranker
        self.context_builder = context_builder
        self.answer_generator = answer_generator
        self.max_rounds = max_rounds
        self.max_followup_queries_per_round = max_followup_queries_per_round
        self.min_new_evidence_to_continue = min_new_evidence_to_continue
        self.final_evidence_top_k = final_evidence_top_k

    async def run(self, question: str) -> ResearchResult:
        total_started = time.perf_counter()
        clean_question = question.strip()
        if not clean_question:
            raise ResearchAgentError("用户问题不能为空。")

        started = time.perf_counter()
        plan = self.planner.plan(clean_question)
        planning_time = time.perf_counter() - started
        state = ResearchState(question=clean_question)
        queries = self.sanitize_queries(plan.queries, [], len(plan.queries))
        state.executed_queries.extend(queries)
        round_results: list[ResearchRoundResult] = []
        round_traces: list[ResearchRoundTrace] = []
        stop_reason: StopReason = "max_rounds"
        critic_error: str | None = None
        total_critic_time = 0.0

        for round_index in range(1, self.max_rounds + 1):
            state.round_index = round_index
            logger.info("Starting research round %d/%d", round_index, self.max_rounds)
            for query in queries:
                logger.info("Round %d query: %s", round_index, query)

            round_result = await self.research_round.run(clean_question, queries)
            round_results.append(round_result)
            added = state.evidence_pool.extend(list(round_result.evidence))
            logger.info("Round %d produced %d new evidence chunks", round_index, added)
            logger.info(
                "Evidence pool contains %d chunks from %d sources",
                state.evidence_pool.size,
                state.evidence_pool.unique_source_count,
            )

            if state.evidence_pool.size == 0:
                raise ResearchAgentError(
                    "初始搜索没有生成可用 Evidence；"
                    "请检查网络、页面读取或检索配置。"
                )
            if round_index > 1 and added < self.min_new_evidence_to_continue:
                stop_reason = "no_new_evidence"
                round_traces.append(
                    self._round_trace(round_index, queries, round_result, added, state)
                )
                logger.info("Stopping research: no new evidence")
                break

            logger.info("Running evidence critic")
            critic_started = time.perf_counter()
            try:
                critique = self.critic.evaluate(
                    clean_question,
                    state.evidence_pool.all(),
                    list(state.executed_queries),
                )
            except Exception as exc:
                critic_time = time.perf_counter() - critic_started
                total_critic_time += critic_time
                critic_error = f"{exc.__class__.__name__}: {exc}"
                stop_reason = "critic_failure"
                round_traces.append(
                    self._round_trace(
                        round_index,
                        queries,
                        round_result,
                        added,
                        state,
                        critic_time=critic_time,
                    )
                )
                logger.warning(
                    "Critic failed; using existing evidence: %s", critic_error
                )
                break
            critic_time = time.perf_counter() - critic_started
            total_critic_time += critic_time
            logger.info(
                "Critic result: %s",
                "sufficient" if critique.sufficient else "insufficient",
            )
            for aspect in critique.missing_aspects:
                logger.info("Missing aspect: %s", aspect)

            sanitized = self.sanitize_queries(
                critique.follow_up_queries,
                state.executed_queries,
                self.max_followup_queries_per_round,
            )
            round_traces.append(
                self._round_trace(
                    round_index,
                    queries,
                    round_result,
                    added,
                    state,
                    critique.sufficient,
                    critique.missing_aspects,
                    sanitized,
                    critique.reason,
                    critic_time,
                )
            )
            if critique.sufficient:
                stop_reason = "sufficient"
                logger.info("Stopping research: evidence sufficient")
                break
            if round_index >= self.max_rounds:
                stop_reason = "max_rounds"
                logger.info("Stopping research: maximum rounds reached")
                break
            if not critique.follow_up_queries:
                stop_reason = "no_follow_up_queries"
                logger.info("Stopping research: critic supplied no follow-up queries")
                break
            if not sanitized:
                stop_reason = "duplicate_queries"
                logger.info("Stopping research: all follow-up queries were duplicates")
                break

            queries = sanitized
            state.executed_queries.extend(queries)

        started = time.perf_counter()
        pooled = state.evidence_pool.all()
        logger.info("Final reranking %d pooled evidence chunks", len(pooled))
        candidates = [item.to_scored_chunk() for item in pooled]
        final_evidence = self.final_reranker.rerank(
            clean_question, candidates, top_k=self.final_evidence_top_k
        )
        final_rerank_time = time.perf_counter() - started
        self._offload(self.final_reranker)
        if not final_evidence:
            raise ResearchAgentError("最终全局重排没有选出可用 Evidence。")
        logger.info("Selected %d final evidence chunks", len(final_evidence))

        started = time.perf_counter()
        context = self.context_builder.build(final_evidence)
        context_time = time.perf_counter() - started
        if not context:
            raise ResearchAgentError("Final Evidence 无法构造成有效模型上下文。")

        started = time.perf_counter()
        logger.info("Calling answer generator for the original question")
        answer = self.answer_generator.generate(clean_question, context)
        generation_time = time.perf_counter() - started
        timings = self._aggregate_timings(
            round_results,
            planning_time,
            total_critic_time,
            final_rerank_time,
            context_time,
            generation_time,
            time.perf_counter() - total_started,
        )
        trace = ResearchTrace(
            rounds=tuple(round_traces),
            stop_reason=stop_reason,
            timings=timings,
            critic_error=critic_error,
        )
        return ResearchResult(
            plan=plan,
            rounds=tuple(round_results),
            pooled_evidence=tuple(pooled),
            evidence=tuple(final_evidence),
            research_trace=trace,
            answer=answer,
        )

    @staticmethod
    def sanitize_queries(
        queries: list[str], executed_queries: list[str], limit: int
    ) -> list[str]:
        """Strip, exact-deduplicate, remove executed queries, and cap output."""
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

    @staticmethod
    def _round_trace(
        round_index: int,
        queries: list[str],
        result: ResearchRoundResult,
        added: int,
        state: ResearchState,
        sufficient: bool | None = None,
        missing_aspects: list[str] | None = None,
        follow_up_queries: list[str] | None = None,
        critic_reason: str | None = None,
        critic_time: float = 0.0,
    ) -> ResearchRoundTrace:
        timings = dict(result.retrieval_trace.timings)
        timings["critic"] = critic_time
        return ResearchRoundTrace(
            round_index=round_index,
            queries=tuple(queries),
            search_result_count=len(result.search_batch.results),
            new_evidence_count=added,
            total_evidence_count=state.evidence_pool.size,
            critic_sufficient=sufficient,
            missing_aspects=tuple(missing_aspects or []),
            follow_up_queries=tuple(follow_up_queries or []),
            critic_reason=critic_reason,
            timings=timings,
        )

    @staticmethod
    def _aggregate_timings(
        rounds: list[ResearchRoundResult],
        planning: float,
        critic: float,
        final_rerank: float,
        context: float,
        generation: float,
        total: float,
    ) -> dict[str, float]:
        return {
            "planning": planning,
            "search": sum(
                item.retrieval_trace.timings.get("search", 0.0) for item in rounds
            ),
            "crawl_extract": sum(
                item.retrieval_trace.timings.get("crawl_extract", 0.0)
                for item in rounds
            ),
            "retrieval": sum(
                sum(
                    item.retrieval_trace.timings.get(stage, 0.0)
                    for stage in ("chunk", "embedding", "rerank")
                )
                for item in rounds
            ),
            "critic": critic,
            "final_rerank": final_rerank,
            "context": context,
            "generation": generation,
            "total": total,
        }

    @staticmethod
    def _offload(component: object) -> None:
        offload = getattr(component, "offload", None)
        if callable(offload):
            offload()
