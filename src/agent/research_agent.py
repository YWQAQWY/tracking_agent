"""Single-agent Search → Critic → targeted Search control loop."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from src.action.models import ActionKind, StopReason
from src.action.policy import ResearchActionPolicy, sanitize_queries
from src.agent.critic import EvidenceCritic
from src.agent.models import (
    ResearchResult,
    ResearchRoundTrace,
    ResearchTrace,
)
from src.answer.answer_generator import AnswerGenerator
from src.context.context_builder import ContextBuilder
from src.grounding.generator import GroundedAnswerGenerator, GroundedGenerationError
from src.grounding.models import CitationSource, GroundingTrace
from src.memory.research_state import ResearchResumeState, ResearchState
from src.models.evidence import Evidence
from src.plan.models import SearchPlan
from src.plan.search_planner import SearchPlanner
from src.retrieval.reranker import Reranker
from src.runtime.context import consume_runtime_budget, set_runtime_stage
from src.runtime.errors import BudgetExceededError
from src.runtime.models import RunStage
from src.tools.research import ResearchTool, ResearchToolResult


logger = logging.getLogger(__name__)


class ResearchAgentError(RuntimeError):
    """Raised when no usable evidence or final answer can be produced."""


class ResearchAgent:
    """Coordinate one bounded evidence-driven loop around V0.4 retrieval."""

    def __init__(
        self,
        planner: SearchPlanner,
        research_round: ResearchTool,
        critic: EvidenceCritic,
        final_reranker: Reranker,
        context_builder: ContextBuilder,
        answer_generator: AnswerGenerator,
        max_rounds: int = 3,
        max_followup_queries_per_round: int = 3,
        min_new_evidence_to_continue: int = 1,
        final_evidence_top_k: int = 8,
        grounded_answer_generator: GroundedAnswerGenerator | None = None,
        action_policy: ResearchActionPolicy | None = None,
    ) -> None:
        if min(
            max_rounds,
            max_followup_queries_per_round,
            min_new_evidence_to_continue,
            final_evidence_top_k,
        ) < 1:
            raise ValueError("Research Agent budgets 必须大于 0")
        self.planner = planner
        self.research_tool = research_round
        self.critic = critic
        self.final_reranker = final_reranker
        self.context_builder = context_builder
        self.answer_generator = answer_generator
        self.max_rounds = max_rounds
        self.max_followup_queries_per_round = max_followup_queries_per_round
        self.min_new_evidence_to_continue = min_new_evidence_to_continue
        self.final_evidence_top_k = final_evidence_top_k
        self.grounded_answer_generator = grounded_answer_generator
        self.action_policy = action_policy or ResearchActionPolicy(
            max_rounds=max_rounds,
            max_followup_queries=max_followup_queries_per_round,
        )

    async def run(
        self,
        question: str,
        resume_state: ResearchResumeState | None = None,
        checkpoint_callback: Callable[[ResearchResumeState], None] | None = None,
    ) -> ResearchResult:
        total_started = time.perf_counter()
        clean_question = question.strip()
        if not clean_question:
            raise ResearchAgentError("用户问题不能为空。")

        if resume_state is not None:
            if resume_state.question != clean_question:
                raise ResearchAgentError("resume checkpoint 与当前问题不一致。")
            plan = resume_state.plan
            planning_time = 0.0
            state = ResearchState(
                question=clean_question,
                round_index=resume_state.round_index,
                executed_queries=list(resume_state.executed_queries),
            )
            state.evidence_pool.extend(list(resume_state.evidence))
            queries = list(resume_state.next_queries)
            first_round = resume_state.round_index + 1
            research_complete = resume_state.research_complete
            resume_stop_reason = resume_state.stop_reason
            logger.info(
                "Resuming research after round %d with %d pooled evidence chunks",
                resume_state.round_index,
                state.evidence_pool.size,
            )
        else:
            started = time.perf_counter()
            set_runtime_stage(RunStage.PLANNING)
            plan = self.planner.plan(clean_question)
            planning_time = time.perf_counter() - started
            state = ResearchState(question=clean_question)
            queries = sanitize_queries(plan.queries, [], len(plan.queries))
            first_round = 1
            research_complete = False
            resume_stop_reason = None
        round_results: list[ResearchToolResult] = []
        round_traces: list[ResearchRoundTrace] = []
        stop_reason: StopReason = (
            resume_stop_reason if resume_stop_reason is not None else "max_rounds"
        )
        critic_error: str | None = None
        total_critic_time = 0.0

        for round_index in range(first_round, self.max_rounds + 1):
            if research_complete:
                break
            consume_runtime_budget("research_round")
            set_runtime_stage(RunStage.SEARCH)
            state.round_index = round_index
            for query in queries:
                if query not in state.executed_queries:
                    state.executed_queries.append(query)
            logger.info("Starting research round %d/%d", round_index, self.max_rounds)
            for query in queries:
                logger.info("Round %d query: %s", round_index, query)

            round_result = await self.research_tool.run(clean_question, queries)
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
                    self._round_trace(
                        round_index,
                        queries,
                        round_result,
                        added,
                        state,
                        action_kind="finish",
                        action_stop_reason="no_new_evidence",
                    )
                )
                logger.info("Stopping research: no new evidence")
                break

            # The expensive Search → Read → Retrieve stage is complete. If the
            # process stops during Critic, resume from this evidence and use the
            # existing critic-failure degradation instead of repeating the round.
            self._save_resume_state(
                checkpoint_callback,
                clean_question,
                plan,
                state,
                [],
                research_complete=True,
                stop_reason="critic_failure",
            )
            logger.info("Running evidence critic")
            set_runtime_stage(RunStage.EVALUATION)
            critic_started = time.perf_counter()
            try:
                critique = self.critic.evaluate(
                    clean_question,
                    state.evidence_pool.all(),
                    list(state.executed_queries),
                )
            except BudgetExceededError:
                raise
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
                        action_kind="finish",
                        action_stop_reason="critic_failure",
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

            action = self.action_policy.decide(
                sufficient=critique.sufficient,
                follow_up_queries=critique.follow_up_queries,
                executed_queries=state.executed_queries,
                round_index=round_index,
            )
            sanitized = list(action.queries)
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
                    action_kind=action.kind,
                    action_stop_reason=action.stop_reason,
                )
            )
            if action.kind == "finish":
                if action.stop_reason is None:  # guarded by ResearchAction
                    raise RuntimeError("finish action missing stop reason")
                stop_reason = action.stop_reason
                logger.info("Stopping research: %s", stop_reason)
                break

            queries = sanitized
            self._save_resume_state(
                checkpoint_callback,
                clean_question,
                plan,
                state,
                queries,
                research_complete=False,
                stop_reason=None,
            )

        self._save_resume_state(
            checkpoint_callback,
            clean_question,
            plan,
            state,
            [],
            research_complete=True,
            stop_reason=stop_reason,
        )

        started = time.perf_counter()
        set_runtime_stage(RunStage.RETRIEVAL)
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

        context_time = 0.0
        sources: tuple[CitationSource, ...] = ()
        grounding_trace: GroundingTrace | None = None
        grounding_verified = False
        started = time.perf_counter()
        if self.grounded_answer_generator is not None:
            set_runtime_stage(RunStage.GROUNDING)
            logger.info("Starting claim-level grounded answer generation")
            try:
                grounded = self.grounded_answer_generator.generate(
                    clean_question, final_evidence
                )
            except GroundedGenerationError as exc:
                logger.warning(
                    "Grounding failed; falling back to legacy generation: %s", exc
                )
                answer, context_time = self._legacy_answer(
                    clean_question, final_evidence
                )
                grounding_trace = GroundingTrace(
                    draft_claim_count=0,
                    initially_supported_claim_count=0,
                    unsupported_claim_count=0,
                    rewritten_claim_count=0,
                    rewrite_passed_claim_count=0,
                    dropped_claim_count=0,
                    verified_claim_count=0,
                    cited_source_count=0,
                    fallback_reason=str(exc),
                )
            else:
                answer = grounded.text
                sources = grounded.sources
                grounding_trace = grounded.grounding_trace
                grounding_verified = grounded.grounding_verified
        else:
            set_runtime_stage(RunStage.RENDERING)
            answer, context_time = self._legacy_answer(clean_question, final_evidence)
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
            sources=sources,
            grounding_trace=grounding_trace,
            grounding_verified=grounding_verified,
        )

    def _legacy_answer(
        self, question: str, evidence: list[Evidence]
    ) -> tuple[str, float]:
        started = time.perf_counter()
        context = self.context_builder.build(evidence)
        context_time = time.perf_counter() - started
        if not context:
            raise ResearchAgentError("Final Evidence 无法构造成有效模型上下文。")
        logger.info("Calling legacy answer generator for the original question")
        return self.answer_generator.generate(question, context), context_time

    @property
    def research_round(self) -> ResearchTool:
        """Backward-compatible name for integrations using ResearchRound."""
        return self.research_tool

    @research_round.setter
    def research_round(self, value: ResearchTool) -> None:
        self.research_tool = value

    @staticmethod
    def _save_resume_state(
        callback: Callable[[ResearchResumeState], None] | None,
        question: str,
        plan: SearchPlan,
        state: ResearchState,
        next_queries: list[str],
        research_complete: bool,
        stop_reason: StopReason | None,
    ) -> None:
        if callback is None:
            return
        callback(
            ResearchResumeState(
                question=question,
                plan=plan,
                round_index=state.round_index,
                executed_queries=tuple(state.executed_queries),
                evidence=tuple(state.evidence_pool.all()),
                next_queries=tuple(next_queries),
                research_complete=research_complete,
                stop_reason=stop_reason,
            )
        )

    @staticmethod
    def sanitize_queries(
        queries: list[str], executed_queries: list[str], limit: int
    ) -> list[str]:
        """Compatibility entry point; action policy owns query normalization."""
        return sanitize_queries(queries, executed_queries, limit)

    @staticmethod
    def _round_trace(
        round_index: int,
        queries: list[str],
        result: ResearchToolResult,
        added: int,
        state: ResearchState,
        sufficient: bool | None = None,
        missing_aspects: list[str] | None = None,
        follow_up_queries: list[str] | None = None,
        critic_reason: str | None = None,
        critic_time: float = 0.0,
        action_kind: ActionKind | None = None,
        action_stop_reason: StopReason | None = None,
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
            action=action_kind,
            action_stop_reason=action_stop_reason,
            missing_aspects=tuple(missing_aspects or []),
            follow_up_queries=tuple(follow_up_queries or []),
            critic_reason=critic_reason,
            timings=timings,
        )

    @staticmethod
    def _aggregate_timings(
        rounds: list[ResearchToolResult],
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
