"""Single-machine execution runtime around the existing ResearchAgent."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from src.action.models import STOP_REASONS, StopReason
from src.agent.models import ResearchResult
from src.memory.research_state import ResearchResumeState
from src.models.evidence import Evidence
from src.plan.models import SearchPlan
from src.runtime.budgets import BudgetTracker
from src.runtime.checkpoint import CheckpointStore
from src.runtime.context import (
    RuntimeExecutionContext,
    activate_runtime,
    deactivate_runtime,
    is_retryable_exception,
    is_timeout_exception,
)
from src.runtime.errors import (
    BudgetExceededError,
    CheckpointError,
    NonRetryableError,
)
from src.runtime.models import (
    RunRequest,
    RunStatus,
    RuntimeConfig,
    RuntimeCounters,
    RuntimeErrorInfo,
    RuntimeResult,
    RuntimeState,
    RuntimeTrace,
)


logger = logging.getLogger(__name__)


class ResearchRunner(Protocol):
    async def run(
        self,
        question: str,
        resume_state: ResearchResumeState | None = None,
        checkpoint_callback: Any | None = None,
    ) -> ResearchResult: ...


class AgentRuntime:
    """Own lifecycle and execution constraints, not research decisions."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.checkpoints = checkpoint_store or CheckpointStore(
            self.config.checkpoint_dir
        )

    async def run(
        self,
        request: RunRequest,
        agent: ResearchRunner,
    ) -> RuntimeResult:
        run_id = request.run_id or f"run_{uuid.uuid4().hex}"
        clean_question = request.question.strip()
        if not clean_question:
            return self._immediate_failure(
                run_id, request.question, NonRetryableError("question 不能为空")
            )
        return await self._execute(run_id, clean_question, agent)

    async def resume(self, run_id: str, agent: ResearchRunner) -> RuntimeResult:
        try:
            payload = self.checkpoints.load_state(run_id)
            question = str(payload["question"])
            counters = RuntimeCounters.from_dict(
                _require_dict(payload.get("counters"), "counters")
            )
            agent_state = _deserialize_agent_state(
                _require_dict(payload.get("agent_state"), "agent_state")
            )
            history = [
                RunStatus(item) for item in payload.get("status_history", [])
            ]
        except Exception as exc:
            return self._immediate_failure(run_id, "", exc)
        return await self._execute(
            run_id,
            question,
            agent,
            resume_state=agent_state,
            counters=counters,
            prior_history=history,
        )

    async def _execute(
        self,
        run_id: str,
        question: str,
        agent: ResearchRunner,
        resume_state: ResearchResumeState | None = None,
        counters: RuntimeCounters | None = None,
        prior_history: list[RunStatus] | None = None,
    ) -> RuntimeResult:
        state = RuntimeState(run_id=run_id, question=question)
        if prior_history:
            state.status_history = list(prior_history)
            state.transition(RunStatus.PENDING)
        state.counters = counters or RuntimeCounters()
        tracker = BudgetTracker(self.config.budget, state.counters)
        latest_agent_state = resume_state
        started_perf = time.perf_counter()
        overall_deadline = started_perf + self.config.overall_timeout_seconds
        state.started_at = datetime.now(UTC)
        try:
            self._save_state(state, latest_agent_state)
        except Exception as exc:
            return self._immediate_failure(run_id, question, exc)
        state.transition(RunStatus.RUNNING)
        try:
            self._save_state(state, latest_agent_state)
        except Exception as exc:
            return self._immediate_failure(run_id, question, exc)
        logger.info(
            "status=running", extra={"run_id": run_id, "stage": "runtime"}
        )

        def checkpoint_callback(checkpoint: ResearchResumeState) -> None:
            nonlocal latest_agent_state
            latest_agent_state = checkpoint
            state.current_research_round = checkpoint.round_index
            self._save_state(state, latest_agent_state)

        execution = RuntimeExecutionContext(
            run_id=run_id,
            tracker=tracker,
            network_retry=self.config.network_retry,
            llm_retry=self.config.llm_retry,
            search_timeout=self.config.search_timeout_seconds,
            crawl_timeout=self.config.crawl_timeout_seconds,
            overall_deadline=overall_deadline,
            stage_callback=lambda stage: setattr(state, "current_stage", stage),
        )
        token = activate_runtime(execution)
        research_result: ResearchResult | None = None
        error: BaseException | None = None
        try:
            async with asyncio.timeout(self.config.overall_timeout_seconds):
                research_result = await agent.run(
                    question,
                    resume_state=resume_state,
                    checkpoint_callback=checkpoint_callback,
                )
        except BudgetExceededError as exc:
            state.transition(RunStatus.BUDGET_EXCEEDED)
            state.termination_reason = {
                "research_round": "max_research_rounds",
                "search": "max_search_requests",
                "crawl": "max_crawl_requests",
                "llm": "max_llm_calls",
            }.get(exc.resource, "budget_exceeded")
            error = exc
        except TimeoutError as exc:
            state.counters.timeouts += 1
            state.transition(RunStatus.TIMED_OUT)
            state.termination_reason = "overall_timeout"
            error = exc
            logger.warning(
                "overall_timeout=%.1fs",
                self.config.overall_timeout_seconds,
                extra={"run_id": run_id, "stage": "runtime"},
            )
        except asyncio.CancelledError as exc:
            state.transition(RunStatus.CANCELLED)
            state.termination_reason = "cancelled"
            error = exc
            logger.info(
                "status=cancelled",
                extra={"run_id": run_id, "stage": "runtime"},
            )
        except Exception as exc:
            error = exc
            if time.perf_counter() >= overall_deadline and is_timeout_exception(exc):
                state.transition(RunStatus.TIMED_OUT)
                state.termination_reason = "overall_timeout"
                logger.warning(
                    "overall_timeout=%.1fs",
                    self.config.overall_timeout_seconds,
                    extra={"run_id": run_id, "stage": "runtime"},
                )
            else:
                state.transition(RunStatus.FAILED)
                state.termination_reason = "runtime_error"
                logger.error(
                    "status=failed error=%s",
                    f"{exc.__class__.__name__}: {exc}",
                    extra={"run_id": run_id, "stage": "runtime"},
                )
        else:
            state.transition(RunStatus.SUCCEEDED)
            stop_reason = research_result.research_trace.stop_reason
            state.termination_reason = (
                "evidence_sufficient" if stop_reason == "sufficient" else stop_reason
            )
        finally:
            deactivate_runtime(token)

        state.finished_at = datetime.now(UTC)
        total_duration = time.perf_counter() - started_perf
        trace = self._build_trace(
            state,
            research_result,
            total_duration,
            execution.stage_durations,
        )
        try:
            self._save_state(state, latest_agent_state)
            self._save_trace(trace)
        except Exception as checkpoint_error:
            logger.error(
                "checkpoint_save_failed error=%s",
                f"{checkpoint_error.__class__.__name__}: {checkpoint_error}",
                extra={"run_id": run_id, "stage": "runtime"},
            )
            if error is None:
                error = checkpoint_error
                state.transition(RunStatus.FAILED)
                state.termination_reason = "checkpoint_failure"
                trace = self._build_trace(
                    state,
                    research_result,
                    total_duration,
                    execution.stage_durations,
                )
        logger.info(
            "status=%s termination_reason=%s duration=%.3fs",
            state.status.value,
            state.termination_reason,
            total_duration,
            extra={"run_id": run_id, "stage": "runtime"},
        )
        return RuntimeResult(
            run_id=run_id,
            status=state.status,
            research_result=research_result,
            error=_error_info(error) if error else None,
            trace=trace,
            termination_reason=state.termination_reason or "runtime_error",
        )

    def _build_trace(
        self,
        state: RuntimeState,
        result: ResearchResult | None,
        total_duration: float,
        runtime_stage_durations: dict[str, float] | None = None,
    ) -> RuntimeTrace:
        stages: dict[str, float] = {}
        if result:
            stages.update(result.research_trace.timings)
            if result.grounding_trace:
                stages.update(
                    {
                        f"grounding.{name}": duration
                        for name, duration in result.grounding_trace.timings.items()
                    }
                )
        stages.update(runtime_stage_durations or {})
        checkpoint_path = None
        if self.config.checkpoint_enabled:
            try:
                checkpoint_path = str(self.checkpoints.state_path(state.run_id))
            except CheckpointError:
                pass
        return RuntimeTrace(
            run_id=state.run_id,
            question=state.question,
            status=state.status,
            started_at=state.started_at or datetime.now(UTC),
            finished_at=state.finished_at or datetime.now(UTC),
            total_duration=total_duration,
            counters=RuntimeCounters.from_dict(state.counters.as_dict()),
            termination_reason=state.termination_reason or "runtime_error",
            budget_limits={
                "research_rounds": self.config.budget.max_research_rounds,
                "search_requests": self.config.budget.max_search_requests,
                "crawl_requests": self.config.budget.max_crawl_requests,
                "llm_calls": self.config.budget.max_llm_calls,
            },
            stage_durations=stages,
            status_history=tuple(state.status_history),
            checkpoint_path=checkpoint_path,
        )

    def _save_state(
        self,
        state: RuntimeState,
        agent_state: ResearchResumeState | None,
    ) -> None:
        if not self.config.checkpoint_enabled:
            return
        self.checkpoints.save_state(
            state.run_id,
            {
                "version": 1,
                "run_id": state.run_id,
                "question": state.question,
                "status": state.status.value,
                "started_at": _iso(state.started_at),
                "finished_at": _iso(state.finished_at),
                "termination_reason": state.termination_reason,
                "current_stage": state.current_stage.value,
                "current_research_round": state.current_research_round,
                "counters": state.counters.as_dict(),
                "status_history": [item.value for item in state.status_history],
                "agent_state": _serialize_agent_state(agent_state),
            },
        )

    def _save_trace(self, trace: RuntimeTrace) -> None:
        if not self.config.checkpoint_enabled:
            return
        self.checkpoints.save_trace(
            trace.run_id,
            {
                "run_id": trace.run_id,
                "question": trace.question,
                "status": trace.status.value,
                "started_at": trace.started_at.isoformat(),
                "finished_at": trace.finished_at.isoformat(),
                "total_duration": trace.total_duration,
                "counters": trace.counters.as_dict(),
                "termination_reason": trace.termination_reason,
                "budget_limits": trace.budget_limits,
                "stage_durations": trace.stage_durations,
                "status_history": [item.value for item in trace.status_history],
                "checkpoint_path": trace.checkpoint_path,
            },
        )

    def _immediate_failure(
        self, run_id: str, question: str, error: BaseException
    ) -> RuntimeResult:
        now = datetime.now(UTC)
        state = RuntimeState(
            run_id=run_id,
            question=question,
            status=RunStatus.FAILED,
            started_at=now,
            finished_at=now,
            termination_reason="runtime_error",
            status_history=[RunStatus.PENDING, RunStatus.FAILED],
        )
        trace = self._build_trace(state, None, 0.0)
        return RuntimeResult(
            run_id,
            RunStatus.FAILED,
            None,
            _error_info(error),
            trace,
            "runtime_error",
        )


def _serialize_agent_state(state: ResearchResumeState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "question": state.question,
        "plan": state.plan.model_dump(mode="json"),
        "round_index": state.round_index,
        "executed_queries": list(state.executed_queries),
        "evidence": [
            {
                "text": item.text,
                "url": item.url,
                "title": item.title,
                "chunk_index": item.chunk_index,
                "embedding_score": item.embedding_score,
                "rerank_score": item.rerank_score,
            }
            for item in state.evidence
        ],
        "next_queries": list(state.next_queries),
        "research_complete": state.research_complete,
        "stop_reason": state.stop_reason,
    }


def _deserialize_agent_state(payload: dict[str, Any]) -> ResearchResumeState:
    try:
        raw_stop_reason = payload.get("stop_reason")
        if (
            raw_stop_reason is not None
            and raw_stop_reason not in STOP_REASONS
        ):
            raise ValueError("invalid stop_reason")
        return ResearchResumeState(
            question=str(payload["question"]),
            plan=SearchPlan.model_validate(payload["plan"]),
            round_index=int(payload["round_index"]),
            executed_queries=tuple(str(item) for item in payload["executed_queries"]),
            evidence=tuple(Evidence(**item) for item in payload["evidence"]),
            next_queries=tuple(str(item) for item in payload["next_queries"]),
            research_complete=bool(payload.get("research_complete", False)),
            stop_reason=cast(StopReason | None, raw_stop_reason),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint agent_state 格式无效") from exc


def _require_dict(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointError(f"checkpoint {name} 缺失或格式无效")
    return value


def _error_info(error: BaseException) -> RuntimeErrorInfo:
    return RuntimeErrorInfo(
        error_type=error.__class__.__name__,
        message=str(error) or error.__class__.__name__,
        retryable=is_retryable_exception(error),
    )


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None
