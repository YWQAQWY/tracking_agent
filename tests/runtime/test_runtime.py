from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.agent.models import ResearchResumeState
from src.models.evidence import Evidence
from src.models.search_plan import SearchPlan
from src.runtime.context import (
    consume_runtime_budget,
    run_async_operation,
    run_sync_operation,
)
from src.runtime.errors import NonRetryableError, RetryableError
from src.runtime.models import (
    RunRequest,
    RunStatus,
    RuntimeBudget,
    RuntimeConfig,
)
from src.runtime.retry import RetryPolicy
from src.runtime.runtime import AgentRuntime


def fake_research_result(answer: str = "answer"):
    return SimpleNamespace(
        answer=answer,
        research_trace=SimpleNamespace(
            stop_reason="sufficient", timings={"planning": 0.01}
        ),
        grounding_trace=None,
    )


def config(tmp_path: Path, **changes) -> RuntimeConfig:
    values = {
        "overall_timeout_seconds": 1.0,
        "search_timeout_seconds": 0.1,
        "crawl_timeout_seconds": 0.1,
        "budget": RuntimeBudget(3, 5, 5, 5),
        "network_retry": RetryPolicy(2, 0.0, 0.0, 0.0),
        "llm_retry": RetryPolicy(2, 0.0, 0.0, 0.0),
        "checkpoint_enabled": True,
        "checkpoint_dir": str(tmp_path),
    }
    values.update(changes)
    return RuntimeConfig(**values)


class SuccessAgent:
    async def run(self, question, resume_state=None, checkpoint_callback=None):
        consume_runtime_budget("research_round")
        await run_async_operation("search", "fake-search", self._search)
        await run_async_operation("crawl", "fake-crawl", self._crawl)
        run_sync_operation("llm", "fake-llm", lambda: "ok")
        return fake_research_result()

    async def _search(self):
        return "ok"

    async def _crawl(self):
        return "ok"


def test_success_lifecycle_run_id_counters_trace_and_checkpoint(tmp_path) -> None:
    runtime = AgentRuntime(config(tmp_path))
    result = asyncio.run(runtime.run(RunRequest("question", "provided"), SuccessAgent()))
    assert result.run_id == "provided"
    assert result.status is RunStatus.SUCCEEDED
    assert result.answer == "answer"
    assert result.trace.status_history == (
        RunStatus.PENDING,
        RunStatus.RUNNING,
        RunStatus.SUCCEEDED,
    )
    assert result.trace.counters.research_rounds == 1
    assert result.trace.counters.search_requests == 1
    assert result.trace.counters.crawl_requests == 1
    assert result.trace.counters.llm_calls == 1
    assert result.trace.budget_limits["search_requests"] == 5
    assert result.trace.stage_durations["planning"] == 0.01
    assert Path(result.trace.checkpoint_path or "").exists()


def test_generated_run_ids_are_unique(tmp_path) -> None:
    runtime = AgentRuntime(config(tmp_path, checkpoint_enabled=False))
    first = asyncio.run(runtime.run(RunRequest("q"), SuccessAgent()))
    second = asyncio.run(runtime.run(RunRequest("q"), SuccessAgent()))
    assert first.run_id != second.run_id
    assert first.run_id.startswith("run_")


def test_fatal_error_becomes_structured_failed_result(tmp_path) -> None:
    class Broken:
        async def run(self, question, resume_state=None, checkpoint_callback=None):
            raise NonRetryableError("invalid schema")

    result = asyncio.run(AgentRuntime(config(tmp_path)).run(RunRequest("q"), Broken()))
    assert result.status is RunStatus.FAILED
    assert result.error is not None
    assert result.error.error_type == "NonRetryableError"
    assert result.error.retryable is False


def test_empty_question_is_a_structured_failure(tmp_path) -> None:
    result = asyncio.run(
        AgentRuntime(config(tmp_path)).run(RunRequest("  "), SuccessAgent())
    )
    assert result.status is RunStatus.FAILED
    assert result.trace.question == "  "
    assert result.error is not None


def test_unsafe_provided_run_id_is_a_structured_failure(tmp_path) -> None:
    result = asyncio.run(
        AgentRuntime(config(tmp_path)).run(RunRequest("q", "../bad"), SuccessAgent())
    )
    assert result.status is RunStatus.FAILED
    assert result.trace.checkpoint_path is None
    assert result.error is not None
    assert "不安全" in result.error.message


def test_missing_resume_checkpoint_is_a_structured_failure(tmp_path) -> None:
    result = asyncio.run(AgentRuntime(config(tmp_path)).resume("run_missing", SuccessAgent()))
    assert result.status is RunStatus.FAILED
    assert result.error is not None
    assert "找不到" in result.error.message


def test_corrupted_resume_checkpoint_is_a_structured_failure(tmp_path) -> None:
    path = tmp_path / "run_bad" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    result = asyncio.run(
        AgentRuntime(config(tmp_path)).resume("run_bad", SuccessAgent())
    )
    assert result.status is RunStatus.FAILED
    assert result.error is not None
    assert result.error.error_type == "CheckpointCorruptedError"


def test_overall_timeout_returns_timed_out(tmp_path) -> None:
    class Slow:
        async def run(self, question, resume_state=None, checkpoint_callback=None):
            await asyncio.sleep(1)

    runtime = AgentRuntime(config(tmp_path, overall_timeout_seconds=0.01))
    result = asyncio.run(runtime.run(RunRequest("q"), Slow()))
    assert result.status is RunStatus.TIMED_OUT
    assert result.termination_reason == "overall_timeout"
    assert result.trace.counters.timeouts == 1


def test_operation_timeout_retries_then_fails_without_becoming_overall_timeout(
    tmp_path,
) -> None:
    class SlowOperation:
        async def run(self, question, resume_state=None, checkpoint_callback=None):
            async def slow():
                await asyncio.sleep(1)

            await run_async_operation("search", "slow", slow)

    runtime = AgentRuntime(config(tmp_path, search_timeout_seconds=0.01))
    result = asyncio.run(runtime.run(RunRequest("q"), SlowOperation()))
    assert result.status is RunStatus.FAILED
    assert result.error is not None
    assert result.error.error_type == "SearchTimeoutError"
    assert result.trace.counters.search_requests == 2
    assert result.trace.counters.retries == 1
    assert result.trace.counters.timeouts == 2


def test_budget_exceeded_is_terminal_before_extra_call(tmp_path) -> None:
    class TooManySearches:
        async def run(self, question, resume_state=None, checkpoint_callback=None):
            async def ok():
                return None

            await run_async_operation("search", "one", ok)
            await run_async_operation("search", "two", ok)

    budget = RuntimeBudget(3, 1, 5, 5)
    runtime = AgentRuntime(config(tmp_path, budget=budget))
    result = asyncio.run(runtime.run(RunRequest("q"), TooManySearches()))
    assert result.status is RunStatus.BUDGET_EXCEEDED
    assert result.trace.counters.search_requests == 1
    assert result.termination_reason == "max_search_requests"


def test_research_round_budget_uses_precise_termination_reason(tmp_path) -> None:
    class TooManyRounds:
        async def run(self, question, resume_state=None, checkpoint_callback=None):
            consume_runtime_budget("research_round")
            consume_runtime_budget("research_round")

    budget = RuntimeBudget(1, 5, 5, 5)
    result = asyncio.run(
        AgentRuntime(config(tmp_path, budget=budget)).run(
            RunRequest("q"), TooManyRounds()
        )
    )
    assert result.status is RunStatus.BUDGET_EXCEEDED
    assert result.termination_reason == "max_research_rounds"
    assert result.trace.counters.research_rounds == 1


def test_llm_budget_prevents_second_sync_call(tmp_path) -> None:
    class TooManyLLMCalls:
        async def run(self, question, resume_state=None, checkpoint_callback=None):
            run_sync_operation("llm", "one", lambda: "ok")
            run_sync_operation("llm", "two", lambda: "ok")

    budget = RuntimeBudget(3, 5, 5, 1)
    result = asyncio.run(
        AgentRuntime(config(tmp_path, budget=budget)).run(
            RunRequest("q"), TooManyLLMCalls()
        )
    )
    assert result.status is RunStatus.BUDGET_EXCEEDED
    assert result.termination_reason == "max_llm_calls"
    assert result.trace.counters.llm_calls == 1


def test_cancellation_returns_cancelled_and_stops_child(tmp_path) -> None:
    async def scenario():
        child_cancelled = asyncio.Event()
        child_started = asyncio.Event()

        class WithChild:
            async def run(self, question, resume_state=None, checkpoint_callback=None):
                async def child():
                    try:
                        child_started.set()
                        await asyncio.sleep(10)
                    finally:
                        child_cancelled.set()

                await asyncio.create_task(child())

        task = asyncio.create_task(
            AgentRuntime(config(tmp_path)).run(RunRequest("q"), WithChild())
        )
        await child_started.wait()
        task.cancel()
        result = await task
        assert result.status is RunStatus.CANCELLED
        assert child_cancelled.is_set()

    asyncio.run(scenario())


def test_retryable_operation_updates_runtime_retry_counter(tmp_path) -> None:
    class Flaky:
        async def run(self, question, resume_state=None, checkpoint_callback=None):
            attempts = 0

            async def operation():
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RetryableError("503")
                return "ok"

            await run_async_operation("search", "flaky", operation)
            return fake_research_result()

    result = asyncio.run(AgentRuntime(config(tmp_path)).run(RunRequest("q"), Flaky()))
    assert result.status is RunStatus.SUCCEEDED
    assert result.trace.counters.search_requests == 2
    assert result.trace.counters.retries == 1
    assert result.trace.stage_durations["runtime.search"] >= 0


def test_checkpoint_disabled_creates_no_directory(tmp_path) -> None:
    directory = tmp_path / "disabled"
    runtime = AgentRuntime(
        config(directory, checkpoint_enabled=False, checkpoint_dir=str(directory))
    )
    result = asyncio.run(runtime.run(RunRequest("q"), SuccessAgent()))
    assert result.status is RunStatus.SUCCEEDED
    assert result.trace.checkpoint_path is None
    assert not directory.exists()


def resume_state() -> ResearchResumeState:
    return ResearchResumeState(
        question="question",
        plan=SearchPlan(queries=["q1"]),
        round_index=1,
        executed_queries=("q1",),
        evidence=(Evidence("A", "https://a", "A", 0, 1.0, 1.0),),
        next_queries=("q2",),
        research_complete=False,
    )


def test_stage_level_resume_loads_agent_state_and_counters(tmp_path) -> None:
    class Interrupted:
        async def run(self, question, resume_state=None, checkpoint_callback=None):
            consume_runtime_budget("research_round")
            checkpoint_callback(globals()["resume_state"]())
            raise NonRetryableError("simulated interruption")

    runtime = AgentRuntime(config(tmp_path))
    first = asyncio.run(
        runtime.run(RunRequest("question", "resumable"), Interrupted())
    )
    assert first.status is RunStatus.FAILED

    class Resumed:
        async def run(self, question, resume_state=None, checkpoint_callback=None):
            assert resume_state is not None
            assert resume_state.round_index == 1
            assert resume_state.executed_queries == ("q1",)
            assert resume_state.next_queries == ("q2",)
            assert [item.text for item in resume_state.evidence] == ["A"]
            consume_runtime_budget("research_round")
            return fake_research_result("resumed answer")

    second = asyncio.run(runtime.resume("resumable", Resumed()))
    assert second.status is RunStatus.SUCCEEDED
    assert second.answer == "resumed answer"
    assert second.trace.counters.research_rounds == 2
    assert second.trace.status_history[-2:] == (
        RunStatus.RUNNING,
        RunStatus.SUCCEEDED,
    )
