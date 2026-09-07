import asyncio
import json

import pytest

from src.eval.models import EvalRunConfig
from src.eval.runner import EvaluationRunner
from src.runtime.models import RunStatus
from tests.eval.conftest import runtime_result


def config(tmp_path, limit=None):
    path = tmp_path / "dataset.jsonl"
    path.write_text('\n'.join(json.dumps({"id": str(i), "question": f"q{i}"}) for i in range(3)))
    return EvalRunConfig(run_name="test", dataset_path=str(path), output_dir=str(tmp_path), limit=limit)


def test_runner_isolates_failure_persists_traces_and_judge_off(tmp_path):
    calls = []
    async def execute(question):
        calls.append(question)
        if question == "q1":
            raise RuntimeError("setup failed")
        return runtime_result()
    class ForbiddenJudge:
        def evaluate(self, *args):
            raise AssertionError("judge must not run by default")
    target = asyncio.run(EvaluationRunner(execute, judge=ForbiddenJudge()).run(config(tmp_path)))
    rows = [json.loads(line) for line in (target / "cases.jsonl").read_text().splitlines()]
    assert calls == ["q0", "q1", "q2"]
    assert [row["status"] for row in rows] == ["succeeded", "failed", "succeeded"]
    assert rows[0]["raw"]["grounding_trace"]["verified_claim_count"] == 8
    assert rows[0]["raw"]["final_evidence"][0]["url"] == "https://example.org/cache"
    assert all(row["llm_judged"] is None for row in rows)
    assert (target / "report.md").is_file()
    assert json.loads((target / "config.json").read_text())["completed"]


@pytest.mark.parametrize("status", [RunStatus.TIMED_OUT, RunStatus.BUDGET_EXCEEDED])
def test_runtime_terminal_failures_continue(tmp_path, status):
    async def execute(question):
        return runtime_result(status if question == "q1" else RunStatus.SUCCEEDED)
    target = asyncio.run(EvaluationRunner(execute).run(config(tmp_path)))
    metrics = json.loads((target / "metrics.json").read_text())
    assert metrics["case_count"] == 3
    assert metrics["status_counts"][status.value] == 1


@pytest.mark.parametrize("returned", [True, False])
def test_cancellation_stops_and_saves_partial_report(tmp_path, returned):
    calls = []
    async def execute(question):
        calls.append(question)
        if question == "q1":
            if returned:
                return runtime_result(RunStatus.CANCELLED)
            raise asyncio.CancelledError
        return runtime_result()
    conf = config(tmp_path)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(EvaluationRunner(execute).run(conf))
    assert calls == ["q0", "q1"]
    metrics = json.loads((tmp_path / "test/metrics.json").read_text())
    assert metrics["completed"] is False
    assert metrics["case_count"] == (2 if returned else 1)


def test_limit_and_refuse_overwrite(tmp_path):
    async def execute(question):
        return runtime_result()
    conf = config(tmp_path, limit=1)
    target = asyncio.run(EvaluationRunner(execute).run(conf))
    original = (target / "cases.jsonl").read_text()
    assert len(original.splitlines()) == 1
    with pytest.raises(FileExistsError):
        asyncio.run(EvaluationRunner(execute).run(conf))
    assert (target / "cases.jsonl").read_text() == original


def test_optional_judge_saved_separately(tmp_path):
    from src.eval.judges.llm_judge import JudgeResult
    async def execute(question):
        return runtime_result()
    class Judge:
        def evaluate(self, case, answer):
            return JudgeResult(relevance=4, completeness=3, clarity=5, reason="opinion")
    conf = config(tmp_path, limit=1).model_copy(update={"judge": True})
    target = asyncio.run(EvaluationRunner(execute, judge=Judge()).run(conf))
    row = json.loads((target / "cases.jsonl").read_text())
    assert row["llm_judged"]["relevance"] == 4
    assert row["raw"]["runtime_trace"]["counters"]["llm_calls"] == 5


def test_real_runtime_boundary_with_fake_agent(tmp_path):
    from src.runtime.runtime import AgentRuntime
    from src.runtime.models import RunRequest, RuntimeConfig
    class Agent:
        async def run(self, question, **kwargs):
            return runtime_result().research_result
    runtime = AgentRuntime(RuntimeConfig(checkpoint_enabled=False))
    async def execute(question):
        return await runtime.run(RunRequest(question), Agent())
    target = asyncio.run(EvaluationRunner(execute).run(config(tmp_path, limit=1)))
    row = json.loads((target / "cases.jsonl").read_text())
    assert row["status"] == "succeeded"
    assert row["raw"]["run_id"].startswith("run_")
