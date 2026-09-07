from datetime import UTC, datetime

import pytest

from src.agent.models import ResearchResult, ResearchRoundTrace, ResearchTrace
from src.eval.extractor import extract_runtime
from src.eval.models import EvalCaseResult
from src.grounding.models import ClaimTrace, CoverageResult, GroundingTrace
from src.models.evidence import Evidence
from src.plan.models import SearchPlan
from src.runtime.models import RunStatus, RuntimeCounters, RuntimeResult, RuntimeTrace


def runtime_result(status=RunStatus.SUCCEEDED, duration=2.0):
    now = datetime.now(UTC)
    evidence = Evidence("Caching reuses data.", "https://example.org/cache", "Cache", 0, 0.5, 0.8)
    grounding = GroundingTrace(
        10, 6, 4, 3, 2, 2, 8, 1,
        claim_traces=tuple(ClaimTrace(f"C{i}", "original", "Caching reuses data.",
                                     ("E1",), True, False, "supported") for i in range(8)),
        coverage=CoverageResult(adequate=False, covered_aspects=["method"],
                                missing_aspects=["limits"], reason="missing limits"),
    )
    trace = ResearchTrace(tuple(
        ResearchRoundTrace(index, (f"q{index}",), 2, count, 2 + count, False)
        for index, count in ((1, 2), (2, 2), (3, 0))
    ), "no_new_evidence")
    research = ResearchResult(SearchPlan(queries=["q1"]), (), (evidence,), (evidence,),
                              trace, "Caching reuses data. [1]", grounding_trace=grounding,
                              grounding_verified=True)
    rt = RuntimeTrace("run-test", "question", status, now, now, duration,
                      RuntimeCounters(3, 4, 2, 5, 1, 0), "no_new_evidence",
                      stage_durations={"search": 0.5})
    return RuntimeResult("run-test", status,
                         research if status == RunStatus.SUCCEEDED else None,
                         None, rt, "no_new_evidence")


@pytest.fixture
def row():
    result = runtime_result()
    return EvalCaseResult(case_id="one", question="question", category="demo",
                          task="end_to_end", status="succeeded", latency_seconds=2,
                          raw=extract_runtime(result))
