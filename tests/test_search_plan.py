import pytest
from pydantic import ValidationError

from src.models.search_plan import SearchPlan
from src.planner.search_planner import PlannerError, SearchPlanner


class StubLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def chat(self, user_prompt: str, system_prompt: str | None = None) -> str:
        self.calls += 1
        return next(self.responses)


def test_search_plan_accepts_and_strips_query() -> None:
    plan = SearchPlan(query="  Qwen3 recent updates  ", max_results=5)
    assert plan.query == "Qwen3 recent updates"
    assert plan.max_results == 5


@pytest.mark.parametrize("max_results", [0, 11])
def test_search_plan_rejects_out_of_range(max_results: int) -> None:
    with pytest.raises(ValidationError):
        SearchPlan(query="Qwen3", max_results=max_results)


def test_search_plan_rejects_blank_query() -> None:
    with pytest.raises(ValidationError):
        SearchPlan(query="   ", max_results=5)


def test_planner_cleans_thinking_and_markdown() -> None:
    llm = StubLLM(
        ['<think>hidden</think>\n```json\n{"query":"Qwen3 news","max_results":5}\n```']
    )
    plan = SearchPlanner(llm).plan("Qwen3 有什么进展？")
    assert plan == SearchPlan(query="Qwen3 news", max_results=5)
    assert llm.calls == 1


def test_planner_extracts_json_from_extra_text() -> None:
    llm = StubLLM(['说明：{"query":"local Qwen3","max_results":4} 完成'])
    plan = SearchPlanner(llm).plan("Qwen3")
    assert plan.query == "local Qwen3"


def test_planner_retries_once_then_raises_clear_error() -> None:
    llm = StubLLM(["not json", '{"query":"","max_results":99}'])
    with pytest.raises(PlannerError, match="连续两次"):
        SearchPlanner(llm).plan("test")
    assert llm.calls == 2


def test_planner_rejects_blank_question_without_calling_llm() -> None:
    llm = StubLLM(['{"query":"unused","max_results":5}'])
    with pytest.raises(PlannerError, match="不能为空"):
        SearchPlanner(llm).plan("   ")
    assert llm.calls == 0

