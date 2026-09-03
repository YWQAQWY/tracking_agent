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


def test_search_plan_accepts_strips_and_deduplicates_queries() -> None:
    plan = SearchPlan(
        queries=["  Qwen3 recent updates  ", "Qwen3 recent updates", "Qwen3 release"]
    )
    assert plan.queries == ["Qwen3 recent updates", "Qwen3 release"]


def test_search_plan_rejects_only_blank_queries() -> None:
    with pytest.raises(ValidationError):
        SearchPlan(queries=["   ", ""])


def test_planner_cleans_thinking_and_markdown() -> None:
    llm = StubLLM(
        [
            '<think>hidden</think>\n```json\n'
            '{"queries":["Qwen3 news","Qwen3 release notes"]}\n```'
        ]
    )
    plan = SearchPlanner(llm).plan("Qwen3 有什么进展？")
    assert plan == SearchPlan(queries=["Qwen3 news", "Qwen3 release notes"])
    assert llm.calls == 1


def test_planner_extracts_json_from_extra_text() -> None:
    llm = StubLLM(['说明：{"queries":["local Qwen3"]} 完成'])
    plan = SearchPlanner(llm).plan("Qwen3")
    assert plan.queries == ["local Qwen3"]


def test_planner_limits_query_count() -> None:
    queries = [f"query {index}" for index in range(10)]
    llm = StubLLM([f'{{"queries": {queries!r}}}'.replace("'", '"')])
    plan = SearchPlanner(llm, max_queries=3).plan("broad question")
    assert plan.queries == ["query 0", "query 1", "query 2"]


def test_planner_removes_exact_query_duplicates() -> None:
    llm = StubLLM(['{"queries":["query a"," query a ","query b"]}'])
    plan = SearchPlanner(llm, max_queries=3).plan("question")
    assert plan.queries == ["query a", "query b"]


def test_planner_retries_once_then_raises_clear_error() -> None:
    llm = StubLLM(["not json", '{"queries":[]}'])
    with pytest.raises(PlannerError, match="连续两次"):
        SearchPlanner(llm).plan("test")
    assert llm.calls == 2


def test_planner_rejects_blank_question_without_calling_llm() -> None:
    llm = StubLLM(['{"queries":["unused"]}'])
    with pytest.raises(PlannerError, match="不能为空"):
        SearchPlanner(llm).plan("   ")
    assert llm.calls == 0
