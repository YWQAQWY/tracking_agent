import pytest

from src.answer.answer_generator import AnswerGenerationError, AnswerGenerator
from src.models.search_result import SearchResult


class CapturingLLM:
    def __init__(self) -> None:
        self.user_prompt = ""
        self.system_prompt = ""

    def chat(self, user_prompt: str, system_prompt: str | None = None) -> str:
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt or ""
        return "基于结果的回答 [1]"


def test_answer_generator_formats_numbered_context() -> None:
    llm = CapturingLLM()
    result = SearchResult(title="Title", url="https://example.com", snippet="Summary")
    answer = AnswerGenerator(llm).generate("问题", [result])
    assert answer == "基于结果的回答 [1]"
    assert "[1]" in llm.user_prompt
    assert "Title: Title" in llm.user_prompt
    assert "Snippet: Summary" in llm.user_prompt
    assert "不要虚构" in llm.system_prompt


def test_answer_generator_rejects_empty_results() -> None:
    with pytest.raises(AnswerGenerationError, match="搜索没有"):
        AnswerGenerator(CapturingLLM()).generate("问题", [])

