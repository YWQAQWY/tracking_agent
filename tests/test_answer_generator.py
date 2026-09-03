import pytest

from src.answer.answer_generator import AnswerGenerationError, AnswerGenerator


class CapturingLLM:
    def __init__(self) -> None:
        self.user_prompt = ""
        self.system_prompt = ""

    def chat(self, user_prompt: str, system_prompt: str | None = None) -> str:
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt or ""
        return "基于结果的回答 [1]"


def test_answer_generator_uses_prepared_document_context() -> None:
    llm = CapturingLLM()
    context = (
        "[Source 1]\nTitle: Title\nURL: https://example.com\n"
        "Content:\nFull article text"
    )
    answer = AnswerGenerator(llm).generate("问题", context)
    assert answer == "基于结果的回答 [1]"
    assert "[Source 1]" in llm.user_prompt
    assert "Title: Title" in llm.user_prompt
    assert "Full article text" in llm.user_prompt
    assert "Do not invent" in llm.system_prompt


def test_answer_generator_rejects_empty_context() -> None:
    with pytest.raises(AnswerGenerationError, match="上下文为空"):
        AnswerGenerator(CapturingLLM()).generate("问题", "  ")

