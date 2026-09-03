import pytest

from src.agent.critic import CriticError, EvidenceCritic
from src.agent.critic import SYSTEM_PROMPT
from src.agent.critic_context import CriticContextBuilder
from src.models.evidence import Evidence


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def chat(self, prompt: str, system: str) -> str:
        self.prompts.append(f"{system}\n{prompt}")
        return self.responses.pop(0)


def evidence() -> Evidence:
    return Evidence("method evidence", "https://a.example", "A", 0, 0.8, 0.9)


def test_critic_parses_structured_gap_result() -> None:
    llm = FakeLLM(
        [
            """```json
            {"sufficient": false, "missing_aspects": ["limitations"],
             "follow_up_queries": ["robot manipulation RL limitations"],
             "reason": "Limitations are missing."}
            ```"""
        ]
    )
    critic = EvidenceCritic(llm, CriticContextBuilder())

    result = critic.evaluate("question", [evidence()], ["initial query"])

    assert result.sufficient is False
    assert result.missing_aspects == ["limitations"]
    assert result.follow_up_queries == ["robot manipulation RL limitations"]
    assert "SEARCHES ALREADY PERFORMED" in llm.prompts[0]
    assert "method evidence" in llm.prompts[0]


def test_critic_parses_sufficient_result() -> None:
    critic = EvidenceCritic(
        FakeLLM(
            [
                '{"sufficient": true, "missing_aspects": [], '
                '"follow_up_queries": [], "reason": "Covered."}'
            ]
        ),
        CriticContextBuilder(),
    )
    assert critic.evaluate("question", [evidence()], []).sufficient is True


def test_critic_repairs_once_then_raises_typed_error() -> None:
    critic = EvidenceCritic(
        FakeLLM(["not json", "still not json"]), CriticContextBuilder()
    )
    with pytest.raises(CriticError, match="连续两次"):
        critic.evaluate("question", [evidence()], [])


def test_critic_context_is_bounded() -> None:
    builder = CriticContextBuilder(
        max_evidence=1, max_chars_per_evidence=10, max_total_chars=100
    )
    context = builder.build([evidence(), evidence()])
    assert context.count("EVIDENCE") == 1
    assert "method evi" in context


def test_critic_prompt_forbids_expanding_beyond_the_question() -> None:
    assert "only from what the user explicitly asks" in SYSTEM_PROMPT
    assert "missing aspect must map" in SYSTEM_PROMPT
