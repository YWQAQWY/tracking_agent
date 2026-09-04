"""Check completeness after unsupported claims have been removed."""

from src.grounding.models import AnswerClaim, CoverageResult
from src.llm.client import LLMClient
from src.llm.structured import request_structured


SYSTEM_PROMPT = """You check whether verified factual claims adequately answer
the user's original question. Do not add facts and do not ask for more search.
Judge only the user's actual requested aspects, not exhaustive world knowledge.
Report evidence-backed missing aspects as limitations.

Output only:
{"adequate":true,"covered_aspects":["aspect"],"missing_aspects":[],
"reason":"brief coverage judgment"}"""


class AnswerCoverageChecker:
    """Evaluate answer completeness without invoking search."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def check(self, question: str, claims: list[AnswerClaim]) -> CoverageResult:
        claims_text = "\n".join(f"{item.id}: {item.text}" for item in claims)
        return request_structured(
            self.llm,
            f"ORIGINAL QUESTION:\n{question}\n\nVERIFIED CLAIMS:\n{claims_text}",
            SYSTEM_PROMPT,
            CoverageResult,
            "AnswerCoverageChecker",
        )
