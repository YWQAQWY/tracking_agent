"""LLM-backed evaluator for evidence sufficiency and targeted gap search."""

from __future__ import annotations

from json import JSONDecodeError

from pydantic import ValidationError

from src.agent.critic_context import CriticContextBuilder
from src.agent.models import CriticResult
from src.llm.client import LLMClient, LLMError
from src.llm.structured import parse_json_object
from src.models.evidence import Evidence


class CriticError(RuntimeError):
    """Raised when the critic cannot produce a valid structured decision."""


SYSTEM_PROMPT = """You are an evidence critic for a web research agent.

Your job is NOT to answer the user's question. Evaluate whether the current
evidence is sufficient to answer it accurately and comprehensively.

Evaluate:
1. Coverage of the user's major sub-questions.
2. Whether important claims have supporting evidence.
3. Whether evidence is diverse enough for the question.
4. Whether critical missing aspects remain.
5. Whether another web search would materially improve the answer.

Judge practical sufficiency for this question, not whether more information
exists somewhere in the world. Do not demand exhaustive or perfect coverage.
Derive the coverage checklist only from what the user explicitly asks and the
minimum facts needed to answer it. Do not invent extra scope such as history,
comparisons, applications, equations, taxonomies, or limitations unless the
question requests them or the answer would be incorrect without them. A narrow
definition question can be sufficient with concise definition evidence. Every
missing aspect must map to a specific requirement in the original question; if
you cannot name that mapping, do not request another search.

Mandatory scope example: for a question shaped only as "What is X?", evidence
that accurately defines X and its basic mechanism is sufficient. You MUST mark
it sufficient; do not demand comparisons, applications, history, equations,
algorithm lists, implementation details, or limitations that were not asked.

Scope example:
- Question: "What is reinforcement learning?"
- Evidence: it defines agent/environment interaction, actions, rewards, policy,
  cumulative reward, exploration, and exploitation.
- Required decision: sufficient=true, with empty gaps and queries. MDP equations,
  comparisons, applications, algorithm catalogs, and limitations are out of scope.
If evidence is insufficient, identify concrete gaps and propose 1-3 targeted,
search-engine-friendly follow-up queries. Never repeat an executed query and
never use a vague general-topic query.

Output only one JSON object with this shape:
{"sufficient": false, "confidence": 0.7,
 "missing_aspects": ["specific gap"],
 "follow_up_queries": ["targeted query"],
 "reason": "brief coverage judgment"}
Do not output an answer, reasoning transcript, Markdown, or code fences."""


class EvidenceCritic:
    """Evaluate pooled evidence with one bounded structured-output repair."""

    def __init__(self, llm: LLMClient, context_builder: CriticContextBuilder) -> None:
        self.llm = llm
        self.context_builder = context_builder

    def evaluate(
        self,
        question: str,
        evidence: list[Evidence],
        executed_queries: list[str],
    ) -> CriticResult:
        context = self.context_builder.build(evidence)
        searches = "\n".join(
            f"{index}. {query}"
            for index, query in enumerate(executed_queries, start=1)
        ) or "(none)"
        prompt = (
            f"ORIGINAL QUESTION:\n{question}\n\n"
            f"SEARCHES ALREADY PERFORMED:\n{searches}\n\n"
            f"CURRENT EVIDENCE:\n{context or '(no evidence)'}\n\n"
            "FINAL SCOPE CHECK: Judge only requirements in the exact original "
            "question. Do not expand its scope. If it only asks what something "
            "is, a supported definition and basic mechanism are sufficient."
        )
        raw = ""
        try:
            raw = self.llm.chat(prompt, SYSTEM_PROMPT)
            return self._parse(raw)
        except LLMError:
            raise
        except (JSONDecodeError, ValidationError, TypeError, ValueError):
            repair = (
                "The prior response was invalid. Return only a valid JSON object "
                "with sufficient, confidence, missing_aspects, "
                "follow_up_queries, and reason.\n"
                f"Prior response:\n{raw}"
            )
            try:
                return self._parse(self.llm.chat(repair, SYSTEM_PROMPT))
            except LLMError:
                raise
            except (JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                raise CriticError(
                    "本地模型连续两次未生成合法 CriticResult JSON。"
                ) from exc

    @staticmethod
    def _parse(raw: str) -> CriticResult:
        return CriticResult.model_validate(parse_json_object(raw))
