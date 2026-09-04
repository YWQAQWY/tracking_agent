"""Create evidence-linked atomic claims instead of free-form prose."""

from src.grounding.models import AnswerSection, GroundedAnswerDraft
from src.grounding.registry import EvidenceRegistry
from src.llm.client import LLMClient
from src.llm.structured import request_structured


SYSTEM_PROMPT = """You are a grounded research answer planner.

Construct a structured answer using only the supplied evidence. Do not write
final prose. Break the answer into atomic factual claims.

Rules:
1. Every factual claim must be supported by one or more supplied evidence IDs.
2. Never create a factual claim without evidence or use outside knowledge.
3. Use evidence IDs exactly as provided.
4. Prefer precise claims; split compound statements when support differs.
5. If an aspect lacks evidence, omit the factual answer instead of guessing.
6. Do not put citations inside claim text.
7. Use the user's language.
8. It is valid to return {"sections":[]} when no factual claim is supported.

Output only this JSON shape:
{"sections":[{"heading":"optional heading or null","claims":[
{"id":"C1","text":"one factual claim","evidence_ids":["E1"]}
]}]}"""


class GroundedAnswerPlanner:
    """Plan a bounded set of claims from the original question and final evidence."""

    def __init__(
        self,
        llm: LLMClient,
        max_claims: int = 30,
        max_evidence_per_claim: int = 3,
        max_chars_per_evidence: int = 1_200,
        max_total_context_chars: int = 15_000,
    ) -> None:
        self.llm = llm
        self.max_claims = max_claims
        self.max_evidence_per_claim = max_evidence_per_claim
        self.max_chars_per_evidence = max_chars_per_evidence
        self.max_total_context_chars = max_total_context_chars

    def plan(
        self, question: str, evidence_registry: EvidenceRegistry
    ) -> GroundedAnswerDraft:
        context = evidence_registry.format(
            max_chars_per_evidence=self.max_chars_per_evidence,
            max_total_chars=self.max_total_context_chars,
        )
        prompt = (
            f"ORIGINAL QUESTION:\n{question}\n\n"
            f"FINAL EVIDENCE:\n{context}\n\n"
            f"Create at most {self.max_claims} claims. Each claim may cite at most "
            f"{self.max_evidence_per_claim} evidence IDs."
        )
        draft = request_structured(
            self.llm,
            prompt,
            SYSTEM_PROMPT,
            GroundedAnswerDraft,
            "GroundedAnswerPlanner",
        )
        return self._enforce_limits(draft)

    def _enforce_limits(self, draft: GroundedAnswerDraft) -> GroundedAnswerDraft:
        remaining = self.max_claims
        sections: list[AnswerSection] = []
        for section in draft.sections:
            claims = [
                claim.model_copy(
                    update={
                        "evidence_ids": claim.evidence_ids[
                            : self.max_evidence_per_claim
                        ]
                    }
                )
                for claim in section.claims[:remaining]
            ]
            if claims:
                sections.append(AnswerSection(heading=section.heading, claims=claims))
                remaining -= len(claims)
            if remaining == 0:
                break
        return GroundedAnswerDraft(sections=sections)
