"""One-shot weakening of unsupported claims."""

from src.grounding.models import AnswerClaim, ClaimVerificationResult, RewriteBatch
from src.grounding.registry import EvidenceRegistry
from src.llm.client import LLMClient
from src.llm.structured import request_structured


SYSTEM_PROMPT = """You rewrite unsupported factual claims conservatively.

For each claim, use only its supplied cited evidence. Rewrite it into one atomic
claim fully entailed by that evidence. If no meaningful supported claim can be
made, return rewritten_text as null. Never add outside facts or new evidence IDs.

Output only:
{"rewrites":[{"claim_id":"C1","rewritten_text":"weaker claim or null",
"evidence_ids":["E1"]}]}"""


class ClaimRewriter:
    """Rewrite a batch once; retry policy belongs to GroundingService."""

    def __init__(
        self,
        llm: LLMClient,
        max_chars_per_evidence: int = 1_200,
        max_total_context_chars: int = 15_000,
    ) -> None:
        self.llm = llm
        self.max_chars_per_evidence = max_chars_per_evidence
        self.max_total_context_chars = max_total_context_chars

    def rewrite_many(
        self,
        claims: list[AnswerClaim],
        verifications: list[ClaimVerificationResult],
        registry: EvidenceRegistry,
    ) -> list[AnswerClaim | None]:
        if not claims:
            return []
        reasons = {item.claim_id: item.reason for item in verifications}
        ids = list(
            dict.fromkeys(item for claim in claims for item in claim.evidence_ids)
        )
        claim_text = "\n\n".join(
            f"{claim.id}\nOriginal: {claim.text}\n"
            f"Cited: {', '.join(claim.evidence_ids)}\n"
            f"Verifier: {reasons.get(claim.id, 'unsupported')}"
            for claim in claims
        )
        response = request_structured(
            self.llm,
            f"UNSUPPORTED CLAIMS:\n{claim_text}\n\nEVIDENCE:\n"
            + registry.format(
                ids, self.max_chars_per_evidence, self.max_total_context_chars
            ),
            SYSTEM_PROMPT,
            RewriteBatch,
            "ClaimRewriter",
        )
        returned = {item.claim_id: item for item in response.rewrites}
        rewritten: list[AnswerClaim | None] = []
        for claim in claims:
            item = returned.get(claim.id)
            allowed = set(claim.evidence_ids)
            if item is None or item.rewritten_text is None:
                rewritten.append(None)
                continue
            evidence_ids = [
                evidence_id
                for evidence_id in item.evidence_ids
                if evidence_id in allowed and registry.contains(evidence_id)
            ]
            rewritten.append(
                AnswerClaim(
                    id=claim.id,
                    text=item.rewritten_text,
                    evidence_ids=evidence_ids,
                )
                if evidence_ids
                else None
            )
        return rewritten
