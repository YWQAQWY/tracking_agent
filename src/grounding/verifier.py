"""Strict claim-to-cited-evidence support verification."""

from src.grounding.models import (
    AnswerClaim,
    ClaimVerificationResult,
    VerificationBatch,
)
from src.grounding.registry import EvidenceRegistry
from src.llm.client import LLMClient
from src.llm.structured import request_structured


SYSTEM_PROMPT = """You are a strict claim-evidence verifier.

Decide whether each claim is supported by its cited evidence. Support means the
evidence directly states or reasonably entails the specific factual content.

Rules:
1. Topic relevance is not support.
2. If a claim is stronger than the evidence, mark it unsupported.
3. If only part of a compound claim is supported, mark the claim unsupported.
4. Use no outside knowledge and evaluate only the supplied evidence.
5. supported_evidence_ids must contain only cited IDs that actually support it.

Output only:
{"results":[{"claim_id":"C1","supported":true,"support_score":0.9,
"supported_evidence_ids":["E1"],"reason":"brief support judgment"}]}"""


class CitationVerifier:
    """Batch verification with safe local rejection of invalid Evidence IDs."""

    def __init__(
        self,
        llm: LLMClient,
        batch_size: int = 8,
        max_chars_per_evidence: int = 1_200,
        max_total_context_chars: int = 15_000,
    ) -> None:
        self.llm = llm
        self.batch_size = batch_size
        self.max_chars_per_evidence = max_chars_per_evidence
        self.max_total_context_chars = max_total_context_chars

    def verify_many(
        self, claims: list[AnswerClaim], registry: EvidenceRegistry
    ) -> list[ClaimVerificationResult]:
        results: list[ClaimVerificationResult] = []
        valid: list[AnswerClaim] = []
        for claim in claims:
            unknown = [
                item for item in claim.evidence_ids if not registry.contains(item)
            ]
            if not claim.evidence_ids or unknown:
                detail = (
                    f"unknown evidence IDs: {', '.join(unknown)}"
                    if unknown
                    else "no evidence IDs"
                )
                results.append(
                    ClaimVerificationResult(
                        claim_id=claim.id,
                        supported=False,
                        supported_evidence_ids=[],
                        reason=detail,
                    )
                )
            else:
                valid.append(claim)

        for start in range(0, len(valid), self.batch_size):
            batch = valid[start : start + self.batch_size]
            results.extend(self._verify_batch(batch, registry))

        by_id = {result.claim_id: result for result in results}
        return [by_id[claim.id] for claim in claims]

    def _verify_batch(
        self, claims: list[AnswerClaim], registry: EvidenceRegistry
    ) -> list[ClaimVerificationResult]:
        evidence_ids = list(
            dict.fromkeys(item for claim in claims for item in claim.evidence_ids)
        )
        claims_text = "\n".join(
            f"{claim.id}: {claim.text}\nCited: {', '.join(claim.evidence_ids)}"
            for claim in claims
        )
        evidence_text = registry.format(
            evidence_ids,
            self.max_chars_per_evidence,
            self.max_total_context_chars,
        )
        response = request_structured(
            self.llm,
            f"CLAIMS:\n{claims_text}\n\nCITED EVIDENCE:\n{evidence_text}",
            SYSTEM_PROMPT,
            VerificationBatch,
            "CitationVerifier",
        )
        returned = {item.claim_id: item for item in response.results}
        normalized: list[ClaimVerificationResult] = []
        for claim in claims:
            result = returned.get(claim.id)
            if result is None:
                normalized.append(
                    ClaimVerificationResult(
                        claim_id=claim.id,
                        supported=False,
                        supported_evidence_ids=[],
                        reason="verifier omitted this claim",
                    )
                )
                continue
            allowed = set(claim.evidence_ids)
            supported_ids = [
                item
                for item in result.supported_evidence_ids
                if item in allowed and registry.contains(item)
            ]
            supported = result.supported and bool(supported_ids)
            normalized.append(
                result.model_copy(
                    update={
                        "supported": supported,
                        "supported_evidence_ids": supported_ids,
                        "reason": result.reason
                        if supported or not result.supported
                        else (
                            "supported result did not identify valid supporting "
                            "evidence"
                        ),
                    }
                )
            )
        return normalized
