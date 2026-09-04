"""Resolve unsupported claims with one bounded rewrite-and-verify pass."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.grounding.models import (
    AnswerClaim,
    AnswerSection,
    ClaimTrace,
    GroundedAnswerDraft,
    VerifiedAnswerDraft,
)
from src.grounding.registry import EvidenceRegistry
from src.grounding.rewriter import ClaimRewriter
from src.grounding.verifier import CitationVerifier
from src.llm.client import LLMError
from src.llm.structured import StructuredOutputError


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GroundingResolution:
    draft: VerifiedAnswerDraft
    claim_traces: tuple[ClaimTrace, ...]
    initially_supported_count: int
    unsupported_count: int
    rewritten_count: int
    rewrite_passed_count: int
    dropped_count: int


class GroundingService:
    """Verify all claims, rewrite unsupported claims once, then drop failures."""

    def __init__(
        self,
        verifier: CitationVerifier,
        rewriter: ClaimRewriter,
        max_rewrite_attempts: int = 1,
    ) -> None:
        if max_rewrite_attempts not in {0, 1}:
            raise ValueError("V0.6 MAX_CLAIM_REWRITE_ATTEMPTS 只支持 0 或 1")
        self.verifier = verifier
        self.rewriter = rewriter
        self.max_rewrite_attempts = max_rewrite_attempts

    def resolve(
        self, draft: GroundedAnswerDraft, registry: EvidenceRegistry
    ) -> GroundingResolution:
        claims = draft.claims
        verification = self.verifier.verify_many(claims, registry)
        decisions = dict(zip((item.id for item in claims), verification, strict=True))
        final: dict[str, AnswerClaim] = {}
        reasons: dict[str, str] = {}
        rewritten_ids: set[str] = set()

        for claim in claims:
            result = decisions[claim.id]
            reasons[claim.id] = result.reason
            if result.supported:
                final[claim.id] = claim.model_copy(
                    update={"evidence_ids": result.supported_evidence_ids}
                )

        unsupported = [claim for claim in claims if claim.id not in final]
        rewritten_count = 0
        rewrite_passed = 0
        if self.max_rewrite_attempts and unsupported:
            rewritable = [
                claim
                for claim in unsupported
                if claim.evidence_ids
                and all(registry.contains(item) for item in claim.evidence_ids)
            ]
            candidates: list[AnswerClaim | None] = []
            if rewritable:
                try:
                    candidates = self.rewriter.rewrite_many(
                        rewritable,
                        [decisions[item.id] for item in rewritable],
                        registry,
                    )
                except (LLMError, StructuredOutputError) as exc:
                    logger.warning(
                        "Claim rewrite failed; unsupported claims will drop: %s", exc
                    )
                    candidates = [None] * len(rewritable)

            rewritten = [item for item in candidates if item is not None]
            rewritten_count = len(rewritten)
            rewritten_ids = {item.id for item in rewritten}
            if rewritten:
                second_pass = self.verifier.verify_many(rewritten, registry)
                for claim, result in zip(rewritten, second_pass, strict=True):
                    reasons[claim.id] = result.reason
                    if result.supported:
                        final[claim.id] = claim.model_copy(
                            update={"evidence_ids": result.supported_evidence_ids}
                        )
                        rewrite_passed += 1

        sections = tuple(
            AnswerSection(
                heading=section.heading,
                claims=[
                    final[claim.id]
                    for claim in section.claims
                    if claim.id in final
                ],
            )
            for section in draft.sections
            if any(claim.id in final for claim in section.claims)
        )
        traces = tuple(
            ClaimTrace(
                claim_id=claim.id,
                original_text=claim.text,
                final_text=final[claim.id].text if claim.id in final else None,
                evidence_ids=tuple(
                    final[claim.id].evidence_ids
                    if claim.id in final
                    else claim.evidence_ids
                ),
                supported=claim.id in final,
                rewritten=claim.id in rewritten_ids,
                reason=reasons[claim.id],
            )
            for claim in claims
        )
        return GroundingResolution(
            draft=VerifiedAnswerDraft(sections),
            claim_traces=traces,
            initially_supported_count=len(claims) - len(unsupported),
            unsupported_count=len(unsupported),
            rewritten_count=rewritten_count,
            rewrite_passed_count=rewrite_passed,
            dropped_count=len(claims) - len(final),
        )
