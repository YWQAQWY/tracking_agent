"""Coordinate claim planning, grounding, coverage, and safe rendering."""

from __future__ import annotations

import logging
import time

from src.grounding.coverage import AnswerCoverageChecker
from src.grounding.models import CoverageResult, GroundedAnswer, GroundingTrace
from src.grounding.planner import GroundedAnswerPlanner
from src.grounding.registry import EvidenceRegistry
from src.grounding.renderer import GroundedAnswerRenderer
from src.grounding.service import GroundingService
from src.models.evidence import Evidence


logger = logging.getLogger(__name__)


class GroundedGenerationError(RuntimeError):
    """Raised when planning or verification cannot establish a grounded answer."""


class GroundedAnswerGenerator:
    """Small orchestrator for the V0.6 generation layer only."""

    def __init__(
        self,
        planner: GroundedAnswerPlanner,
        grounding_service: GroundingService,
        coverage_checker: AnswerCoverageChecker,
        renderer: GroundedAnswerRenderer,
        enable_coverage_check: bool = True,
    ) -> None:
        self.planner = planner
        self.grounding_service = grounding_service
        self.coverage_checker = coverage_checker
        self.renderer = renderer
        self.enable_coverage_check = enable_coverage_check

    def generate(self, question: str, evidence: list[Evidence]) -> GroundedAnswer:
        timings: dict[str, float] = {}
        registry = EvidenceRegistry(evidence)
        logger.info("Preparing %d final evidence items", len(evidence))
        try:
            started = time.perf_counter()
            draft = self.planner.plan(question, registry)
            timings["claim_planning"] = time.perf_counter() - started
            logger.info(
                "Grounded answer planner generated %d factual claims",
                len(draft.claims),
            )

            started = time.perf_counter()
            resolution = self.grounding_service.resolve(draft, registry)
            timings["claim_grounding"] = time.perf_counter() - started
        except Exception as exc:
            raise GroundedGenerationError(
                "grounded planning/verification failed: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc

        logger.info(
            "%d claims verified; %d unsupported; %d rewritten; %d dropped",
            len(resolution.draft.claims),
            resolution.unsupported_count,
            resolution.rewritten_count,
            resolution.dropped_count,
        )
        coverage: CoverageResult | None = None
        if self.enable_coverage_check:
            started = time.perf_counter()
            try:
                coverage = self.coverage_checker.check(
                    question, resolution.draft.claims
                )
            except Exception as exc:
                logger.warning(
                    "Coverage check failed; rendering verified claims: %s", exc
                )
            timings["coverage"] = time.perf_counter() - started

        started = time.perf_counter()
        text, sources = self.renderer.render(
            question, resolution.draft, coverage, registry
        )
        timings["render"] = time.perf_counter() - started
        trace = GroundingTrace(
            draft_claim_count=len(draft.claims),
            initially_supported_claim_count=resolution.initially_supported_count,
            unsupported_claim_count=resolution.unsupported_count,
            rewritten_claim_count=resolution.rewritten_count,
            rewrite_passed_claim_count=resolution.rewrite_passed_count,
            dropped_claim_count=resolution.dropped_count,
            verified_claim_count=len(resolution.draft.claims),
            cited_source_count=len(sources),
            claim_traces=resolution.claim_traces,
            coverage=coverage,
            timings=timings,
        )
        logger.info("Final grounded answer uses %d unique sources", len(sources))
        return GroundedAnswer(text, sources, trace)
