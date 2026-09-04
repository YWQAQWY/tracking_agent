from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.grounding.coverage import AnswerCoverageChecker
from src.grounding.generator import GroundedAnswerGenerator, GroundedGenerationError
from src.grounding.models import (
    AnswerClaim,
    AnswerSection,
    ClaimRewrite,
    ClaimTrace,
    ClaimVerificationResult,
    CoverageResult,
    GroundedAnswerDraft,
    RewriteBatch,
    VerifiedAnswerDraft,
)
from src.grounding.planner import GroundedAnswerPlanner
from src.grounding.registry import EvidenceRegistry
from src.grounding.renderer import GroundedAnswerRenderer
from src.grounding.service import GroundingService
from src.grounding.verifier import CitationVerifier
from src.models.evidence import Evidence


def evidence(name: str, url: str | None = None, chunk: int = 0) -> Evidence:
    return Evidence(
        text=f"Evidence text for {name}",
        url=url or f"https://{name.lower()}.example",
        title=name,
        chunk_index=chunk,
        embedding_score=0.8,
        rerank_score=0.9,
    )


class FakeLLM:
    def __init__(self, responses: Sequence[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str | None]] = []

    def chat(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeVerifier:
    def __init__(self, responses: Sequence[list[ClaimVerificationResult]]) -> None:
        self.responses = list(responses)
        self.calls: list[list[AnswerClaim]] = []

    def verify_many(
        self, claims: list[AnswerClaim], registry: EvidenceRegistry
    ) -> list[ClaimVerificationResult]:
        self.calls.append(claims)
        return self.responses.pop(0)


class FakeRewriter:
    def __init__(self, responses: list[AnswerClaim | None]) -> None:
        self.responses = responses
        self.calls = 0

    def rewrite_many(self, claims, verifications, registry):
        self.calls += 1
        return self.responses


def claim(
    claim_id: str = "C1", text: str = "Supported fact", ids: list[str] | None = None
) -> AnswerClaim:
    return AnswerClaim(id=claim_id, text=text, evidence_ids=ids or ["E1"])


def result(
    claim_id: str = "C1",
    supported: bool = True,
    ids: list[str] | None = None,
    reason: str = "direct support",
) -> ClaimVerificationResult:
    return ClaimVerificationResult(
        claim_id=claim_id,
        supported=supported,
        supported_evidence_ids=(ids if ids is not None else (["E1"] if supported else [])),
        reason=reason,
    )


def draft(*claims: AnswerClaim) -> GroundedAnswerDraft:
    return GroundedAnswerDraft(
        sections=[AnswerSection(heading="Findings", claims=list(claims))]
    )


def test_evidence_registry_assigns_stable_ids_and_safe_unknown_lookup() -> None:
    items = [evidence("A"), evidence("B"), evidence("C")]
    registry = EvidenceRegistry(items)
    assert registry.ids() == ("E1", "E2", "E3")
    assert registry.get("E2") is items[1]
    assert registry.get("E99") is None
    assert "[E1]" in registry.format()


def test_answer_claim_normalizes_evidence_ids() -> None:
    item = AnswerClaim(id=" C1 ", text=" fact ", evidence_ids=[" E1 ", "E1"])
    assert item.id == "C1"
    assert item.text == "fact"
    assert item.evidence_ids == ["E1"]


def test_empty_draft_is_valid_when_evidence_supports_no_claim() -> None:
    assert GroundedAnswerDraft(sections=[]).claims == []


def test_draft_accepts_common_single_section_wrapper_omission() -> None:
    parsed = GroundedAnswerDraft.model_validate(
        {"heading": "H", "claims": [{"id": "C1", "text": "A", "evidence_ids": ["E1"]}]}
    )
    assert parsed.sections[0].heading == "H"


def test_grounded_planner_parses_claims_and_enforces_limits() -> None:
    llm = FakeLLM(
        [
            '{"sections":[{"heading":"H","claims":['
            '{"id":"C1","text":"A","evidence_ids":["E1","E2"]},'
            '{"id":"C2","text":"B","evidence_ids":["E2"]}]}]}'
        ]
    )
    planned = GroundedAnswerPlanner(
        llm, max_claims=1, max_evidence_per_claim=1
    ).plan("question", EvidenceRegistry([evidence("A"), evidence("B")]))
    assert [item.id for item in planned.claims] == ["C1"]
    assert planned.claims[0].evidence_ids == ["E1"]
    assert "ORIGINAL QUESTION" in llm.calls[0][0]


def test_structured_planner_repairs_one_invalid_response() -> None:
    llm = FakeLLM(
        [
            "not json",
            '{"sections":[{"heading":null,"claims":['
            '{"id":"C1","text":"A","evidence_ids":["E1"]}]}]}',
        ]
    )
    planned = GroundedAnswerPlanner(llm).plan(
        "question", EvidenceRegistry([evidence("A")])
    )
    assert len(planned.claims) == 1
    assert len(llm.calls) == 2


def test_planner_parse_failure_becomes_grounded_generation_error() -> None:
    generator = GroundedAnswerGenerator(
        GroundedAnswerPlanner(FakeLLM(["bad", "still bad"])),
        GroundingService(FakeVerifier([]), FakeRewriter([])),
        StaticCoverage(RuntimeError("unused")),
        GroundedAnswerRenderer(),
    )
    with pytest.raises(GroundedGenerationError, match="GroundedAnswerPlanner"):
        generator.generate("question", [evidence("A")])


def test_verifier_accepts_supported_claim_and_normalizes_ids() -> None:
    llm = FakeLLM(
        ['{"results":[{"claim_id":"C1","supported":true,'
         '"support_score":0.9,"supported_evidence_ids":["E1","E99"],'
         '"reason":"stated directly"}]}']
    )
    verified = CitationVerifier(llm).verify_many(
        [claim()], EvidenceRegistry([evidence("A")])
    )
    assert verified[0].supported is True
    assert verified[0].supported_evidence_ids == ["E1"]


def test_unknown_evidence_id_is_unsupported_without_llm_call() -> None:
    llm = FakeLLM([])
    verified = CitationVerifier(llm).verify_many(
        [claim(ids=["E99"])], EvidenceRegistry([evidence("A")])
    )
    assert verified[0].supported is False
    assert "unknown" in verified[0].reason
    assert llm.calls == []


def test_verifier_topic_relevance_can_be_marked_unsupported() -> None:
    llm = FakeLLM(
        ['{"results":[{"claim_id":"C1","supported":false,'
         '"support_score":0.2,"supported_evidence_ids":[],'
         '"reason":"same topic but no percentage"}]}']
    )
    verified = CitationVerifier(llm).verify_many(
        [claim(text="Method improves success by 20%")],
        EvidenceRegistry([evidence("A")]),
    )
    assert not verified[0].supported


def test_verifier_batches_claims() -> None:
    llm = FakeLLM(
        [
            '{"results":[{"claim_id":"C1","supported":true,'
            '"support_score":null,"supported_evidence_ids":["E1"],"reason":"yes"}]}' ,
            '{"results":[{"claim_id":"C2","supported":true,'
            '"support_score":null,"supported_evidence_ids":["E1"],"reason":"yes"}]}' ,
        ]
    )
    results = CitationVerifier(llm, batch_size=1).verify_many(
        [claim("C1"), claim("C2")], EvidenceRegistry([evidence("A")])
    )
    assert len(results) == 2
    assert len(llm.calls) == 2


def test_verifier_parse_failure_becomes_grounded_generation_error() -> None:
    generator = GroundedAnswerGenerator(
        StaticPlanner(),
        GroundingService(
            CitationVerifier(FakeLLM(["bad", "still bad"])), FakeRewriter([])
        ),
        StaticCoverage(RuntimeError("unused")),
        GroundedAnswerRenderer(),
    )
    with pytest.raises(GroundedGenerationError, match="CitationVerifier"):
        generator.generate("question", [evidence("A")])


def test_rewriter_returns_supported_subset_or_null() -> None:
    llm = FakeLLM(
        ['{"rewrites":['
         '{"claim_id":"C1","rewritten_text":"Weaker fact","evidence_ids":["E1","E9"]},'
         '{"claim_id":"C2","rewritten_text":null,"evidence_ids":[]}]}']
    )
    from src.grounding.rewriter import ClaimRewriter

    rewritten = ClaimRewriter(llm).rewrite_many(
        [claim("C1"), claim("C2")],
        [result("C1", False), result("C2", False)],
        EvidenceRegistry([evidence("A")]),
    )
    assert rewritten[0] == claim("C1", "Weaker fact")
    assert rewritten[1] is None


def test_grounding_service_keeps_supported_claim() -> None:
    service = GroundingService(FakeVerifier([[result()]]), FakeRewriter([]))
    resolution = service.resolve(draft(claim()), EvidenceRegistry([evidence("A")]))
    assert resolution.draft.claims == [claim()]
    assert resolution.initially_supported_count == 1
    assert resolution.dropped_count == 0


def test_grounding_service_rewrite_success_is_verified_once() -> None:
    verifier = FakeVerifier(
        [[result(supported=False)], [result(supported=True)]]
    )
    rewritten = claim(text="Weaker fact")
    service = GroundingService(verifier, FakeRewriter([rewritten]))
    resolution = service.resolve(draft(claim()), EvidenceRegistry([evidence("A")]))
    assert resolution.draft.claims == [rewritten]
    assert resolution.rewritten_count == 1
    assert resolution.rewrite_passed_count == 1
    assert len(verifier.calls) == 2


@pytest.mark.parametrize("rewrite", [None, claim(text="Still unsupported")])
def test_grounding_service_drops_no_or_failed_rewrite(
    rewrite: AnswerClaim | None,
) -> None:
    responses = [[result(supported=False)]]
    if rewrite is not None:
        responses.append([result(supported=False)])
    service = GroundingService(FakeVerifier(responses), FakeRewriter([rewrite]))
    resolution = service.resolve(draft(claim()), EvidenceRegistry([evidence("A")]))
    assert resolution.draft.claims == []
    assert resolution.dropped_count == 1
    assert resolution.claim_traces[0].supported is False


def test_max_rewrite_attempts_zero_never_calls_rewriter() -> None:
    rewriter = FakeRewriter([claim(text="unused")])
    service = GroundingService(
        FakeVerifier([[result(supported=False)]]),
        rewriter,
        max_rewrite_attempts=0,
    )
    service.resolve(draft(claim()), EvidenceRegistry([evidence("A")]))
    assert rewriter.calls == 0


def test_unknown_evidence_claim_is_dropped_without_rewrite() -> None:
    rewriter = FakeRewriter([claim(text="unused")])
    service = GroundingService(
        FakeVerifier([[result(supported=False, reason="unknown")]]), rewriter
    )
    resolution = service.resolve(
        draft(claim(ids=["E99"])), EvidenceRegistry([evidence("A")])
    )
    assert resolution.dropped_count == 1
    assert rewriter.calls == 0


def test_coverage_checker_parses_adequate_and_missing_results() -> None:
    llm = FakeLLM(
        ['{"adequate":false,"covered_aspects":["methods"],'
         '"missing_aspects":["limitations"],"reason":"missing limitations"}']
    )
    coverage = AnswerCoverageChecker(llm).check("methods and limitations", [claim()])
    assert not coverage.adequate
    assert coverage.missing_aspects == ["limitations"]


def test_renderer_maps_citations_by_url_deterministically_and_omits_unused() -> None:
    same_url = "https://a.example"
    registry = EvidenceRegistry(
        [
            evidence("A1", same_url, 0),
            evidence("A2", same_url, 1),
            evidence("Unused", "https://unused.example"),
            evidence("B", "https://b.example"),
        ]
    )
    verified = VerifiedAnswerDraft(
        (
            AnswerSection(
                heading="Results",
                claims=[
                    claim("C1", ids=["E2"]),
                    claim("C2", ids=["E1", "E4"]),
                ],
            ),
        )
    )
    text, sources = GroundedAnswerRenderer().render(
        "question", verified, None, registry
    )
    assert "Supported fact [1]" in text
    assert "Supported fact [1][2]" in text
    assert [item.url for item in sources] == [same_url, "https://b.example"]
    assert sources[0].chunk_indexes == (1, 0)
    assert "unused" not in text


def test_renderer_adds_coverage_limitation_without_search_or_llm() -> None:
    coverage = CoverageResult(
        adequate=False,
        covered_aspects=["方法"],
        missing_aspects=["部署成本"],
        reason="缺证据",
    )
    text, _ = GroundedAnswerRenderer().render(
        "比较方法和部署成本",
        VerifiedAnswerDraft((AnswerSection(claims=[claim()]),)),
        coverage,
        EvidenceRegistry([evidence("A")]),
    )
    assert "证据局限" in text
    assert "部署成本" in text


def test_renderer_never_receives_or_renders_dropped_claim() -> None:
    text, _ = GroundedAnswerRenderer().render(
        "question",
        VerifiedAnswerDraft((AnswerSection(claims=[claim(text="kept")]),)),
        None,
        EvidenceRegistry([evidence("A")]),
    )
    assert "kept" in text
    assert "unsupported" not in text


class StaticPlanner:
    def plan(self, question, registry):
        return draft(claim())


class StaticCoverage:
    def __init__(self, value: CoverageResult | Exception) -> None:
        self.value = value

    def check(self, question, claims):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def test_full_grounded_generator_builds_trace_and_only_used_sources() -> None:
    coverage = CoverageResult(
        adequate=True,
        covered_aspects=["answer"],
        missing_aspects=[],
        reason="covered",
    )
    generator = GroundedAnswerGenerator(
        StaticPlanner(),
        GroundingService(FakeVerifier([[result()]]), FakeRewriter([])),
        StaticCoverage(coverage),
        GroundedAnswerRenderer(),
    )
    answer = generator.generate("question", [evidence("A"), evidence("unused")])
    assert answer.grounding_verified
    assert answer.grounding_trace.draft_claim_count == 1
    assert answer.grounding_trace.verified_claim_count == 1
    assert answer.grounding_trace.citation_coverage == 1.0
    assert len(answer.sources) == 1


def test_coverage_failure_still_renders_verified_claims() -> None:
    generator = GroundedAnswerGenerator(
        StaticPlanner(),
        GroundingService(FakeVerifier([[result()]]), FakeRewriter([])),
        StaticCoverage(RuntimeError("coverage down")),
        GroundedAnswerRenderer(),
    )
    answer = generator.generate("question", [evidence("A")])
    assert "Supported fact" in answer.text
    assert answer.grounding_trace.coverage is None


def test_claim_trace_records_rewritten_and_final_text() -> None:
    trace = ClaimTrace("C1", "strong", "weak", ("E1",), True, True, "supported")
    assert trace.original_text == "strong"
    assert trace.final_text == "weak"
