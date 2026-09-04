"""Structured claim models and inspectable grounding results."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnswerClaim(BaseModel):
    """One atomic factual statement and its proposed evidence mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("id", "text")
    @classmethod
    def require_text(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("claim id/text 不能为空")
        return clean

    @field_validator("evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)


class AnswerSection(BaseModel):
    """A display section containing independently verifiable claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    heading: str | None = None
    claims: list[AnswerClaim] = Field(default_factory=list)

    @field_validator("heading")
    @classmethod
    def normalize_heading(cls, value: str | None) -> str | None:
        clean = value.strip() if value else ""
        return clean or None


class GroundedAnswerDraft(BaseModel):
    """Structured intermediate representation before deterministic rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sections: list[AnswerSection]

    @model_validator(mode="before")
    @classmethod
    def accept_single_section(cls, value: object) -> object:
        # Small local models commonly omit the outer list wrapper while still
        # returning the exact section/claim schema. Normalize that shape once.
        if isinstance(value, dict) and "sections" not in value and "claims" in value:
            return {"sections": [value]}
        return value

    @model_validator(mode="after")
    def require_unique_claim_ids(self) -> "GroundedAnswerDraft":
        ids = [claim.id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("claim id 必须唯一")
        return self

    @property
    def claims(self) -> list[AnswerClaim]:
        return [claim for section in self.sections for claim in section.claims]


@dataclass(frozen=True, slots=True)
class VerifiedAnswerDraft:
    """Section structure after verification; may be empty when all claims drop."""

    sections: tuple[AnswerSection, ...]

    @property
    def claims(self) -> list[AnswerClaim]:
        return [claim for section in self.sections for claim in section.claims]


class ClaimVerificationResult(BaseModel):
    """Whether cited passages entail one specific claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    supported: bool
    support_score: float | None = Field(default=None, ge=0.0, le=1.0)
    supported_evidence_ids: list[str] = Field(default_factory=list)
    reason: str

    @field_validator("claim_id", "reason")
    @classmethod
    def require_text(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("verification claim_id/reason 不能为空")
        return clean

    @field_validator("supported_evidence_ids")
    @classmethod
    def normalize_ids(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)


class VerificationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    results: list[ClaimVerificationResult]


class ClaimRewrite(BaseModel):
    """A weaker supported statement, or null when no useful rewrite exists."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    rewritten_text: str | None
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("claim_id")
    @classmethod
    def require_id(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("rewrite claim_id 不能为空")
        return clean

    @field_validator("rewritten_text")
    @classmethod
    def normalize_rewrite(cls, value: str | None) -> str | None:
        clean = value.strip() if value else ""
        return clean or None

    @field_validator("evidence_ids")
    @classmethod
    def normalize_ids(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)


class RewriteBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rewrites: list[ClaimRewrite]


class CoverageResult(BaseModel):
    """Completeness of verified claims relative to the original question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adequate: bool
    covered_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    reason: str

    @field_validator("covered_aspects", "missing_aspects")
    @classmethod
    def normalize_aspects(cls, values: list[str]) -> list[str]:
        return _unique_nonempty(values)

    @field_validator("reason")
    @classmethod
    def require_reason(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("coverage reason 不能为空")
        return clean


@dataclass(frozen=True, slots=True)
class CitationSource:
    citation_number: int
    url: str
    title: str | None
    evidence_ids: tuple[str, ...]
    chunk_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ClaimTrace:
    claim_id: str
    original_text: str
    final_text: str | None
    evidence_ids: tuple[str, ...]
    supported: bool
    rewritten: bool
    reason: str


@dataclass(frozen=True, slots=True)
class GroundingTrace:
    draft_claim_count: int
    initially_supported_claim_count: int
    unsupported_claim_count: int
    rewritten_claim_count: int
    rewrite_passed_claim_count: int
    dropped_claim_count: int
    verified_claim_count: int
    cited_source_count: int
    claim_traces: tuple[ClaimTrace, ...] = ()
    coverage: CoverageResult | None = None
    fallback_reason: str | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def citation_coverage(self) -> float:
        # Every claim reaching the renderer has at least one verifier-approved ID.
        return 1.0 if self.verified_claim_count else 0.0

    @property
    def verification_pass_rate(self) -> float:
        if not self.draft_claim_count:
            return 0.0
        return self.verified_claim_count / self.draft_claim_count

    @property
    def drop_rate(self) -> float:
        if not self.draft_claim_count:
            return 0.0
        return self.dropped_claim_count / self.draft_claim_count


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    text: str
    sources: tuple[CitationSource, ...]
    grounding_trace: GroundingTrace
    grounding_verified: bool = True


def _unique_nonempty(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = value.strip()
        if clean and clean not in seen:
            seen.add(clean)
            normalized.append(clean)
    return normalized
