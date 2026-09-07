"""Evaluation inputs and persisted envelopes; production trace types are reused."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.grounding.models import AnswerClaim
from src.models.chunk import DocumentChunk
from src.models.evidence import Evidence


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: str = "uncategorized"
    difficulty: Literal["easy", "medium", "hard"] | None = None
    expected_aspects: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EndToEndEvalCase(EvaluationCase):
    task: Literal["end_to_end"] = "end_to_end"
    relevant_urls: list[str] | None = None
    relevant_chunk_ids: list[str] | None = None

    @field_validator("relevant_urls")
    @classmethod
    def validate_urls(cls, values):
        from src.search.url_normalizer import URLNormalizer
        for value in values or []:
            URLNormalizer().normalize(value)
        return values


class PlannerEvalCase(EvaluationCase):
    task: Literal["planner"] = "planner"


class CriticEvalCase(EvaluationCase):
    task: Literal["critic"] = "critic"
    evidence: list[Evidence]
    expected_sufficient: bool = Field(strict=True)
    expected_missing_aspects: list[str] = Field(default_factory=list)
    executed_queries: list[str] = Field(default_factory=list)


class VerifierEvalCase(EvaluationCase):
    task: Literal["verifier"] = "verifier"
    claim: AnswerClaim
    evidence: list[Evidence]
    label: Literal["supported", "unsupported"]


class RetrievalEvalCase(EvaluationCase):
    task: Literal["retrieval"] = "retrieval"
    candidates: list[DocumentChunk]
    relevant_urls: list[str] | None = None
    relevant_chunk_ids: list[str] | None = None

    @model_validator(mode="after")
    def validate_corpus(self):
        EndToEndEvalCase.validate_urls(self.relevant_urls)
        ids = [item.id for item in self.candidates]
        keys = [(item.url, item.chunk_index) for item in self.candidates]
        if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
            raise ValueError("candidate chunk IDs and URL/chunk_index pairs must be unique")
        if self.relevant_chunk_ids and not set(self.relevant_chunk_ids) <= set(ids):
            raise ValueError("relevant_chunk_ids must belong to the fixed corpus")
        return self


class EvalRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str
    dataset_path: str
    output_dir: str = "eval_runs"
    limit: int | None = Field(default=None, ge=1)
    retrieval_k: int = Field(default=10, ge=1)
    judge: bool = False
    model_variant: str | None = None
    adapter_name: str | None = None
    prompt_version: str | None = None

    @field_validator("run_name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value):
            raise ValueError("run_name must be a simple directory name")
        return value


class EvalCaseResult(BaseModel):
    case_id: str
    question: str
    category: str
    difficulty: str | None = None
    task: str
    status: str
    answer: str | None = None
    termination_reason: str | None = None
    error: dict[str, Any] | None = None
    latency_seconds: float
    raw: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, Any] = Field(default_factory=dict)
    llm_judged: dict[str, Any] | None = None
