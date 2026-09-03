"""Inspectable counts and timings for the retrieval funnel."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """Describe how broad search candidates became final Evidence."""

    raw_search_results: int
    combined_search_results: int
    domain_filtered_results: int
    unique_urls: int
    documents: int
    unique_documents: int
    chunks: int
    embedding_candidates: int
    final_evidence: int
    evidence_source_count: int
    embedding_model: str
    reranker_model: str
    device: str
    timings: dict[str, float] = field(default_factory=dict)
