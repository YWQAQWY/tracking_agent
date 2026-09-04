import pytest
from pydantic import ValidationError

from src.config import Settings


def test_settings_accept_local_ollama() -> None:
    configured = Settings(ollama_host="http://127.0.0.1:11434/")
    assert configured.ollama_host == "http://127.0.0.1:11434"


def test_settings_reject_remote_llm_host() -> None:
    with pytest.raises(ValidationError, match="本机"):
        Settings(ollama_host="https://cloud.example.com")


def test_settings_validate_search_limits() -> None:
    with pytest.raises(ValidationError):
        Settings(results_per_query_per_provider=20)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_search_queries", 0),
        ("max_combined_search_results", 0),
        ("max_pages_to_read", 0),
        ("max_search_concurrency", 0),
        ("search_timeout", 0),
        ("http_timeout", 0),
        ("max_page_bytes", 99_999),
        ("min_content_length", 0),
        ("embedding_batch_size", 0),
        ("rerank_batch_size", 0),
        ("chunk_size", 99),
        ("min_chunk_length", 0),
        ("embedding_top_k", 0),
        ("rerank_top_k", 0),
        ("max_chunks_per_document", 0),
        ("max_research_rounds", 0),
        ("max_followup_queries_per_round", 0),
        ("min_new_evidence_to_continue", 0),
        ("max_critic_evidence", 0),
        ("max_chars_per_critic_evidence", 99),
        ("max_total_critic_context_chars", 499),
        ("final_evidence_top_k", 0),
        ("max_chars_per_evidence", 99),
        ("max_total_context_chars", 499),
        ("max_claim_rewrite_attempts", 2),
        ("max_claims", 0),
        ("max_evidence_per_claim", 0),
        ("verification_batch_size", 0),
    ],
)
def test_settings_validate_runtime_limits(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_settings_normalizes_domain_lists() -> None:
    settings = Settings(
        allowed_domains=" Example.com, arxiv.org,example.com ",
        blocked_domains=[" Pinterest.com "],
    )
    assert settings.allowed_domains == ("example.com", "arxiv.org")
    assert settings.blocked_domains == ("pinterest.com",)


def test_settings_rejects_domain_urls() -> None:
    with pytest.raises(ValidationError, match="格式无效"):
        Settings(allowed_domains=["https://example.com/path"])


def test_settings_validates_retrieval_device() -> None:
    with pytest.raises(ValidationError, match="RETRIEVAL_DEVICE"):
        Settings(retrieval_device="mps")


def test_grounded_generation_can_be_disabled_for_v05_legacy_mode() -> None:
    settings = Settings(enable_grounded_generation=False)
    assert settings.enable_grounded_generation is False


@pytest.mark.parametrize(
    "values",
    [
        {"chunk_size": 200, "chunk_overlap": 200},
        {"chunk_size": 200, "min_chunk_length": 201},
    ],
)
def test_settings_validates_chunk_relationships(values: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        Settings(**values)
