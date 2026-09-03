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
        ("max_chars_per_document", 99),
        ("max_total_context_chars", 499),
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
