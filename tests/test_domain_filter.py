import pytest

from src.models.search_result import SearchResult
from src.search.domain_filter import DomainFilter


def result(url: str) -> SearchResult:
    return SearchResult(
        title="title",
        url=url,
        snippet="snippet",
        provider="test",
        query="query",
    )


def test_allowlist_keeps_only_matching_domain_and_subdomains() -> None:
    candidates = [
        result("https://arxiv.org/abs/123"),
        result("https://export.arxiv.org/abs/456"),
        result("https://github.com/project"),
    ]

    kept = DomainFilter(allowed_domains=["arxiv.org"]).filter(candidates)

    assert [str(item.url) for item in kept] == [
        "https://arxiv.org/abs/123",
        "https://export.arxiv.org/abs/456",
    ]


def test_subdomain_matches_parent_domain() -> None:
    policy = DomainFilter(allowed_domains=["example.com"])
    assert policy.allows("https://www.example.com/page")
    assert policy.allows("https://docs.example.com/page")
    assert not policy.allows("https://notexample.com/page")


def test_blocklist_removes_domain_and_subdomains() -> None:
    policy = DomainFilter(blocked_domains=["pinterest.com"])
    assert not policy.allows("https://pinterest.com/pin/1")
    assert not policy.allows("https://www.pinterest.com/pin/2")
    assert policy.allows("https://example.com/article")


def test_blocklist_wins_after_allowlist() -> None:
    policy = DomainFilter(
        allowed_domains=["example.com"],
        blocked_domains=["docs.example.com"],
    )
    assert policy.allows("https://example.com/page")
    assert not policy.allows("https://docs.example.com/page")
    assert not policy.allows("https://outside.org/page")


def test_domain_filter_rejects_url_instead_of_domain_rule() -> None:
    with pytest.raises(ValueError, match="格式无效"):
        DomainFilter(allowed_domains=["https://example.com/path"])
