from src.agent.evidence_pool import EvidencePool
from src.models.evidence import Evidence


def evidence(url: str, chunk_index: int) -> Evidence:
    return Evidence(
        text=f"evidence {url} {chunk_index}",
        url=url,
        title="Title",
        chunk_index=chunk_index,
        embedding_score=0.8,
        rerank_score=0.9,
    )


def test_evidence_pool_add_and_count() -> None:
    pool = EvidencePool()
    assert pool.add(evidence("https://a.example", 0)) is True
    assert pool.size == 1
    assert len(pool) == 1


def test_evidence_pool_deduplicates_url_and_chunk_index() -> None:
    pool = EvidencePool()
    first = evidence("https://a.example", 0)
    assert pool.extend([first, first]) == 1
    assert pool.add(evidence("https://a.example", 0)) is False
    assert pool.all() == [first]


def test_evidence_pool_counts_unique_sources_in_first_seen_order() -> None:
    pool = EvidencePool(
        [
            evidence("https://a.example", 0),
            evidence("https://a.example", 1),
            evidence("https://b.example", 0),
        ]
    )
    assert pool.unique_source_count == 2
    assert pool.source_urls == ("https://a.example", "https://b.example")
