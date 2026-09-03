import asyncio

import pytest

from src.crawling.models import FetchResult
from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.models.evidence import Evidence, ScoredChunk
from src.models.search_plan import SearchPlan
from src.models.search_result import SearchResult
from src.pipeline import PipelineError, TrackerPipeline
from src.search.domain_filter import DomainFilter
from src.search.source_manager import SearchBatch, SearchCoverage


def search_result(title: str, url: str, provider: str, query: str) -> SearchResult:
    return SearchResult(
        title=title,
        url=url,
        snippet="DO NOT SEND SNIPPET",
        provider=provider,
        query=query,
    )


def make_pipeline(events: list[str], crawled_urls: list[str]) -> TrackerPipeline:
    class Planner:
        def plan(self, question: str) -> SearchPlan:
            events.append("plan")
            return SearchPlan(queries=["rag architecture", "retrieval grounding"])

    class Sources:
        async def search(self, queries: list[str]) -> SearchBatch:
            events.append("search")
            results = (
                search_result("A", "https://a.com", "provider-a", queries[0]),
                search_result(
                    "Blocked", "https://blocked.com", "provider-b", queries[0]
                ),
                search_result("B", "https://b.com", "provider-b", queries[1]),
                search_result("C", "https://c.com", "provider-a", queries[1]),
            )
            return SearchBatch(
                results=results,
                coverage=(SearchCoverage(1, queries[0], "provider-a", 1),),
                providers=("provider-a", "provider-b"),
                raw_result_count=4,
                duplicate_count=0,
            )

    class URLDeduplicator:
        def deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
            events.append("url_dedup")
            return results

    class Crawler:
        async def fetch(self, url: str) -> FetchResult | None:
            crawled_urls.append(url)
            events.append(f"fetch:{url}")
            await asyncio.sleep(0)
            if url == "https://b.com/":
                return None
            return FetchResult(
                url=url,
                html=f"<article>full document from {url}</article>",
                status_code=200,
                content_type="text/html",
            )

    class Extractor:
        def extract(self, html: str, url: str, title_hint: str) -> Document:
            events.append(f"extract:{url}")
            return Document(url=url, title=title_hint, text=f"READ CONTENT {url}")

    class ContentDeduplicator:
        def deduplicate(self, documents: list[Document]) -> list[Document]:
            events.append("content_dedup")
            return documents

    class Chunker:
        def chunk(self, documents: list[Document]) -> list[DocumentChunk]:
            events.append("chunk")
            return [
                DocumentChunk(
                    id=f"chunk-{index}",
                    text=document.text,
                    url=str(document.url),
                    title=document.title,
                    chunk_index=0,
                )
                for index, document in enumerate(documents)
            ]

    class FakeEmbedder:
        model_name = "fake-embedder"
        device = "cpu"

    class Retriever:
        embedder = FakeEmbedder()

        def retrieve(
            self, question: str, chunks: list[DocumentChunk]
        ) -> list[ScoredChunk]:
            events.append("embedding")
            assert question == "What is RAG?"
            return [
                ScoredChunk(chunk=chunk, embedding_score=1.0 - index * 0.1)
                for index, chunk in enumerate(chunks)
            ]

    class Reranker:
        model_name = "fake-reranker"

        def rerank(
            self, question: str, candidates: list[ScoredChunk]
        ) -> list[Evidence]:
            events.append("rerank")
            return [Evidence.from_scored_chunk(item, 2.0) for item in candidates]

    class Context:
        def build(self, evidence: list[Evidence]) -> str:
            events.append("context")
            return "\n".join(item.text for item in evidence)

    class Answer:
        def generate(self, question: str, context: str) -> str:
            events.append("answer")
            assert "READ CONTENT" in context
            assert "DO NOT SEND SNIPPET" not in context
            return "grounded answer"

    return TrackerPipeline(
        planner=Planner(),
        source_manager=Sources(),
        domain_filter=DomainFilter(blocked_domains=["blocked.com"]),
        url_deduplicator=URLDeduplicator(),
        crawler=Crawler(),
        extractor=Extractor(),
        content_deduplicator=ContentDeduplicator(),
        chunker=Chunker(),
        retriever=Retriever(),
        reranker=Reranker(),
        context_builder=Context(),
        answer_generator=Answer(),
        max_pages=3,
    )


def test_pipeline_runs_complete_v04_retrieval_funnel() -> None:
    events: list[str] = []
    crawled_urls: list[str] = []
    pipeline = make_pipeline(events, crawled_urls)

    result = asyncio.run(pipeline.run("What is RAG?"))

    assert result.answer == "grounded answer"
    assert result.plan.queries == ["rag architecture", "retrieval grounding"]
    assert [document.title for document in result.documents] == ["A", "C"]
    assert [item.title for item in result.evidence] == ["A", "C"]
    assert events[:3] == ["plan", "search", "url_dedup"]
    assert events[-6:] == [
        "content_dedup",
        "chunk",
        "embedding",
        "rerank",
        "context",
        "answer",
    ]
    assert "https://blocked.com/" not in crawled_urls
    assert len(crawled_urls) == 3
    assert result.retrieval_trace.raw_search_results == 4
    assert result.retrieval_trace.unique_urls == 3
    assert result.retrieval_trace.documents == 2
    assert result.retrieval_trace.chunks == 2
    assert result.retrieval_trace.final_evidence == 2
    assert result.retrieval_trace.embedding_model == "fake-embedder"
    assert set(result.retrieval_trace.timings) == {
        "planning",
        "search",
        "crawl_extract",
        "chunk",
        "embedding",
        "rerank",
        "context",
        "llm",
        "total",
    }


def test_pipeline_reports_empty_documents_clearly() -> None:
    events: list[str] = []
    pipeline = make_pipeline(events, [])

    async def failed_fetch(url: str) -> None:
        return None

    pipeline.crawler.fetch = failed_fetch
    with pytest.raises(PipelineError, match="Document/Evidence"):
        asyncio.run(pipeline.run("What is RAG?"))


def test_pipeline_reports_empty_chunks_clearly() -> None:
    events: list[str] = []
    pipeline = make_pipeline(events, [])
    pipeline.chunker.chunk = lambda documents: []

    with pytest.raises(PipelineError, match="Chunk"):
        asyncio.run(pipeline.run("What is RAG?"))
