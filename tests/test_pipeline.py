import asyncio

from src.crawling.models import FetchResult
from src.models.document import Document
from src.models.search_plan import SearchPlan
from src.models.search_result import SearchResult
from src.pipeline import TrackerPipeline
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


def test_pipeline_runs_multi_query_multi_source_filter_read_answer() -> None:
    events: list[str] = []
    crawled_urls: list[str] = []
    active_fetches = 0
    peak_fetches = 0

    class Planner:
        def plan(self, question: str) -> SearchPlan:
            events.append("plan")
            return SearchPlan(queries=["rag architecture", "retrieval grounding"])

    class Sources:
        async def search(self, queries: list[str]) -> SearchBatch:
            events.append("search")
            assert queries == ["rag architecture", "retrieval grounding"]
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
                coverage=(
                    SearchCoverage(1, queries[0], "provider-a", 1),
                    SearchCoverage(1, queries[0], "provider-b", 1),
                    SearchCoverage(2, queries[1], "provider-a", 1),
                    SearchCoverage(2, queries[1], "provider-b", 1),
                ),
                providers=("provider-a", "provider-b"),
                raw_result_count=4,
                duplicate_count=0,
            )

    class Crawler:
        async def fetch(self, url: str) -> FetchResult | None:
            nonlocal active_fetches, peak_fetches
            crawled_urls.append(url)
            events.append(f"fetch:{url}")
            active_fetches += 1
            peak_fetches = max(peak_fetches, active_fetches)
            await asyncio.sleep(0)
            active_fetches -= 1
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

    class Context:
        def build(self, documents: list[Document]) -> str:
            events.append("context")
            return "\n".join(document.text for document in documents)

    class Answer:
        def generate(self, question: str, context: str) -> str:
            events.append("answer")
            assert "READ CONTENT" in context
            assert "DO NOT SEND SNIPPET" not in context
            return "grounded answer"

    pipeline = TrackerPipeline(
        planner=Planner(),
        source_manager=Sources(),
        domain_filter=DomainFilter(blocked_domains=["blocked.com"]),
        crawler=Crawler(),
        extractor=Extractor(),
        context_builder=Context(),
        answer_generator=Answer(),
        max_pages=3,
    )

    result = asyncio.run(pipeline.run("What is RAG?"))

    assert result.answer == "grounded answer"
    assert result.plan.queries == ["rag architecture", "retrieval grounding"]
    assert [document.title for document in result.documents] == ["A", "C"]
    assert events[:2] == ["plan", "search"]
    assert events[-2:] == ["context", "answer"]
    assert "https://blocked.com/" not in crawled_urls
    assert len(crawled_urls) == 3
    assert peak_fetches == 3
