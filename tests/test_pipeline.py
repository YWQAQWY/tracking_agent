import asyncio

from src.crawling.models import FetchResult
from src.models.document import Document
from src.models.search_plan import SearchPlan
from src.models.search_result import SearchResult
from src.pipeline import TrackerPipeline


def test_pipeline_runs_search_read_context_answer_and_tolerates_one_failure() -> None:
    events: list[str] = []
    active_fetches = 0
    peak_fetches = 0

    class Planner:
        def plan(self, question: str) -> SearchPlan:
            events.append("plan")
            return SearchPlan(query="rag", max_results=3)

    class Search:
        def search(self, query: str, max_results: int) -> list[SearchResult]:
            events.append("search")
            return [
                SearchResult(
                    title="A", url="https://a.com", snippet="DO NOT SEND SNIPPET"
                ),
                SearchResult(
                    title="B", url="https://b.com", snippet="DO NOT SEND SNIPPET"
                ),
                SearchResult(
                    title="C", url="https://c.com", snippet="DO NOT SEND SNIPPET"
                ),
            ]

    class Crawler:
        async def fetch(self, url: str) -> FetchResult | None:
            nonlocal active_fetches, peak_fetches
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
        search_provider=Search(),
        crawler=Crawler(),
        extractor=Extractor(),
        context_builder=Context(),
        answer_generator=Answer(),
        max_pages=3,
    )

    result = asyncio.run(pipeline.run("What is RAG?"))

    assert result.answer == "grounded answer"
    assert [document.title for document in result.documents] == ["A", "C"]
    assert events[:2] == ["plan", "search"]
    assert events[-2:] == ["context", "answer"]
    assert sum(event.startswith("fetch:") for event in events) == 3
    assert peak_fetches == 3
