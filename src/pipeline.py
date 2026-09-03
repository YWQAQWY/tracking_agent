"""V0.3 Multi-Query -> Multi-Source -> Read -> Answer pipeline."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from src.answer.answer_generator import AnswerGenerator
from src.context.context_builder import ContextBuilder
from src.crawling.crawler import WebCrawler
from src.extraction.content_extractor import ContentExtractor
from src.models.document import Document
from src.models.search_plan import SearchPlan
from src.models.search_result import SearchResult
from src.planner.search_planner import SearchPlanner
from src.search.domain_filter import DomainFilter
from src.search.source_manager import SearchBatch, SourceManager


logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when the complete pipeline cannot produce a grounded answer."""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Values the CLI needs to explain what the pipeline actually did."""

    plan: SearchPlan
    search_batch: SearchBatch
    search_results: tuple[SearchResult, ...]
    documents: tuple[Document, ...]
    answer: str


class TrackerPipeline:
    """Orchestrate one bounded search-and-read pass without an agent framework."""

    def __init__(
        self,
        planner: SearchPlanner,
        source_manager: SourceManager,
        domain_filter: DomainFilter,
        crawler: WebCrawler,
        extractor: ContentExtractor,
        context_builder: ContextBuilder,
        answer_generator: AnswerGenerator,
        max_pages: int = 3,
    ) -> None:
        self.planner = planner
        self.source_manager = source_manager
        self.domain_filter = domain_filter
        self.crawler = crawler
        self.extractor = extractor
        self.context_builder = context_builder
        self.answer_generator = answer_generator
        self.max_pages = max_pages

    async def run(self, question: str) -> PipelineResult:
        """Run Planner -> Search -> Crawl -> Extract -> Context -> Answer."""
        clean_question = question.strip()
        if not clean_question:
            raise PipelineError("用户问题不能为空。")

        # Question != Search Query: one natural-language question can benefit
        # from several complementary keyword formulations and research angles.
        plan = self.planner.plan(clean_question)
        logger.info("Generated %d search queries", len(plan.queries))
        for index, query in enumerate(plan.queries, start=1):
            logger.info("Query[%d]: %s", index, query)

        search_batch = await self.source_manager.search(plan.queries)
        if not search_batch.results:
            raise PipelineError("搜索没有返回结果，请换一种问法或检查网络。")

        # Domain policy must run before crawling so disallowed sources never
        # consume network bandwidth or enter the model's evidence context.
        search_results = self.domain_filter.filter(list(search_batch.results))
        if not search_results:
            raise PipelineError("搜索结果均被域名策略过滤，无法继续读取网页。")

        # Search != Read: SearchResult only says where an answer may exist.
        # Fetching and extracting that URL creates the Document we truly read.
        selected_results = search_results[: self.max_pages]
        logger.info("Selected %d pages for crawling", len(selected_results))
        loaded = await asyncio.gather(
            *(self._load_document(result) for result in selected_results)
        )
        documents = [document for document in loaded if document is not None]
        if not documents:
            raise PipelineError(
                "搜索成功，但候选网页均无法读取到足够正文。请稍后重试或更换问题。"
            )

        logger.info("Building context from %d documents", len(documents))
        context = self.context_builder.build(documents)
        if not context:
            raise PipelineError("网页正文无法构造成有效的模型上下文。")

        logger.info("Calling local LLM with extracted web documents")
        answer = self.answer_generator.generate(clean_question, context)
        return PipelineResult(
            plan=plan,
            search_batch=search_batch,
            search_results=tuple(search_results),
            documents=tuple(documents),
            answer=answer,
        )

    async def _load_document(self, result: SearchResult) -> Document | None:
        """Fetch and extract one candidate while isolating per-page failures."""
        url = str(result.url)
        try:
            fetched = await self.crawler.fetch(url)
            if fetched is None:
                return None
            return self.extractor.extract(
                fetched.html,
                url=str(fetched.url),
                title_hint=result.title,
            )
        except Exception as exc:
            # External pages are untrusted and inconsistent. One broken page
            # should not discard useful Documents loaded from the other URLs.
            logger.warning(
                "Failed to load document %s: %s", url, exc.__class__.__name__
            )
            return None
