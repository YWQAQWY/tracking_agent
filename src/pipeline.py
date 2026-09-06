"""V0.4 broad discovery followed by precision evidence retrieval."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.answer.answer_generator import AnswerGenerator
from src.context.context_builder import ContextBuilder
from src.crawling.crawler import WebCrawler
from src.extraction.content_extractor import ContentExtractor
from src.models.document import Document
from src.models.evidence import Evidence, ScoredChunk
from src.models.search_result import SearchResult
from src.plan.models import SearchPlan
from src.plan.search_planner import SearchPlanner
from src.retrieval.chunker import DocumentChunker
from src.retrieval.deduplicator import ContentDeduplicator
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import SemanticRetriever
from src.retrieval.trace import RetrievalTrace
from src.search.domain_filter import DomainFilter
from src.search.source_manager import SearchBatch, SourceManager
from src.search.url_normalizer import URLDeduplicator
from src.tools.research import ResearchTool


logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when the complete pipeline cannot produce a grounded answer."""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Answer, sources, and traces needed to explain one complete run."""

    plan: SearchPlan
    search_batch: SearchBatch
    search_results: tuple[SearchResult, ...]
    documents: tuple[Document, ...]
    embedding_candidates: tuple[ScoredChunk, ...]
    evidence: tuple[Evidence, ...]
    retrieval_trace: RetrievalTrace
    answer: str


class TrackerPipeline:
    """Orchestrate Planning → Search → Read → Retrieve → Generate."""

    def __init__(
        self,
        planner: SearchPlanner,
        source_manager: SourceManager,
        domain_filter: DomainFilter,
        url_deduplicator: URLDeduplicator,
        crawler: WebCrawler,
        extractor: ContentExtractor,
        content_deduplicator: ContentDeduplicator,
        chunker: DocumentChunker,
        retriever: SemanticRetriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        answer_generator: AnswerGenerator,
        max_pages: int = 10,
    ) -> None:
        self.planner = planner
        self.source_manager = source_manager
        self.domain_filter = domain_filter
        self.url_deduplicator = url_deduplicator
        self.crawler = crawler
        self.extractor = extractor
        self.content_deduplicator = content_deduplicator
        self.chunker = chunker
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.answer_generator = answer_generator
        self.max_pages = max_pages
        self.research_tool = ResearchTool(
            source_manager=source_manager,
            domain_filter=domain_filter,
            url_deduplicator=url_deduplicator,
            crawler=crawler,
            extractor=extractor,
            content_deduplicator=content_deduplicator,
            chunker=chunker,
            retriever=retriever,
            reranker=reranker,
            max_pages=max_pages,
        )

    @property
    def research_round(self) -> ResearchTool:
        """Backward-compatible alias for the canonical research tool."""
        return self.research_tool

    async def run(self, question: str) -> PipelineResult:
        """Run the complete V0.4 retrieval funnel for one question."""
        total_started = time.perf_counter()
        timings: dict[str, float] = {}
        clean_question = question.strip()
        if not clean_question:
            raise PipelineError("用户问题不能为空。")

        started = time.perf_counter()
        plan = self.planner.plan(clean_question)
        timings["planning"] = time.perf_counter() - started
        logger.info("Generated %d search queries", len(plan.queries))
        for index, query in enumerate(plan.queries, start=1):
            logger.info("Query[%d]: %s", index, query)

        round_result = await self.research_tool.run(clean_question, plan.queries)
        failure_messages = {
            "search": "搜索没有返回结果，请换一种问法或检查网络。",
            "domain_filter": "搜索结果均被域名策略过滤，无法继续读取网页。",
            "url_dedup": "URL 标准化后没有可读取的候选网页。",
            "documents": "搜索成功，但没有找到可用 Document/Evidence。请稍后重试或更换问题。",
            "chunks": "Document 存在，但没有生成满足最小长度的 Chunk。",
            "embedding": "Embedding retrieval 没有返回候选 Chunk。",
            "rerank": "Reranker 没有选出可用 Evidence。",
        }
        if round_result.failure_stage:
            raise PipelineError(failure_messages[round_result.failure_stage])
        timings.update(
            (key, value)
            for key, value in round_result.retrieval_trace.timings.items()
            if key != "total"
        )
        search_batch = round_result.search_batch
        unique_results = list(round_result.search_results)
        unique_documents = list(round_result.documents)
        candidates = list(round_result.embedding_candidates)
        evidence = list(round_result.evidence)

        logger.info(
            "Evidence sources: %d unique URLs", len({item.url for item in evidence})
        )
        started = time.perf_counter()
        logger.info("Building final evidence context")
        context = self.context_builder.build(evidence)
        timings["context"] = time.perf_counter() - started
        if not context:
            raise PipelineError("Evidence 无法构造成有效的模型上下文。")

        started = time.perf_counter()
        logger.info("Calling local Qwen3 with selected Evidence")
        answer = self.answer_generator.generate(clean_question, context)
        timings["llm"] = time.perf_counter() - started
        timings["total"] = time.perf_counter() - total_started
        logger.info(
            "Performance search=%.3fs crawl=%.3fs chunk=%.3fs "
            "embedding=%.3fs rerank=%.3fs llm=%.3fs total=%.3fs",
            timings["search"],
            timings["crawl_extract"],
            timings["chunk"],
            timings["embedding"],
            timings["rerank"],
            timings["llm"],
            timings["total"],
        )

        trace = RetrievalTrace(
            raw_search_results=round_result.retrieval_trace.raw_search_results,
            combined_search_results=round_result.retrieval_trace.combined_search_results,
            domain_filtered_results=round_result.retrieval_trace.domain_filtered_results,
            unique_urls=round_result.retrieval_trace.unique_urls,
            documents=round_result.retrieval_trace.documents,
            unique_documents=round_result.retrieval_trace.unique_documents,
            chunks=round_result.retrieval_trace.chunks,
            embedding_candidates=round_result.retrieval_trace.embedding_candidates,
            final_evidence=round_result.retrieval_trace.final_evidence,
            evidence_source_count=round_result.retrieval_trace.evidence_source_count,
            embedding_model=round_result.retrieval_trace.embedding_model,
            reranker_model=round_result.retrieval_trace.reranker_model,
            device=round_result.retrieval_trace.device,
            timings=timings,
        )
        return PipelineResult(
            plan=plan,
            search_batch=search_batch,
            search_results=tuple(unique_results),
            documents=tuple(unique_documents),
            embedding_candidates=tuple(candidates),
            evidence=tuple(evidence),
            retrieval_trace=trace,
            answer=answer,
        )
