"""One reusable V0.4 Search → Read → Retrieve execution round."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from src.crawling.crawler import WebCrawler
from src.extraction.content_extractor import ContentExtractor
from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.models.evidence import Evidence, ScoredChunk
from src.models.search_result import SearchResult
from src.retrieval.chunker import DocumentChunker
from src.retrieval.deduplicator import ContentDeduplicator
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import SemanticRetriever
from src.retrieval.trace import RetrievalTrace
from src.search.domain_filter import DomainFilter
from src.search.source_manager import SearchBatch, SourceManager
from src.search.url_normalizer import URLDeduplicator


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResearchRoundResult:
    """All observable products of one full retrieval pass."""

    queries: tuple[str, ...]
    search_batch: SearchBatch
    search_results: tuple[SearchResult, ...]
    documents: tuple[Document, ...]
    embedding_candidates: tuple[ScoredChunk, ...]
    evidence: tuple[Evidence, ...]
    retrieval_trace: RetrievalTrace
    failure_stage: str | None = None


class ResearchRound:
    """Reuse the existing V0.4 components without knowing agent loop state."""

    def __init__(
        self,
        source_manager: SourceManager,
        domain_filter: DomainFilter,
        url_deduplicator: URLDeduplicator,
        crawler: WebCrawler,
        extractor: ContentExtractor,
        content_deduplicator: ContentDeduplicator,
        chunker: DocumentChunker,
        retriever: SemanticRetriever,
        reranker: Reranker,
        max_pages: int = 10,
    ) -> None:
        self.source_manager = source_manager
        self.domain_filter = domain_filter
        self.url_deduplicator = url_deduplicator
        self.crawler = crawler
        self.extractor = extractor
        self.content_deduplicator = content_deduplicator
        self.chunker = chunker
        self.retriever = retriever
        self.reranker = reranker
        self.max_pages = max_pages

    async def run(self, question: str, queries: list[str]) -> ResearchRoundResult:
        """Run complete retrieval and return an empty result at normal dead ends."""
        total_started = time.perf_counter()
        timings: dict[str, float] = {}

        started = time.perf_counter()
        search_batch = await self.source_manager.search(queries)
        timings["search"] = time.perf_counter() - started
        domain_results = self.domain_filter.filter(list(search_batch.results))
        unique_results = self.url_deduplicator.deduplicate(domain_results)
        if not search_batch.results:
            return self._result(
                queries, search_batch, domain_results, unique_results, timings,
                total_started, "search",
            )
        if not domain_results:
            return self._result(
                queries, search_batch, domain_results, unique_results, timings,
                total_started, "domain_filter",
            )
        if not unique_results:
            return self._result(
                queries, search_batch, domain_results, unique_results, timings,
                total_started, "url_dedup",
            )

        selected_results = unique_results[: self.max_pages]
        logger.info("Selected %d unique pages for crawling", len(selected_results))
        started = time.perf_counter()
        loaded = await asyncio.gather(
            *(self._load_document(result) for result in selected_results)
        )
        documents = [document for document in loaded if document is not None]
        timings["crawl_extract"] = time.perf_counter() - started
        logger.info("Crawled %d pages successfully", len(documents))
        if not documents:
            return self._result(
                queries, search_batch, domain_results, unique_results, timings,
                total_started, "documents",
            )

        unique_documents = self.content_deduplicator.deduplicate(documents)
        started = time.perf_counter()
        chunks = self.chunker.chunk(unique_documents)
        timings["chunk"] = time.perf_counter() - started
        if not chunks:
            return self._result(
                queries, search_batch, domain_results, unique_results, timings,
                total_started, "chunks", documents, unique_documents,
            )

        started = time.perf_counter()
        candidates = self.retriever.retrieve(question, chunks)
        timings["embedding"] = time.perf_counter() - started
        if not candidates:
            return self._result(
                queries, search_batch, domain_results, unique_results, timings,
                total_started, "embedding", documents, unique_documents, chunks,
            )
        self._offload(self.retriever)

        started = time.perf_counter()
        evidence = self.reranker.rerank(question, candidates)
        timings["rerank"] = time.perf_counter() - started
        self._offload(self.reranker)
        if not evidence:
            return self._result(
                queries, search_batch, domain_results, unique_results, timings,
                total_started, "rerank", documents, unique_documents, chunks,
                candidates,
            )
        return self._result(
            queries, search_batch, domain_results, unique_results, timings,
            total_started, None, documents, unique_documents, chunks, candidates,
            evidence,
        )

    async def _load_document(self, result: SearchResult) -> Document | None:
        url = str(result.url)
        try:
            fetched = await self.crawler.fetch(url)
            if fetched is None:
                return None
            return self.extractor.extract(
                fetched.html, url=str(fetched.url), title_hint=result.title
            )
        except Exception as exc:
            logger.warning(
                "Failed to load document %s: %s", url, exc.__class__.__name__
            )
            return None

    def _result(
        self,
        queries: list[str],
        search_batch: SearchBatch,
        domain_results: list[SearchResult],
        unique_results: list[SearchResult],
        timings: dict[str, float],
        total_started: float,
        failure_stage: str | None,
        documents: list[Document] | None = None,
        unique_documents: list[Document] | None = None,
        chunks: list[DocumentChunk] | None = None,
        candidates: list[ScoredChunk] | None = None,
        evidence: list[Evidence] | None = None,
    ) -> ResearchRoundResult:
        documents = documents or []
        unique_documents = unique_documents or []
        chunks = chunks or []
        candidates = candidates or []
        evidence = evidence or []
        timings["total"] = time.perf_counter() - total_started
        trace = RetrievalTrace(
            raw_search_results=search_batch.raw_result_count,
            combined_search_results=len(search_batch.results),
            domain_filtered_results=len(domain_results),
            unique_urls=len(unique_results),
            documents=len(documents),
            unique_documents=len(unique_documents),
            chunks=len(chunks),
            embedding_candidates=len(candidates),
            final_evidence=len(evidence),
            evidence_source_count=len({item.url for item in evidence}),
            embedding_model=self.retriever.embedder.model_name,
            reranker_model=self.reranker.model_name,
            device=self.retriever.embedder.device,
            timings=dict(timings),
        )
        return ResearchRoundResult(
            queries=tuple(queries),
            search_batch=search_batch,
            search_results=tuple(unique_results),
            documents=tuple(unique_documents),
            embedding_candidates=tuple(candidates),
            evidence=tuple(evidence),
            retrieval_trace=trace,
            failure_stage=failure_stage,
        )

    @staticmethod
    def _offload(component: object) -> None:
        offload = getattr(component, "offload", None)
        if callable(offload):
            offload()
