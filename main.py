"""Tracker V0.5 evidence-driven adaptive-search CLI."""

from __future__ import annotations

import argparse
import asyncio
import logging

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from src.agent.critic import CriticError, EvidenceCritic
from src.agent.critic_context import CriticContextBuilder
from src.agent.models import ResearchResult
from src.agent.research_agent import ResearchAgent, ResearchAgentError
from src.agent.research_round import ResearchRound
from src.answer.answer_generator import AnswerGenerationError, AnswerGenerator
from src.config import Settings
from src.context.context_builder import ContextBuilder
from src.crawling.crawler import WebCrawler
from src.extraction.content_extractor import ContentExtractor
from src.llm.client import LLMClient, LLMError
from src.planner.search_planner import PlannerError, SearchPlanner
from src.retrieval.chunker import DocumentChunker
from src.retrieval.deduplicator import ContentDeduplicator
from src.retrieval.device import DeviceError
from src.retrieval.embedder import BGEEmbedder, EmbeddingError
from src.retrieval.reranker import (
    BGEReranker,
    EmbeddingOnlyReranker,
    RerankerError,
)
from src.retrieval.retriever import RetrievalError, SemanticRetriever
from src.search.base import SearchError
from src.search.ddgs_provider import DDGSSearchProvider
from src.search.domain_filter import DomainFilter
from src.search.source_manager import SourceManager
from src.search.url_normalizer import URLDeduplicator
from src.search.wikipedia_provider import WikipediaSearchProvider


console = Console()


async def execute_pipeline(
    question: str,
    settings: Settings,
    llm: LLMClient,
    allowed_domains: tuple[str, ...] | None = None,
    blocked_domains: tuple[str, ...] | None = None,
    disable_reranker: bool = False,
    embedding_top_k: int | None = None,
    rerank_top_k: int | None = None,
) -> ResearchResult:
    """Build short-lived HTTP resources and execute one adaptive agent run."""
    async with (
        WebCrawler(
            timeout=settings.http_timeout,
            max_page_bytes=settings.max_page_bytes,
        ) as crawler,
        WikipediaSearchProvider(timeout=settings.search_timeout) as wikipedia,
    ):
        providers = [
            DDGSSearchProvider(timeout=settings.search_timeout),
            wikipedia,
        ]
        embedder = BGEEmbedder(
            model_name=settings.embedding_model_name,
            batch_size=settings.embedding_batch_size,
            device=settings.retrieval_device,
        )
        candidate_count = (
            embedding_top_k
            if embedding_top_k is not None
            else settings.embedding_top_k
        )
        evidence_count = (
            rerank_top_k if rerank_top_k is not None else settings.rerank_top_k
        )
        if settings.enable_reranker and not disable_reranker:
            reranker = BGEReranker(
                model_name=settings.reranker_model_name,
                batch_size=settings.rerank_batch_size,
                top_k=evidence_count,
                max_chunks_per_document=settings.max_chunks_per_document,
                device=settings.retrieval_device,
            )
        else:
            reranker = EmbeddingOnlyReranker(
                top_k=evidence_count,
                max_chunks_per_document=settings.max_chunks_per_document,
            )
        research_round = ResearchRound(
            source_manager=SourceManager(
                providers=providers,
                results_per_task=settings.results_per_query_per_provider,
                max_combined_results=settings.max_combined_search_results,
                max_concurrency=settings.max_search_concurrency,
            ),
            domain_filter=DomainFilter(
                allowed_domains=allowed_domains
                if allowed_domains is not None
                else settings.allowed_domains,
                blocked_domains=blocked_domains
                if blocked_domains is not None
                else settings.blocked_domains,
            ),
            url_deduplicator=URLDeduplicator(),
            crawler=crawler,
            extractor=ContentExtractor(settings.min_content_length),
            content_deduplicator=ContentDeduplicator(),
            chunker=DocumentChunker(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                min_chunk_length=settings.min_chunk_length,
            ),
            retriever=SemanticRetriever(embedder, top_k=candidate_count),
            reranker=reranker,
            max_pages=settings.max_pages_to_read,
        )
        agent = ResearchAgent(
            planner=SearchPlanner(llm, settings.max_search_queries),
            research_round=research_round,
            critic=EvidenceCritic(
                llm,
                CriticContextBuilder(
                    max_evidence=settings.max_critic_evidence,
                    max_chars_per_evidence=settings.max_chars_per_critic_evidence,
                    max_total_chars=settings.max_total_critic_context_chars,
                ),
            ),
            final_reranker=reranker,
            context_builder=ContextBuilder(
                settings.max_chars_per_evidence,
                settings.max_total_context_chars,
            ),
            answer_generator=AnswerGenerator(llm),
            max_rounds=settings.max_research_rounds,
            max_followup_queries_per_round=(
                settings.max_followup_queries_per_round
            ),
            min_new_evidence_to_continue=settings.min_new_evidence_to_continue,
            final_evidence_top_k=settings.final_evidence_top_k,
        )
        return await agent.run(question)


def run(
    question: str | None = None,
    allowed_domains: tuple[str, ...] | None = None,
    blocked_domains: tuple[str, ...] | None = None,
    debug_retrieval: bool = False,
    debug_agent: bool = False,
    disable_reranker: bool = False,
    embedding_top_k: int | None = None,
    rerank_top_k: int | None = None,
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    console.print(
        Panel.fit(
            "[bold cyan]Tracker V0.5[/bold cyan]\n"
            "Search + Evidence Critic + Adaptive Search + Local LLM"
        )
    )

    try:
        settings = Settings.from_env()
        llm = LLMClient(
            settings.ollama_host,
            settings.ollama_model,
            settings.ollama_keep_alive,
        )
        with console.status("检查本地 Ollama 和模型…"):
            llm.check_health()
    except (ValidationError, ValueError, LLMError) as exc:
        console.print(f"[bold red]错误：[/bold red]{exc}")
        return 1

    user_question = question or Prompt.ask("\n[bold]请输入问题[/bold]")
    user_question = user_question.strip()
    if not user_question:
        console.print("[yellow]问题不能为空，未发起模型或搜索请求。[/yellow]")
        return 1

    try:
        result = asyncio.run(
            execute_pipeline(
                user_question,
                settings,
                llm,
                allowed_domains,
                blocked_domains,
                disable_reranker,
                embedding_top_k,
                rerank_top_k,
            )
        )
    except (
        LLMError,
        PlannerError,
        CriticError,
        ResearchAgentError,
        SearchError,
        AnswerGenerationError,
        DeviceError,
        EmbeddingError,
        RetrievalError,
        RerankerError,
        ValueError,
    ) as exc:
        console.print(f"\n[bold red]错误：[/bold red]{exc}")
        return 1

    console.print(f"\n[bold]Question:[/bold] {user_question}")
    if debug_retrieval:
        _print_retrieval_debug(result)
    if debug_agent:
        _print_agent_debug(result)

    console.print(Panel(result.answer, title="Answer", border_style="green"))
    console.print("\n[bold]Sources:[/bold]")
    for index, evidence in enumerate(result.evidence, start=1):
        title = evidence.title or "Untitled"
        console.print(f"{index}. [link={evidence.url}]{title}[/link]")
        console.print(f"   {evidence.url} (chunk {evidence.chunk_index})")
    return 0


def _print_retrieval_debug(result: ResearchResult) -> None:
    for round_index, round_result in enumerate(result.rounds, start=1):
        trace = round_result.retrieval_trace
        console.print(
            f"\n[bold]Round {round_index} Retrieval Funnel "
            f"({len(round_result.queries)} queries):[/bold]"
        )
        console.print(
            f"{trace.raw_search_results} raw → "
            f"{trace.combined_search_results} combined/capped → "
            f"{trace.domain_filtered_results} domain-filtered → "
            f"{trace.unique_urls} URLs → {trace.documents} documents → "
            f"{trace.unique_documents} unique documents → {trace.chunks} chunks → "
            f"{trace.embedding_candidates} candidates → "
            f"{trace.final_evidence} round evidence"
        )
        console.print(
            f"[bold]Models:[/bold] {trace.embedding_model} + "
            f"{trace.reranker_model} on {trace.device}"
        )

    console.print("\n[bold]Final Globally Reranked Evidence:[/bold]")
    for index, evidence in enumerate(result.evidence, start=1):
        rerank = (
            f"{evidence.rerank_score:.4f}"
            if evidence.rerank_score is not None
            else "disabled"
        )
        console.print(
            f"{index}. embedding={evidence.embedding_score:.4f} "
            f"rerank={rerank} chunk={evidence.chunk_index} {evidence.url}"
        )

    console.print("\n[bold]Total Performance:[/bold]")
    for stage, seconds in result.research_trace.timings.items():
        console.print(f"- {stage}: {seconds:.3f}s")


def _print_agent_debug(result: ResearchResult) -> None:
    console.print("\n[bold]Agent Trace:[/bold]")
    for trace in result.research_trace.rounds:
        console.print(f"\n[cyan]Round {trace.round_index}[/cyan]")
        console.print("Queries:")
        for query in trace.queries:
            console.print(f"- {query}")
        console.print(
            f"Search results: {trace.search_result_count}; "
            f"new evidence: {trace.new_evidence_count}; "
            f"pool: {trace.total_evidence_count}"
        )
        if trace.critic_sufficient is not None:
            label = "sufficient" if trace.critic_sufficient else "insufficient"
            console.print(f"Critic: {label}")
        if trace.missing_aspects:
            console.print("Missing aspects:")
            for aspect in trace.missing_aspects:
                console.print(f"- {aspect}")
        if trace.follow_up_queries:
            console.print("Follow-up queries:")
            for query in trace.follow_up_queries:
                console.print(f"- {query}")
        if trace.critic_reason:
            console.print(f"Critic reason: {trace.critic_reason}")
    console.print(f"\n[bold]Stop reason:[/bold] {result.research_trace.stop_reason}")
    if result.research_trace.critic_error:
        console.print(
            f"[yellow]Critic fallback:[/yellow] "
            f"{result.research_trace.critic_error}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Create the shared parser used by both supported CLI entry points."""
    parser = argparse.ArgumentParser(description="Tracker V0.5 local search agent")
    parser.add_argument("question", nargs="*", help="question to search and answer")
    parser.add_argument(
        "--allow-domain",
        action="append",
        dest="allowed_domains",
        help="only crawl this domain or its subdomains; repeatable",
    )
    parser.add_argument(
        "--ban-domain",
        action="append",
        dest="blocked_domains",
        help="exclude this domain and its subdomains; repeatable",
    )
    parser.add_argument(
        "--debug-retrieval",
        action="store_true",
        help="show retrieval funnel, scores, models, and timings",
    )
    parser.add_argument(
        "--debug-agent",
        action="store_true",
        help="show rounds, evidence gaps, follow-up queries, and stop reason",
    )
    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="use embedding order directly without loading the reranker",
    )
    parser.add_argument(
        "--embedding-top-k",
        type=int,
        help="override the embedding candidate count",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        help="override each research round's evidence count",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and run one Tracker invocation."""
    args = build_parser().parse_args(argv)
    return run(
        " ".join(args.question) or None,
        tuple(args.allowed_domains) if args.allowed_domains else None,
        tuple(args.blocked_domains) if args.blocked_domains else None,
        args.debug_retrieval,
        args.debug_agent,
        args.disable_reranker,
        args.embedding_top_k,
        args.rerank_top_k,
    )


if __name__ == "__main__":
    raise SystemExit(cli())
