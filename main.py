"""Tracker V0.3 multi-query and multi-source CLI."""

from __future__ import annotations

import argparse
import asyncio
import logging

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from src.answer.answer_generator import AnswerGenerationError, AnswerGenerator
from src.config import Settings
from src.context.context_builder import ContextBuilder
from src.crawling.crawler import WebCrawler
from src.extraction.content_extractor import ContentExtractor
from src.llm.client import LLMClient, LLMError
from src.pipeline import PipelineError, PipelineResult, TrackerPipeline
from src.planner.search_planner import PlannerError, SearchPlanner
from src.search.base import SearchError
from src.search.ddgs_provider import DDGSSearchProvider
from src.search.domain_filter import DomainFilter
from src.search.source_manager import SourceManager
from src.search.wikipedia_provider import WikipediaSearchProvider


console = Console()


async def execute_pipeline(
    question: str,
    settings: Settings,
    llm: LLMClient,
    allowed_domains: tuple[str, ...] | None = None,
    blocked_domains: tuple[str, ...] | None = None,
) -> PipelineResult:
    """Build short-lived HTTP resources and execute one pipeline run."""
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
        pipeline = TrackerPipeline(
            planner=SearchPlanner(llm, settings.max_search_queries),
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
            crawler=crawler,
            extractor=ContentExtractor(settings.min_content_length),
            context_builder=ContextBuilder(
                settings.max_chars_per_document,
                settings.max_total_context_chars,
            ),
            answer_generator=AnswerGenerator(llm),
            max_pages=settings.max_pages_to_read,
        )
        return await pipeline.run(question)


def run(
    question: str | None = None,
    allowed_domains: tuple[str, ...] | None = None,
    blocked_domains: tuple[str, ...] | None = None,
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    console.print(
        Panel.fit(
            "[bold cyan]Tracker V0.3[/bold cyan]\n"
            "Multi-Query + Multi-Source + Local LLM"
        )
    )

    try:
        settings = Settings.from_env()
        llm = LLMClient(settings.ollama_host, settings.ollama_model)
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
            )
        )
    except (
        LLMError,
        PlannerError,
        SearchError,
        AnswerGenerationError,
        PipelineError,
        ValueError,
    ) as exc:
        console.print(f"\n[bold red]错误：[/bold red]{exc}")
        return 1

    console.print(f"\n[bold]Question:[/bold] {user_question}")
    console.print("\n[bold]Search Queries:[/bold]")
    for index, query in enumerate(result.plan.queries, start=1):
        console.print(f"{index}. {query}")

    console.print("\n[bold]Search Providers:[/bold]")
    for provider in result.search_batch.providers:
        console.print(f"- {provider}")

    console.print("\n[bold]Search Coverage:[/bold]")
    for item in result.search_batch.coverage:
        if item.succeeded:
            outcome = f"[green]OK[/green], {item.result_count} results"
        else:
            outcome = f"[yellow]FAILED[/yellow], {item.error}"
        console.print(f"Query {item.query_index} → {item.provider}: {outcome}")
    console.print(
        "[bold]Combined:[/bold] "
        f"{result.search_batch.raw_result_count} raw → "
        f"{len(result.search_batch.results)} unique/capped → "
        f"{len(result.search_results)} after domain filter"
    )
    console.print("\n[bold]Reading:[/bold]")
    for index, document in enumerate(result.documents, start=1):
        console.print(f"[{index}] {document.url}")

    console.print(Panel(result.answer, title="Answer", border_style="green"))
    console.print("\n[bold]Sources:[/bold]")
    for index, document in enumerate(result.documents, start=1):
        title = document.title or "Untitled"
        console.print(f"{index}. [link={document.url}]{title}[/link]")
        console.print(f"   {document.url}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the shared parser used by both supported CLI entry points."""
    parser = argparse.ArgumentParser(description="Tracker V0.3 local search agent")
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
    return parser


def cli(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and run one Tracker invocation."""
    args = build_parser().parse_args(argv)
    return run(
        " ".join(args.question) or None,
        tuple(args.allowed_domains) if args.allowed_domains else None,
        tuple(args.blocked_domains) if args.blocked_domains else None,
    )


if __name__ == "__main__":
    raise SystemExit(cli())
