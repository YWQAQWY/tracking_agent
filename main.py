"""Tracker V0.2 interactive and one-shot CLI."""

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


console = Console()


async def execute_pipeline(
    question: str, settings: Settings, llm: LLMClient
) -> PipelineResult:
    """Build short-lived HTTP resources and execute one pipeline run."""
    async with WebCrawler(
        timeout=settings.http_timeout,
        max_page_bytes=settings.max_page_bytes,
    ) as crawler:
        pipeline = TrackerPipeline(
            planner=SearchPlanner(llm, settings.search_max_results),
            search_provider=DDGSSearchProvider(),
            crawler=crawler,
            extractor=ContentExtractor(settings.min_content_length),
            context_builder=ContextBuilder(
                settings.max_chars_per_document,
                settings.max_total_context_chars,
            ),
            answer_generator=AnswerGenerator(llm),
            max_pages=settings.max_pages,
        )
        return await pipeline.run(question)


def run(question: str | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    console.print(
        Panel.fit("[bold cyan]Tracker V0.2[/bold cyan]\nSearch + Read + Local LLM")
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
        result = asyncio.run(execute_pipeline(user_question, settings, llm))
    except (
        LLMError,
        PlannerError,
        SearchError,
        AnswerGenerationError,
        PipelineError,
    ) as exc:
        console.print(f"\n[bold red]错误：[/bold red]{exc}")
        return 1

    console.print(f"\n[bold]Search query:[/bold] {result.plan.query}")
    console.print(f"[bold]Search results:[/bold] {len(result.search_results)}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracker V0.2 local search agent")
    parser.add_argument("question", nargs="*", help="question to search and answer")
    args = parser.parse_args()
    raise SystemExit(run(" ".join(args.question) or None))
