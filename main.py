"""Tracker V0.1 interactive CLI."""

from __future__ import annotations

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from src.answer.answer_generator import AnswerGenerationError, AnswerGenerator
from src.config import Settings
from src.llm.client import LLMClient, LLMError
from src.planner.search_planner import PlannerError, SearchPlanner
from src.search.base import SearchError
from src.search.ddgs_provider import DDGSSearchProvider


console = Console()


def run() -> int:
    console.print(Panel.fit("[bold cyan]Tracker V0.1[/bold cyan]\nLocal LLM + Online Search"))

    try:
        settings = Settings.from_env()
        llm = LLMClient(settings.ollama_host, settings.ollama_model)
        with console.status("检查本地 Ollama 和模型…"):
            llm.check_health()
    except (ValidationError, ValueError, LLMError) as exc:
        console.print(f"[bold red]错误：[/bold red]{exc}")
        return 1

    question = Prompt.ask("\n[bold]请输入问题[/bold]").strip()
    if not question:
        console.print("[yellow]问题不能为空，未发起模型或搜索请求。[/yellow]")
        return 1

    planner = SearchPlanner(llm, settings.search_max_results)
    provider = DDGSSearchProvider()
    answer_generator = AnswerGenerator(llm)

    try:
        with console.status("本地模型正在规划搜索…"):
            plan = planner.plan(question)
        console.print(f"\n[bold]Search query:[/bold] {plan.query}")

        with console.status("正在通过 DDGS 联网搜索…"):
            results = provider.search(plan.query, plan.max_results)
        console.print(f"[bold]Found:[/bold] {len(results)} results")
        if not results:
            console.print("[yellow]搜索没有返回结果，请换一种问法或检查网络。[/yellow]")
            return 1

        with console.status("本地模型正在基于搜索摘要回答…"):
            answer = answer_generator.generate(question, results)
    except (LLMError, PlannerError, SearchError, AnswerGenerationError) as exc:
        console.print(f"\n[bold red]错误：[/bold red]{exc}")
        return 1

    console.print(Panel(answer, title="Answer", border_style="green"))
    console.print("\n[bold]Sources:[/bold]")
    for index, result in enumerate(results, start=1):
        console.print(f"{index}. [link={result.url}]{result.title}[/link]")
        console.print(f"   {result.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

