"""Run component or end-to-end evaluation: python -m tracker.eval."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import logging
import platform
from pathlib import Path

from src.eval.models import EvalRunConfig
from src.eval.runner import EvaluationRunner


def system_metadata(settings) -> dict:
    # Settings currently contains only model/configuration fields. Explicitly
    # allowlist them so future API keys cannot silently enter an eval artifact.
    allowed = """ollama_model ollama_keep_alive llm_timeout max_search_queries
        results_per_query_per_provider max_combined_search_results max_pages_to_read
        max_search_concurrency search_timeout allowed_domains blocked_domains http_timeout
        max_page_bytes min_content_length embedding_model_name reranker_model_name
        retrieval_device embedding_batch_size rerank_batch_size chunk_size chunk_overlap
        min_chunk_length embedding_top_k rerank_top_k max_chunks_per_document enable_reranker
        max_research_rounds max_followup_queries_per_round min_new_evidence_to_continue
        max_critic_evidence max_chars_per_critic_evidence max_total_critic_context_chars
        final_evidence_top_k max_chars_per_evidence max_total_context_chars
        enable_grounded_generation max_claim_rewrite_attempts max_claims max_evidence_per_claim
        verification_batch_size enable_coverage_check max_run_seconds
        max_runtime_search_requests max_runtime_crawl_requests max_runtime_llm_calls
        network_retry_max_attempts llm_retry_max_attempts retry_base_delay_seconds
        retry_max_delay_seconds retry_jitter_seconds checkpoint_enabled""".split()
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted([root / "main.py", *root.glob("src/**/*.py"), *root.glob("tracker/*.py")]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    packages = {}
    for name in ("ollama", "httpx", "ddgs", "sentence-transformers", "torch", "pydantic"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"settings": settings.model_dump(mode="json", include=set(allowed)),
            "providers": ["ddgs", "wikipedia"], "python": platform.python_version(),
            "packages": packages, "source_fingerprint": digest.hexdigest()}


def local_model_metadata(settings) -> dict:
    """Best-effort model digest; a failed metadata request cannot abort cases."""
    import httpx

    try:
        with httpx.Client(trust_env=False, timeout=3) as client:
            response = client.get(settings.ollama_host + "/api/tags")
            response.raise_for_status()
        models = response.json().get("models", [])
        for model in models:
            if model.get("name", "").removesuffix(":latest") == settings.ollama_model.removesuffix(":latest"):
                return {key: model.get(key) for key in ("name", "digest", "size", "details")}
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        pass
    return {"name": settings.ollama_model, "digest": None}


async def run(config: EvalRunConfig) -> Path:
    from main import execute_runtime
    from src.config import Settings
    from src.eval.components import ComponentEvaluationRunner
    from src.eval.dataset import EvaluationDatasetLoader
    from src.llm.client import LLMClient

    cases = EvaluationDatasetLoader().load(config.dataset_path)
    tasks = {case.task for case in (cases[:config.limit] if config.limit else cases)}
    settings = Settings.from_env()
    llm = LLMClient(settings.ollama_host, settings.ollama_model,
                    settings.ollama_keep_alive, timeout=settings.llm_timeout)
    components = ComponentEvaluationRunner()
    if "planner" in tasks:
        from src.plan.search_planner import SearchPlanner
        components.planner = SearchPlanner(llm, settings.max_search_queries)
    if "critic" in tasks:
        from src.agent.critic import EvidenceCritic
        from src.agent.critic_context import CriticContextBuilder
        components.critic = EvidenceCritic(llm, CriticContextBuilder(
            settings.max_critic_evidence, settings.max_chars_per_critic_evidence,
            settings.max_total_critic_context_chars))
    if "verifier" in tasks:
        from src.grounding.verifier import CitationVerifier
        components.verifier = CitationVerifier(llm, settings.verification_batch_size,
                                               settings.max_chars_per_evidence,
                                               settings.max_total_context_chars)
    if "retrieval" in tasks:
        from src.retrieval.embedder import BGEEmbedder
        from src.retrieval.reranker import BGEReranker, EmbeddingOnlyReranker
        from src.retrieval.retriever import SemanticRetriever
        components.retriever = SemanticRetriever(BGEEmbedder(
            settings.embedding_model_name, settings.embedding_batch_size, settings.retrieval_device),
            top_k=settings.embedding_top_k)
        args = dict(top_k=settings.rerank_top_k, max_chunks_per_document=settings.max_chunks_per_document)
        components.reranker = BGEReranker(
            model_name=settings.reranker_model_name, batch_size=settings.rerank_batch_size,
            device=settings.retrieval_device, **args
        ) if settings.enable_reranker else EmbeddingOnlyReranker(**args)
    judge = None
    if config.judge:
        from src.eval.judges.llm_judge import LLMJudge
        judge = LLMJudge(llm)

    async def execute(question):
        return await execute_runtime(question, settings, llm)

    metadata = system_metadata(settings)
    metadata["local_model"] = local_model_metadata(settings)
    return await EvaluationRunner(execute, components, judge).run(config, metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", dest="dataset_path", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--output-dir", default="eval_runs")
    parser.add_argument("--model-variant")
    parser.add_argument("--adapter-name")
    parser.add_argument("--prompt-version")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        destination = asyncio.run(run(EvalRunConfig(**vars(args))))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Evaluation cancelled; completed cases and partial report were saved.")
        return 130
    print(f"Report: {destination / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
