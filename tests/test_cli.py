from main import build_parser


def test_cli_parser_accepts_repeatable_domain_flags() -> None:
    args = build_parser().parse_args(
        [
            "--allow-domain",
            "arxiv.org",
            "--allow-domain",
            "github.com",
            "--ban-domain",
            "example.com",
            "--debug-retrieval",
            "--debug-agent",
            "--debug-grounding",
            "--debug-runtime",
            "--run-timeout",
            "30",
            "--max-research-rounds",
            "2",
            "--max-search-requests",
            "8",
            "--max-crawl-requests",
            "6",
            "--max-llm-calls",
            "12",
            "--disable-reranker",
            "--embedding-top-k",
            "12",
            "--rerank-top-k",
            "4",
            "question",
        ]
    )

    assert args.question == ["question"]
    assert args.allowed_domains == ["arxiv.org", "github.com"]
    assert args.blocked_domains == ["example.com"]
    assert args.debug_retrieval is True
    assert args.debug_agent is True
    assert args.debug_grounding is True
    assert args.debug_runtime is True
    assert args.run_timeout == 30
    assert args.max_research_rounds == 2
    assert args.max_search_requests == 8
    assert args.max_crawl_requests == 6
    assert args.max_llm_calls == 12
    assert args.disable_reranker is True
    assert args.embedding_top_k == 12
    assert args.rerank_top_k == 4


def test_cli_parser_accepts_resume_without_question() -> None:
    args = build_parser().parse_args(["--resume", "run_abc"])
    assert args.resume_run_id == "run_abc"
    assert args.question == []
