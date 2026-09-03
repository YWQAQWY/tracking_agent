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
    assert args.disable_reranker is True
    assert args.embedding_top_k == 12
    assert args.rerank_top_k == 4
