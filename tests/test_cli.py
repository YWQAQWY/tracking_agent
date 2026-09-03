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
            "question",
        ]
    )

    assert args.question == ["question"]
    assert args.allowed_domains == ["arxiv.org", "github.com"]
    assert args.blocked_domains == ["example.com"]
