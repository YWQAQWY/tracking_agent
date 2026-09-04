import os

from src.network import hide_unsupported_proxy_environment


def test_model_loading_hides_unsupported_proxy_and_restores_it(monkeypatch) -> None:
    invalid = r"socks\://127.0.0.1:7890"
    monkeypatch.setenv("ALL_PROXY", invalid)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")

    with hide_unsupported_proxy_environment():
        assert "ALL_PROXY" not in os.environ
        assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"

    assert os.environ["ALL_PROXY"] == invalid


def test_model_loading_hides_socks_proxy_even_without_backslash(monkeypatch) -> None:
    monkeypatch.setenv("all_proxy", "socks://127.0.0.1:7890")

    with hide_unsupported_proxy_environment():
        assert "all_proxy" not in os.environ

    assert os.environ["all_proxy"] == "socks://127.0.0.1:7890"
