"""Small network helpers shared by outbound HTTP clients."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlparse


_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def supported_http_proxy_from_environment() -> str | None:
    """Return an HTTP(S) proxy and ignore unsupported SOCKS-only settings."""
    for key in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        value = os.getenv(key)
        if value and urlparse(value).scheme in {"http", "https"}:
            return value
    return None


@contextmanager
def hide_unsupported_proxy_environment() -> Iterator[None]:
    """Temporarily hide proxy values that HTTPX cannot safely parse.

    Search clients still receive valid HTTP(S) proxies. This context is used
    only while local Hugging Face models initialize, after search tasks finish.
    """
    hidden: dict[str, str] = {}
    for key in _PROXY_ENV_KEYS:
        value = os.getenv(key)
        if value and urlparse(value).scheme not in {"http", "https"}:
            hidden[key] = os.environ.pop(key)
    try:
        yield
    finally:
        os.environ.update(hidden)
