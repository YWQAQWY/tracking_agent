"""Small network helpers shared by outbound HTTP clients."""

import os
from urllib.parse import urlparse


def supported_http_proxy_from_environment() -> str | None:
    """Return an HTTP(S) proxy and ignore unsupported SOCKS-only settings."""
    for key in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        value = os.getenv(key)
        if value and urlparse(value).scheme in {"http", "https"}:
            return value
    return None
