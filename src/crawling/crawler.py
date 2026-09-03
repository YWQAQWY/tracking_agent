"""Async HTML downloader with small, explicit safety limits."""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

import httpx

from src.crawling.models import FetchResult


logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 Tracker/0.2"
)


class WebCrawler:
    """Fetch HTML pages without allowing one bad URL to stop the pipeline."""

    def __init__(
        self,
        timeout: float = 10.0,
        max_page_bytes: int = 2_000_000,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_page_bytes = max_page_bytes
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            proxy=self._supported_proxy_from_environment(),
            # We pass a validated proxy explicitly.  This avoids HTTPX crashing
            # on desktop values such as ALL_PROXY=socks://127.0.0.1:7890.
            trust_env=False,
        )

    async def __aenter__(self) -> "WebCrawler":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch(self, url: str) -> FetchResult | None:
        """Download one bounded HTML response, returning None on page failure."""
        if not self._is_http_url(url):
            logger.warning("Skipping unsupported URL: %s", url)
            return None

        logger.info("Fetching URL: %s", url)
        try:
            async with self._client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type")
                if not self._is_html(content_type):
                    logger.warning(
                        "Skipping non-HTML response from %s (%s)",
                        url,
                        content_type or "unknown content type",
                    )
                    return None

                declared_size = self._content_length(response)
                if declared_size and declared_size > self.max_page_bytes:
                    logger.warning("Skipping oversized page: %s", url)
                    return None

                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self.max_page_bytes:
                        logger.warning("Page exceeded size limit: %s", url)
                        return None

                encoding = response.encoding or "utf-8"
                html = bytes(body).decode(encoding, errors="replace")
                return FetchResult(
                    url=str(response.url),
                    html=html,
                    status_code=response.status_code,
                    content_type=content_type,
                )
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Failed to fetch %s: HTTP %s", url, exc.response.status_code
            )
        except httpx.TimeoutException:
            logger.warning("Failed to fetch %s: timeout after %.1fs", url, self.timeout)
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch %s: %s", url, exc.__class__.__name__)
        return None

    @staticmethod
    def _is_http_url(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname)

    @staticmethod
    def _is_html(content_type: str | None) -> bool:
        if content_type is None:
            return True
        media_type = content_type.split(";", 1)[0].strip().lower()
        return media_type in {"text/html", "application/xhtml+xml"}

    @staticmethod
    def _content_length(response: httpx.Response) -> int | None:
        raw_length = response.headers.get("content-length")
        if not raw_length:
            return None
        try:
            return int(raw_length)
        except ValueError:
            return None

    @staticmethod
    def _supported_proxy_from_environment() -> str | None:
        """Prefer an HTTP proxy and ignore unsupported ``socks://`` values."""
        for key in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
            value = os.getenv(key)
            if value and urlparse(value).scheme in {"http", "https"}:
                return value
        return None

