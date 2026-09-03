"""Wikipedia MediaWiki API search provider."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup
from pydantic import ValidationError

from src.models.search_result import SearchResult
from src.network import supported_http_proxy_from_environment
from src.search.base import SearchError, SearchProvider


class WikipediaSearchProvider(SearchProvider):
    """Use Wikipedia's independent search index as a second search source."""

    name = "wikipedia"
    endpoint = "https://en.wikipedia.org/w/api.php"

    def __init__(
        self,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout = timeout
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": "Tracker/0.3 (local research agent)"},
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            proxy=supported_http_proxy_from_environment(),
            trust_env=False,
        )

    async def __aenter__(self) -> "WikipediaSearchProvider":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        clean_query = query.strip()
        if not clean_query:
            raise SearchError("搜索关键词不能为空。")
        if not 1 <= limit <= 10:
            raise SearchError("limit 必须在 1 到 10 之间。")

        params = {
            "action": "query",
            "list": "search",
            "srsearch": clean_query,
            "srlimit": str(limit),
            "format": "json",
            "formatversion": "2",
            "utf8": "1",
        }
        try:
            response = await self._client.get(self.endpoint, params=params)
            response.raise_for_status()
            payload = response.json()
            return self._normalize(payload, clean_query, limit)
        except SearchError:
            raise
        except Exception as exc:
            raise SearchError(
                "Wikipedia 联网搜索失败。请检查网络、DNS 或代理配置后重试。"
            ) from exc

    @classmethod
    def _normalize(
        cls, payload: Any, query: str, limit: int
    ) -> list[SearchResult]:
        if not isinstance(payload, Mapping):
            raise SearchError("Wikipedia 返回了无法识别的数据格式。")
        query_payload = payload.get("query")
        if not isinstance(query_payload, Mapping):
            return []
        rows = query_payload.get("search")
        if not isinstance(rows, list):
            return []

        normalized: list[SearchResult] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            title = str(row.get("title") or "").strip()
            snippet = BeautifulSoup(
                str(row.get("snippet") or ""), "lxml"
            ).get_text(" ", strip=True)
            try:
                normalized.append(
                    SearchResult(
                        title=title,
                        url=cls._article_url(title),
                        snippet=snippet or title,
                        provider=cls.name,
                        query=query,
                    )
                )
            except (ValidationError, TypeError, ValueError):
                continue
            if len(normalized) >= limit:
                break
        return normalized

    @staticmethod
    def _article_url(title: str) -> str:
        slug = quote(title.replace(" ", "_"), safe="()")
        return f"https://en.wikipedia.org/wiki/{slug}"
