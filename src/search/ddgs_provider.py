"""DDGS-backed online search provider."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ddgs import DDGS
from pydantic import ValidationError

from src.models.search_result import SearchResult
from src.search.base import SearchError, SearchProvider

RawSearch = Callable[..., Iterable[Mapping[str, Any]]]


class DDGSSearchProvider(SearchProvider):
    """Search the web and isolate DDGS-specific field names."""

    name = "ddgs"

    def __init__(
        self,
        search_func: RawSearch | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._search_func = search_func
        self.timeout = timeout

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        clean_query = query.strip()
        if not clean_query:
            raise SearchError("搜索关键词不能为空。")
        if not 1 <= limit <= 10:
            raise SearchError("limit 必须在 1 到 10 之间。")

        try:
            # DDGS exposes a blocking API. Moving it to a worker thread keeps
            # the asyncio event loop free to run other provider/query requests.
            raw_results = await asyncio.to_thread(
                self._search_sync,
                clean_query,
                limit,
            )
            return self._normalize(raw_results, clean_query, limit)
        except SearchError:
            raise
        except Exception as exc:
            raise SearchError(
                "DDGS 联网搜索失败。请检查网络、DNS 或代理配置后重试。"
            ) from exc

    def _search_sync(
        self, query: str, limit: int
    ) -> Iterable[Mapping[str, Any]]:
        if self._search_func:
            return self._search_func(query, max_results=limit)
        return DDGS(timeout=max(1, int(self.timeout))).text(
            query, max_results=limit
        )

    @staticmethod
    def _normalize(
        raw_results: Iterable[Mapping[str, Any]], query: str, limit: int
    ) -> list[SearchResult]:
        normalized: list[SearchResult] = []
        for raw in raw_results:
            if not isinstance(raw, Mapping):
                continue
            try:
                normalized.append(
                    SearchResult(
                        title=str(raw.get("title", "")),
                        url=str(raw.get("href") or raw.get("url") or ""),
                        snippet=str(raw.get("body") or raw.get("snippet") or ""),
                        provider=DDGSSearchProvider.name,
                        query=query,
                    )
                )
            except (ValidationError, TypeError, ValueError):
                continue
            if len(normalized) >= limit:
                break
        return normalized
