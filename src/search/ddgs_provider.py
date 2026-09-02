"""DDGS-backed online search provider."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ddgs import DDGS
from pydantic import ValidationError

from src.models.search_result import SearchResult
from src.search.base import SearchError, SearchProvider

RawSearch = Callable[..., Iterable[Mapping[str, Any]]]


class DDGSSearchProvider(SearchProvider):
    """Search the web and isolate DDGS-specific field names."""

    def __init__(self, search_func: RawSearch | None = None) -> None:
        self._search_func = search_func

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        clean_query = query.strip()
        if not clean_query:
            raise SearchError("搜索关键词不能为空。")
        if not 1 <= max_results <= 10:
            raise SearchError("max_results 必须在 1 到 10 之间。")

        try:
            raw_results = (
                self._search_func(clean_query, max_results=max_results)
                if self._search_func
                else DDGS().text(clean_query, max_results=max_results)
            )
            return self._normalize(raw_results, max_results)
        except SearchError:
            raise
        except Exception as exc:
            raise SearchError(
                "DDGS 联网搜索失败。请检查网络、DNS 或代理配置后重试。"
            ) from exc

    @staticmethod
    def _normalize(
        raw_results: Iterable[Mapping[str, Any]], max_results: int
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
                    )
                )
            except (ValidationError, TypeError, ValueError):
                continue
            if len(normalized) >= max_results:
                break
        return normalized

