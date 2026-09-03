"""Small, deterministic URL canonicalization for per-run deduplication."""

from __future__ import annotations

import logging
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.models.search_result import SearchResult


logger = logging.getLogger(__name__)

TRACKING_PARAMETERS = {"gclid", "fbclid"}


class URLNormalizer:
    """Normalize common URL variations without domain-specific heuristics."""

    def normalize(self, url: str) -> str:
        parsed = urlsplit(url.strip())
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()
        if scheme not in {"http", "https"} or not hostname:
            raise ValueError(f"无法标准化非 HTTP URL：{url}")

        host = f"[{hostname}]" if ":" in hostname else hostname
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"URL 端口无效：{url}") from exc
        if port is not None and not (
            (scheme == "http" and port == 80)
            or (scheme == "https" and port == 443)
        ):
            host = f"{host}:{port}"

        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/") or "/"

        query_items = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not self._is_tracking_parameter(key)
        ]
        query_items.sort()
        return urlunsplit((scheme, host, path, urlencode(query_items), ""))

    @staticmethod
    def _is_tracking_parameter(key: str) -> bool:
        normalized = key.lower()
        return normalized.startswith("utm_") or normalized in TRACKING_PARAMETERS


class URLDeduplicator:
    """Keep the first SearchResult for each normalized URL."""

    def __init__(self, normalizer: URLNormalizer | None = None) -> None:
        self.normalizer = normalizer or URLNormalizer()

    def deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        # URL duplicates are removed before crawling because downloading the
        # same page twice wastes the most expensive network/parse stage.
        unique: list[SearchResult] = []
        seen: set[str] = set()
        for result in results:
            normalized_url = self.normalizer.normalize(str(result.url))
            if normalized_url in seen:
                continue
            seen.add(normalized_url)
            payload = result.model_dump()
            payload["url"] = normalized_url
            unique.append(SearchResult.model_validate(payload))

        logger.info(
            "URL normalization reduced %d → %d URLs", len(results), len(unique)
        )
        return unique
