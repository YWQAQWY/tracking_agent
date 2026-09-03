"""Domain allowlist and blocklist policy for search candidates."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from urllib.parse import urlparse

from src.models.search_result import SearchResult


logger = logging.getLogger(__name__)


class DomainFilter:
    """Apply source policy before the crawler spends requests on a URL."""

    def __init__(
        self,
        allowed_domains: Iterable[str] = (),
        blocked_domains: Iterable[str] = (),
    ) -> None:
        self.allowed_domains = self._normalize_domains(allowed_domains)
        self.blocked_domains = self._normalize_domains(blocked_domains)

    def filter(self, results: list[SearchResult]) -> list[SearchResult]:
        """Keep allowlisted hosts first, then apply the blocklist."""
        kept = [result for result in results if self.allows(str(result.url))]
        logger.info("Domain filter removed %d results", len(results) - len(kept))
        logger.info("%d search results remain after domain filtering", len(kept))
        return kept

    def allows(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower().rstrip(".")
        if not host:
            return False

        # Filtering before crawling is both a trust policy and an efficiency
        # boundary: rejected sites should consume no crawler request at all.
        if self.allowed_domains and not any(
            self._matches(host, domain) for domain in self.allowed_domains
        ):
            return False
        if any(self._matches(host, domain) for domain in self.blocked_domains):
            return False
        return True

    @staticmethod
    def _matches(host: str, domain: str) -> bool:
        return host == domain or host.endswith(f".{domain}")

    @staticmethod
    def _normalize_domains(domains: Iterable[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in domains:
            domain = value.strip().lower().strip(".")
            if not domain:
                continue
            if "://" in domain or "/" in domain or any(
                character.isspace() for character in domain
            ):
                raise ValueError(f"域名规则格式无效：{value}")
            if domain not in seen:
                normalized.append(domain)
                seen.add(domain)
        return tuple(normalized)
