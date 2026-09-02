"""Search provider abstraction for future extension."""

from abc import ABC, abstractmethod

from src.models.search_result import SearchResult


class SearchError(RuntimeError):
    """Raised when online search cannot complete."""


class SearchProvider(ABC):
    """Minimal interface implemented by the sole V0.1 provider."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Return normalized search results."""

