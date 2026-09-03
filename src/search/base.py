"""Provider-independent asynchronous search contract."""

from abc import ABC, abstractmethod

from src.models.search_result import SearchResult


class SearchError(RuntimeError):
    """Raised when online search cannot complete."""


class SearchProvider(ABC):
    """Common interface that lets SourceManager orchestrate any search source."""

    name: str

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Return normalized results without exposing provider-specific fields."""
