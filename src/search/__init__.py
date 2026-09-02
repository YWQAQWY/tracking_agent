"""Online search providers."""

from src.search.base import SearchError, SearchProvider
from src.search.ddgs_provider import DDGSSearchProvider

__all__ = ["DDGSSearchProvider", "SearchError", "SearchProvider"]

