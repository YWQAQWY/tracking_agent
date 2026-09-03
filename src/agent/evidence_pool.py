"""Ordered, deduplicated evidence accumulated across research rounds."""

from src.models.evidence import Evidence


class EvidencePool:
    """Keep task-local evidence while preserving first-seen ranking order."""

    def __init__(self, items: list[Evidence] | None = None) -> None:
        self._items: list[Evidence] = []
        self._keys: set[tuple[str, int]] = set()
        if items:
            self.extend(items)

    def add(self, evidence: Evidence) -> bool:
        """Add one unique passage and report whether the pool grew."""
        key = (evidence.url, evidence.chunk_index)
        if key in self._keys:
            return False
        self._keys.add(key)
        self._items.append(evidence)
        return True

    def extend(self, items: list[Evidence]) -> int:
        """Add many passages and return the number of newly accepted items."""
        return sum(self.add(item) for item in items)

    def all(self) -> list[Evidence]:
        """Return a copy so callers cannot mutate pool state accidentally."""
        return list(self._items)

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def unique_source_count(self) -> int:
        return len({item.url for item in self._items})

    @property
    def source_urls(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.url for item in self._items))

    def __len__(self) -> int:
        return self.size
