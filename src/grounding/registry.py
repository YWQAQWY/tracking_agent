"""Stable per-answer Evidence IDs and safe lookups."""

from src.context.context_builder import format_evidence_blocks
from src.models.evidence import Evidence


class EvidenceRegistry:
    """Map E1/E2/... to final Evidence without pretending IDs are persistent."""

    def __init__(self, evidence: list[Evidence]) -> None:
        self._items = {
            f"E{index}": item for index, item in enumerate(evidence, start=1)
        }

    def get(self, evidence_id: str) -> Evidence | None:
        return self._items.get(evidence_id)

    def contains(self, evidence_id: str) -> bool:
        return evidence_id in self._items

    def ids(self) -> tuple[str, ...]:
        return tuple(self._items)

    def format(
        self,
        evidence_ids: list[str] | tuple[str, ...] | None = None,
        max_chars_per_evidence: int = 1_200,
        max_total_chars: int = 15_000,
    ) -> str:
        selected = evidence_ids or self.ids()
        labeled: list[tuple[str, Evidence]] = []
        for evidence_id in selected:
            item = self.get(evidence_id)
            if item is not None:
                labeled.append((evidence_id, item))
        return format_evidence_blocks(
            labeled, max_chars_per_evidence, max_total_chars
        )
