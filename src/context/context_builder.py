"""Build bounded, source-aware prompt context from selected Evidence."""

from src.models.evidence import Evidence


class ContextBuilder:
    """Translate final Evidence into the source blocks expected by the LLM."""

    def __init__(
        self,
        max_chars_per_evidence: int = 1_200,
        max_total_context_chars: int = 15_000,
    ) -> None:
        self.max_chars_per_evidence = max_chars_per_evidence
        self.max_total_context_chars = max_total_context_chars

    def build(self, evidence: list[Evidence]) -> str:
        """Return numbered evidence blocks within per-item and total limits."""
        # Context is now based on retrieval-selected Evidence rather than the
        # first N characters of every Document. Scores stay in the trace so the
        # answer prompt contains only useful source metadata and passage text.
        return format_evidence_blocks(
            [(f"Source {index}", item) for index, item in enumerate(evidence, 1)],
            self.max_chars_per_evidence,
            self.max_total_context_chars,
        )


def format_evidence_blocks(
    labeled_evidence: list[tuple[str, Evidence]],
    max_chars_per_evidence: int,
    max_total_context_chars: int,
) -> str:
    """Shared bounded formatter for legacy Source labels and V0.6 Evidence IDs."""
    blocks: list[str] = []
    used_chars = 0
    for label, item in labeled_evidence:
        separator = "\n\n" if blocks else ""
        header = (
            f"[{label}]\nTitle: {item.title or 'Untitled'}\n"
            f"URL: {item.url}\nChunk: {item.chunk_index}\nContent:\n"
        )
        remaining = max_total_context_chars - used_chars - len(separator)
        content_budget = min(
            max_chars_per_evidence, len(item.text), remaining - len(header)
        )
        if content_budget <= 0:
            break
        block = header + item.text[:content_budget]
        blocks.append(block)
        used_chars += len(separator) + len(block)
    return "\n\n".join(blocks)
