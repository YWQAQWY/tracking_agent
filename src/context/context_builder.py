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
        blocks: list[str] = []
        used_chars = 0

        for index, item in enumerate(evidence, start=1):
            separator = "\n\n" if blocks else ""
            title = item.title or "Untitled"
            header = (
                f"[Source {index}]\n"
                f"Title: {title}\n"
                f"URL: {item.url}\n"
                f"Chunk: {item.chunk_index}\n"
                "Content:\n"
            )
            remaining = self.max_total_context_chars - used_chars - len(separator)
            content_budget = min(
                self.max_chars_per_evidence,
                len(item.text),
                remaining - len(header),
            )
            if content_budget <= 0:
                break

            block = header + item.text[:content_budget]
            blocks.append(block)
            used_chars += len(separator) + len(block)

        return "\n\n".join(blocks)
