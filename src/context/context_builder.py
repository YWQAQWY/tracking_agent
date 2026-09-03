"""Build bounded, source-aware prompt context from Documents."""

from src.models.document import Document


class ContextBuilder:
    """Translate crawler output into the source blocks expected by the LLM."""

    def __init__(
        self,
        max_chars_per_document: int = 6_000,
        max_total_context_chars: int = 15_000,
    ) -> None:
        self.max_chars_per_document = max_chars_per_document
        self.max_total_context_chars = max_total_context_chars

    def build(self, documents: list[Document]) -> str:
        """Return numbered source blocks within per-document and total limits."""
        # V0.2 deliberately uses character truncation so the complete
        # Search -> Read pipeline stays visible. Semantic chunk retrieval is a
        # later-stage replacement for this one small policy.
        blocks: list[str] = []
        used_chars = 0

        for index, document in enumerate(documents, start=1):
            separator = "\n\n" if blocks else ""
            title = document.title or "Untitled"
            header = (
                f"[Source {index}]\n"
                f"Title: {title}\n"
                f"URL: {document.url}\n"
                "Content:\n"
            )
            remaining = self.max_total_context_chars - used_chars - len(separator)
            content_budget = min(
                self.max_chars_per_document,
                len(document.text),
                remaining - len(header),
            )
            if content_budget <= 0:
                break

            block = header + document.text[:content_budget]
            blocks.append(block)
            used_chars += len(separator) + len(block)

        return "\n\n".join(blocks)

