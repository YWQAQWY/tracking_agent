"""Bounded evidence-only context used by the sufficiency critic."""

from src.models.evidence import Evidence


class CriticContextBuilder:
    """Render enough pooled evidence to judge coverage without prompt bloat."""

    def __init__(
        self,
        max_evidence: int = 10,
        max_chars_per_evidence: int = 800,
        max_total_chars: int = 10_000,
    ) -> None:
        if min(max_evidence, max_chars_per_evidence, max_total_chars) < 1:
            raise ValueError("Critic context limits 必须大于 0")
        self.max_evidence = max_evidence
        self.max_chars_per_evidence = max_chars_per_evidence
        self.max_total_chars = max_total_chars

    def build(self, evidence: list[Evidence]) -> str:
        sections: list[str] = []
        used = 0
        for index, item in enumerate(evidence[: self.max_evidence], start=1):
            content = item.text[: self.max_chars_per_evidence].strip()
            section = (
                f"EVIDENCE {index}\n"
                f"Title: {item.title or 'Untitled'}\n"
                f"URL: {item.url}\n"
                f"Content:\n{content}"
            )
            remaining = self.max_total_chars - used
            if remaining <= 0:
                break
            if len(section) > remaining:
                section = section[:remaining].rstrip()
            if section:
                sections.append(section)
                used += len(section) + 2
        return "\n\n".join(sections)
