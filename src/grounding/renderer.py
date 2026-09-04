"""Deterministically render verified claims and only their cited sources."""

from __future__ import annotations

from collections import defaultdict

from src.grounding.models import (
    CitationSource,
    CoverageResult,
    VerifiedAnswerDraft,
)
from src.grounding.registry import EvidenceRegistry


class GroundedAnswerRenderer:
    """Format verified claim text without giving an LLM another chance to alter it."""

    def render(
        self,
        question: str,
        draft: VerifiedAnswerDraft,
        coverage: CoverageResult | None,
        registry: EvidenceRegistry,
    ) -> tuple[str, tuple[CitationSource, ...]]:
        url_numbers: dict[str, int] = {}
        url_evidence_ids: dict[str, list[str]] = defaultdict(list)
        url_chunks: dict[str, list[int]] = defaultdict(list)
        url_titles: dict[str, str | None] = {}
        rendered_sections: list[str] = []

        for section in draft.sections:
            lines: list[str] = []
            for claim in section.claims:
                citations: list[int] = []
                for evidence_id in claim.evidence_ids:
                    evidence = registry.get(evidence_id)
                    if evidence is None:
                        continue
                    if evidence.url not in url_numbers:
                        url_numbers[evidence.url] = len(url_numbers) + 1
                        url_titles[evidence.url] = evidence.title
                    number = url_numbers[evidence.url]
                    if number not in citations:
                        citations.append(number)
                    if evidence_id not in url_evidence_ids[evidence.url]:
                        url_evidence_ids[evidence.url].append(evidence_id)
                    if evidence.chunk_index not in url_chunks[evidence.url]:
                        url_chunks[evidence.url].append(evidence.chunk_index)
                if citations:
                    suffix = "".join(f"[{number}]" for number in citations)
                    lines.append(f"{claim.text} {suffix}")
            if lines:
                heading = f"## {section.heading}\n\n" if section.heading else ""
                rendered_sections.append(heading + "\n\n".join(lines))

        if coverage and coverage.missing_aspects:
            missing = "、".join(coverage.missing_aspects)
            if _uses_cjk(question):
                limitation = f"## 证据局限\n\n现有证据不足以可靠回答：{missing}。"
            else:
                limitation = (
                    "## Evidence limitations\n\nThe retrieved evidence was not "
                    "sufficient to answer reliably: "
                    f"{', '.join(coverage.missing_aspects)}."
                )
            rendered_sections.append(limitation)

        if not rendered_sections:
            rendered_sections.append(
                "现有证据不足以生成经过验证的事实性回答。"
                if _uses_cjk(question)
                else (
                    "The available evidence was insufficient for a verified "
                    "factual answer."
                )
            )

        sources = tuple(
            CitationSource(
                citation_number=number,
                url=url,
                title=url_titles[url],
                evidence_ids=tuple(url_evidence_ids[url]),
                chunk_indexes=tuple(url_chunks[url]),
            )
            for url, number in url_numbers.items()
        )
        return "\n\n".join(rendered_sections), sources


def _uses_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)
