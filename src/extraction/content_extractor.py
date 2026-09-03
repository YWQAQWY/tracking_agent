"""Turn noisy HTML into a clean Document suitable for an LLM."""

from __future__ import annotations

import logging

import trafilatura
from bs4 import BeautifulSoup

from src.models.document import Document


logger = logging.getLogger(__name__)


class ContentExtractor:
    """Extract main text with Trafilatura, then a small BeautifulSoup fallback."""

    def __init__(self, min_content_length: int = 200) -> None:
        self.min_content_length = min_content_length

    def extract(
        self,
        html: str,
        url: str,
        title_hint: str | None = None,
    ) -> Document | None:
        """Return a useful Document, or None when extraction is too weak."""
        if not html.strip():
            logger.warning("No HTML to extract from %s", url)
            return None

        title = self._extract_title(html) or title_hint
        text = self._extract_with_trafilatura(html)
        if not self._is_useful(text):
            text = self._extract_with_beautifulsoup(html)

        if not self._is_useful(text):
            logger.warning("Extracted content is too short from %s", url)
            return None

        clean_text = text.strip()
        logger.info("Extracted %d chars from %s", len(clean_text), url)
        return Document(url=url, title=title, text=clean_text)

    @staticmethod
    def _extract_with_trafilatura(html: str) -> str | None:
        try:
            return trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_links=False,
                include_images=False,
            )
        except Exception as exc:
            logger.warning("Trafilatura extraction failed: %s", exc.__class__.__name__)
            return None

    @classmethod
    def _extract_with_beautifulsoup(cls, html: str) -> str:
        # Raw HTML contains navigation, JavaScript, CSS, cookie banners and
        # adverts.  Removing them preserves the model's limited context window.
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all(
            ["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]
        ):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        return cls._normalize_text(root.get_text("\n", strip=True))

    @staticmethod
    def _extract_title(html: str) -> str | None:
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            return title or None
        return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        lines = (" ".join(line.split()) for line in text.splitlines())
        return "\n".join(line for line in lines if line)

    def _is_useful(self, text: str | None) -> bool:
        return bool(text and len(text.strip()) >= self.min_content_length)

