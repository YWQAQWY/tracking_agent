"""Deterministic content-level deduplication after HTML extraction."""

from __future__ import annotations

import hashlib
import logging

from src.models.document import Document


logger = logging.getLogger(__name__)


class ContentDeduplicator:
    """Remove documents whose normalized body text is exactly equivalent."""

    def deduplicate(self, documents: list[Document]) -> list[Document]:
        # Content equality can only be known after extraction: different URLs
        # may be mirrors or reposts of the same normalized article body.
        unique: list[Document] = []
        seen_hashes: set[str] = set()
        for document in documents:
            digest = self.content_hash(document.text)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            unique.append(document)

        logger.info(
            "Content deduplication reduced %d → %d documents",
            len(documents),
            len(unique),
        )
        return unique

    @classmethod
    def content_hash(cls, text: str) -> str:
        normalized = cls.normalize_text(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def normalize_text(text: str) -> str:
        return " ".join(text.casefold().split())
