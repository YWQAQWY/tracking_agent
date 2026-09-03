"""Manual smoke test for the real local BGE embedding and reranker models.

This script intentionally lives outside pytest: it may download several GB of
model weights on first use and requires substantially more RAM/VRAM than unit
tests built with fakes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow ``python scripts/test_retrieval.py`` from a fresh checkout without
# requiring the project to be installed as a package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.chunk import DocumentChunk
from src.retrieval.embedder import BGEEmbedder
from src.retrieval.reranker import BGEReranker
from src.retrieval.retriever import SemanticRetriever


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    question = "How is reinforcement learning used for robot manipulation?"
    chunks = [
        DocumentChunk(
            id="smoke:0",
            text=(
                "Deep reinforcement learning trains robot manipulation policies "
                "to grasp and move objects through interaction and reward signals."
            ),
            url="https://example.com/robotics",
            title="Robot learning",
            chunk_index=0,
        ),
        DocumentChunk(
            id="smoke:1",
            text="The weather is sunny today and temperatures are mild.",
            url="https://example.com/weather",
            title="Weather",
            chunk_index=0,
        ),
    ]

    embedder = BGEEmbedder()
    candidates = SemanticRetriever(embedder, top_k=2).retrieve(question, chunks)
    embedder.offload()
    reranker = BGEReranker(top_k=2, max_chunks_per_document=2)
    evidence = reranker.rerank(question, candidates)
    reranker.offload()

    print(f"device={embedder.device}")
    print("embedding ranking:")
    for item in candidates:
        print(f"  {item.embedding_score:.6f}  {item.chunk.title}")
    print("reranker ranking:")
    for item in evidence:
        print(f"  {item.rerank_score:.6f}  {item.title}")

    if candidates[0].chunk.title != "Robot learning":
        raise RuntimeError("Embedding smoke test failed: irrelevant text ranked first")
    if evidence[0].title != "Robot learning":
        raise RuntimeError("Reranker smoke test failed: irrelevant text ranked first")
    print("REAL_RETRIEVAL_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
