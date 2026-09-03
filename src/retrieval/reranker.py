"""Cross-encoder reranking and final evidence diversity selection."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np

from src.models.evidence import Evidence, ScoredChunk
from src.retrieval.device import resolve_device


logger = logging.getLogger(__name__)


class RerankerError(RuntimeError):
    """Raised when the local reranker cannot load or score candidates."""


class Reranker(ABC):
    """Convert embedding candidates into a small final Evidence list."""

    model_name: str
    device: str

    @abstractmethod
    def rerank(
        self,
        question: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[Evidence]:
        """Return final Evidence in descending relevance order."""


class BGEReranker(Reranker):
    """Lazy CrossEncoder adapter for BAAI/bge-reranker-v2-m3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 8,
        top_k: int = 6,
        max_chunks_per_document: int = 2,
        device: str = "auto",
        model: Any | None = None,
    ) -> None:
        self._validate_limits(batch_size, top_k, max_chunks_per_document)
        self.model_name = model_name
        self.batch_size = batch_size
        self.top_k = top_k
        self.max_chunks_per_document = max_chunks_per_document
        self.device = resolve_device(device) if model is None else device
        self._model = model
        self._offloaded = False

    def rerank(
        self,
        question: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[Evidence]:
        if not candidates:
            return []
        model = self._load_model()
        pairs = [(question, candidate.chunk.text) for candidate in candidates]
        try:
            # A cross-encoder jointly reads each (question, passage) pair. This
            # is slower but more precise, so it only sees embedding top-N.
            raw_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
        except Exception as exc:
            raise RerankerError(
                f"Reranker model {self.model_name} 评分失败；"
                "请检查显存/内存是否充足。"
            ) from exc
        if len(scores) != len(candidates):
            raise RerankerError(
                "Reranker 返回的 score 数量与候选数量不一致。"
            )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        evidence = self._select_diverse(ranked, top_k or self.top_k)
        logger.info("Reranked %d candidates", len(candidates))
        logger.info("Selected top %d evidence chunks", len(evidence))
        return evidence

    def _load_model(self) -> Any:
        if self._model is not None:
            if self._offloaded:
                try:
                    self._model.model.to(self.device)
                    self._offloaded = False
                except Exception as exc:
                    raise RerankerError(
                        f"Reranker model {self.model_name} "
                        f"无法移回 {self.device}；"
                        "请检查显存/内存是否充足。"
                    ) from exc
            return self._model
        logger.info("Loading reranker model %s on %s", self.model_name, self.device)
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self.model_name, device=self.device, max_length=512)
            if self.device == "cuda":
                model.model.half()
            self._model = model
            return model
        except Exception as exc:
            raise RerankerError(
                f"Failed to load reranker model {self.model_name}. "
                "请检查 Hugging Face 网络连接，或先手动下载该模型。"
            ) from exc

    def offload(self) -> None:
        """Keep loaded weights in RAM while returning CUDA memory to Ollama."""
        if self._model is None or self.device != "cuda" or self._offloaded:
            return
        try:
            import torch

            self._model.model.to("cpu")
            self._offloaded = True
            torch.cuda.empty_cache()
            logger.info("Offloaded reranker model to CPU")
        except Exception as exc:
            logger.warning("Could not offload reranker model: %s", exc)

    def _select_diverse(
        self,
        ranked: list[tuple[ScoredChunk, np.floating[Any]]],
        top_k: int,
    ) -> list[Evidence]:
        evidence: list[Evidence] = []
        per_url: defaultdict[str, int] = defaultdict(int)
        for candidate, score in ranked:
            url = candidate.chunk.url
            if per_url[url] >= self.max_chunks_per_document:
                continue
            per_url[url] += 1
            evidence.append(Evidence.from_scored_chunk(candidate, float(score)))
            if len(evidence) >= top_k:
                break
        return evidence

    @staticmethod
    def _validate_limits(batch_size: int, top_k: int, max_per_document: int) -> None:
        if min(batch_size, top_k, max_per_document) < 1:
            raise ValueError("Reranker batch/top-k/diversity 配置必须大于 0")


class EmbeddingOnlyReranker(Reranker):
    """Ablation mode that preserves embedding order and skips model loading."""

    model_name = "disabled"
    device = "none"

    def __init__(self, top_k: int = 6, max_chunks_per_document: int = 2) -> None:
        if top_k < 1 or max_chunks_per_document < 1:
            raise ValueError("Evidence top-k/diversity 配置必须大于 0")
        self.top_k = top_k
        self.max_chunks_per_document = max_chunks_per_document

    def rerank(
        self,
        question: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[Evidence]:
        del question
        limit = top_k or self.top_k
        evidence: list[Evidence] = []
        per_url: defaultdict[str, int] = defaultdict(int)
        for candidate in candidates:
            url = candidate.chunk.url
            if per_url[url] >= self.max_chunks_per_document:
                continue
            per_url[url] += 1
            evidence.append(Evidence.from_scored_chunk(candidate, None))
            if len(evidence) >= limit:
                break
        logger.info(
            "Reranker disabled; selected %d embedding candidates", len(evidence)
        )
        return evidence
