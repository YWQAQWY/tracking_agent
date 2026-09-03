"""Batch dense embedding abstraction and BGE-M3 implementation."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.retrieval.device import resolve_device


logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when the local embedding model cannot load or encode text."""


class Embedder(ABC):
    """Small interface that keeps model-library details out of the pipeline."""

    model_name: str
    device: str

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into dense float vectors."""


class BGEEmbedder(Embedder):
    """Lazy, reusable SentenceTransformers adapter for BAAI/bge-m3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 16,
        device: str = "auto",
        model: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("embedding batch_size 必须大于 0")
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = resolve_device(device) if model is None else device
        self._model = model
        self._offloaded = False

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = self._load_model()
        try:
            # All chunks are encoded in batches. Normalized vectors let cosine
            # similarity become a fast matrix dot product in the retriever.
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(vectors, dtype=np.float32)
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding model {self.model_name} 编码失败；"
                "请检查显存/内存是否充足。"
            ) from exc

    def _load_model(self) -> Any:
        if self._model is not None:
            if self._offloaded:
                try:
                    self._model.to(self.device)
                    self._offloaded = False
                except Exception as exc:
                    raise EmbeddingError(
                        f"Embedding model {self.model_name} "
                        f"无法移回 {self.device}；"
                        "请检查显存/内存是否充足。"
                    ) from exc
            return self._model
        logger.info("Loading embedding model %s on %s", self.model_name, self.device)
        try:
            # Transformers 5 may spawn a background download of a converted
            # safetensors copy even when the original .bin weights loaded. For
            # bge-m3 that needlessly duplicates more than 2 GB in the cache.
            os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
            from sentence_transformers import SentenceTransformer

            # The official bge-m3 repository publishes PyTorch weights. Newer
            # Transformers versions may otherwise fetch an auto-converted
            # safetensors pull request as a second multi-GB copy.
            model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs={"use_safetensors": False},
            )
            if self.device == "cuda":
                model.half()
            self._model = model
            return model
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model {self.model_name}. "
                "请检查 Hugging Face 网络连接，或先手动下载该模型。"
            ) from exc

    def offload(self) -> None:
        """Keep loaded weights in RAM while returning CUDA memory to Ollama."""
        if self._model is None or self.device != "cuda" or self._offloaded:
            return
        try:
            import torch

            self._model.to("cpu")
            self._offloaded = True
            torch.cuda.empty_cache()
            logger.info("Offloaded embedding model to CPU")
        except Exception as exc:
            logger.warning("Could not offload embedding model: %s", exc)
