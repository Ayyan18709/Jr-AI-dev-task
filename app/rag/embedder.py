"""
Sentence-Transformer embedder with batch support and in-memory cache.
"""

import hashlib
import threading
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class Embedder:
    """
    Wraps SentenceTransformer and adds:
    • batch embedding with configurable batch size
    • simple in-process LRU-style cache (dict-based)
    • thread-safe singleton semantics
    """

    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None
        self._cache: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        """Download / load the embedding model (idempotent)."""
        if self._loaded:
            return
        logger.info(f"[Embedder] Loading {settings.EMBEDDING_MODEL} ...")
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self._loaded = True
        logger.info("[Embedder] Ready.")

    # ── public API ────────────────────────────────────────────────────────
    def embed(self, text: str) -> np.ndarray:
        """Embed a single string; returns a 1-D float32 numpy array."""
        self._ensure_loaded()
        key = self._hash(text)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        vec = self._model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        with self._lock:
            self._cache[key] = vec
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings in batches.

        Returns
        -------
        np.ndarray of shape (N, dim)
        """
        self._ensure_loaded()
        if not texts:
            return np.empty((0, settings.EMBEDDING_DIMENSION), dtype=np.float32)

        # Check cache for each text
        result: List[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        with self._lock:
            for i, t in enumerate(texts):
                key = self._hash(t)
                if key in self._cache:
                    result[i] = self._cache[key]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(t)

        if uncached_texts:
            vecs = self._model.encode(
                uncached_texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(uncached_texts) > 50,
            )
            with self._lock:
                for idx, vec in zip(uncached_indices, vecs):
                    self._cache[self._hash(texts[idx])] = vec
                    result[idx] = vec

        return np.vstack(result).astype(np.float32)

    # ── helpers ───────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call embedder.load() first.")

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @property
    def dimension(self) -> int:
        return settings.EMBEDDING_DIMENSION

    @property
    def is_loaded(self) -> bool:
        return self._loaded
