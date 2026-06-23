"""
FAISS vector store for dense retrieval.
Supports add / search / persist / reload.
"""

import os
import pickle
import threading
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.embedder import Embedder

settings = get_settings()
logger = get_logger(__name__)

_META_FILE = "metadata.pkl"
_INDEX_FILE = "index.faiss"


class VectorStore:
    """
    Flat FAISS index with cosine similarity (via inner-product on L2-normalised
    vectors).  Metadata (chunk text + source info) stored alongside as a list.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: List[Dict] = []
        self._lock = threading.Lock()
        self._path = settings.FAISS_INDEX_PATH
        os.makedirs(self._path, exist_ok=True)

    # ── index management ──────────────────────────────────────────────────
    def _ensure_index(self) -> None:
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._embedder.dimension)

    def add(self, chunks: List[Dict]) -> None:
        """
        Index a list of chunk dicts.

        Each dict must have:
            "text"   : str   – the chunk content
            "id"     : str   – unique chunk identifier
            "source" : str   – source filename
            "section": str   – section title (optional)
        """
        if not chunks:
            return
        self._ensure_index()
        texts = [c["text"] for c in chunks]
        vectors = self._embedder.embed_batch(texts)   # (N, dim), normalised

        with self._lock:
            self._index.add(vectors)
            self._metadata.extend(chunks)

        logger.info(f"[VectorStore] Added {len(chunks)} chunks. Total: {len(self._metadata)}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-k chunks by cosine similarity.

        Returns
        -------
        List of (chunk_dict, score) sorted by descending score.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        q_vec = self._embedder.embed(query).reshape(1, -1).astype(np.float32)

        with self._lock:
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self._metadata[idx], float(score)))
        return results

    def clear(self) -> None:
        with self._lock:
            self._index = faiss.IndexFlatIP(self._embedder.dimension)
            self._metadata = []
        logger.info("[VectorStore] Cleared.")

    # ── persistence ───────────────────────────────────────────────────────
    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        with self._lock:
            if self._index is None:
                return
            faiss.write_index(self._index, os.path.join(self._path, _INDEX_FILE))
            with open(os.path.join(self._path, _META_FILE), "wb") as f:
                pickle.dump(self._metadata, f)
        logger.info(f"[VectorStore] Saved to {self._path}")

    def load(self) -> bool:
        """Load persisted index from disk. Returns True on success."""
        idx_path = os.path.join(self._path, _INDEX_FILE)
        meta_path = os.path.join(self._path, _META_FILE)
        if not os.path.exists(idx_path):
            logger.info("[VectorStore] No persisted index found.")
            return False
        with self._lock:
            self._index = faiss.read_index(idx_path)
            with open(meta_path, "rb") as f:
                self._metadata = pickle.load(f)
        logger.info(
            f"[VectorStore] Loaded {len(self._metadata)} chunks from {self._path}"
        )
        return True

    @property
    def total_chunks(self) -> int:
        return len(self._metadata)

    @property
    def all_chunks(self) -> List[Dict]:
        return self._metadata
