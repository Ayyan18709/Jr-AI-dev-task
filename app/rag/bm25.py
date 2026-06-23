"""
BM25 sparse retrieval using rank-bm25.
Wraps BM25Okapi with simple tokenization and rank-normalised scores.
"""

import re
import threading
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    """Lower-case, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25Retriever:
    """
    Sparse retriever backed by BM25Okapi.

    Usage
    -----
    bm25 = BM25Retriever()
    bm25.index(chunks)
    results = bm25.search("machine learning experience", top_k=5)
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: List[Dict] = []
        self._lock = threading.Lock()

    def index(self, chunks: List[Dict]) -> None:
        """
        Build BM25 index from chunk dicts (must have 'text' key).

        Parameters
        ----------
        chunks : list of dicts with at least {"text": str, "id": str}
        """
        if not chunks:
            return
        corpus = [_tokenize(c["text"]) for c in chunks]
        with self._lock:
            self._bm25 = BM25Okapi(corpus)
            self._chunks = list(chunks)
        logger.info(f"[BM25] Indexed {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = None) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-k chunks by BM25 score (normalised to [0, 1]).

        Returns
        -------
        List of (chunk_dict, normalised_score) sorted descending.
        """
        top_k = top_k or settings.BM25_TOP_K
        with self._lock:
            if self._bm25 is None or not self._chunks:
                return []
            tokens = _tokenize(query)
            raw_scores = self._bm25.get_scores(tokens)

        max_score = float(raw_scores.max()) if raw_scores.max() > 0 else 1.0
        norm_scores = raw_scores / max_score

        top_indices = norm_scores.argsort()[::-1][:top_k]
        return [
            (self._chunks[i], float(norm_scores[i]))
            for i in top_indices
            if norm_scores[i] > 0
        ]

    def clear(self) -> None:
        with self._lock:
            self._bm25 = None
            self._chunks = []
        logger.info("[BM25] Cleared.")

    @property
    def is_indexed(self) -> bool:
        return self._bm25 is not None and len(self._chunks) > 0

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)
