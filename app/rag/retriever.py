"""
Hybrid retriever: 60% semantic (FAISS) + 40% BM25, with cross-encoder reranking.

Pipeline
────────
1. Dense retrieval  → top BM25_TOP_K candidates via FAISS
2. Sparse retrieval → top BM25_TOP_K candidates via BM25
3. Score fusion     → weighted sum (SEMANTIC_WEIGHT + BM25_WEIGHT = 1.0)
4. Reranking        → cross-encoder (or cosine fallback) narrows to top-k
"""

import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.bm25 import BM25Retriever
from app.rag.embedder import Embedder
from app.rag.vectorstore import VectorStore

settings = get_settings()
logger = get_logger(__name__)


class HybridRetriever:
    """
    Combines dense + sparse retrieval with optional cross-encoder reranking.
    """

    def __init__(
        self,
        embedder: Embedder,
        vectorstore: VectorStore,
        bm25: BM25Retriever,
    ) -> None:
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._bm25 = bm25
        self._reranker = None           # loaded lazily
        self._reranker_lock = threading.Lock()

    # ── reranker ──────────────────────────────────────────────────────────
    def _load_reranker(self) -> None:
        with self._reranker_lock:
            if self._reranker is not None:
                return
            if not settings.USE_RERANKER:
                return
            try:
                from sentence_transformers import CrossEncoder  # noqa: PLC0415

                self._reranker = CrossEncoder(
                    settings.RERANKER_MODEL, max_length=512
                )
                logger.info(f"[Retriever] Reranker loaded: {settings.RERANKER_MODEL}")
            except Exception as exc:
                logger.warning(
                    f"[Retriever] CrossEncoder unavailable ({exc}); "
                    "falling back to cosine rerank."
                )
                self._reranker = "cosine"

    # ── main retrieval API ────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Run the full hybrid → rerank pipeline.

        Returns
        -------
        List of chunk dicts (ordered best → worst), each augmented with:
            _score        : final hybrid score
            _dense_score  : FAISS cosine score
            _sparse_score : BM25 normalised score
        """
        top_k = top_k or settings.RETRIEVAL_TOP_K
        t0 = time.perf_counter()

        # 1. Dense
        dense_results = self._vectorstore.search(query, top_k=settings.BM25_TOP_K)
        # 2. Sparse
        sparse_results = self._bm25.search(query, top_k=settings.BM25_TOP_K)

        # 3. Fuse
        fused = self._fuse(dense_results, sparse_results)

        # 4. Rerank
        if settings.USE_RERANKER:
            self._load_reranker()
            if self._reranker and self._reranker != "cosine":
                fused = self._crossencoder_rerank(query, fused, top_k)
            else:
                fused = self._cosine_rerank(query, fused, top_k)
        else:
            fused = fused[:top_k]

        elapsed = time.perf_counter() - t0
        logger.info(
            f"[Retriever] Retrieved {len(fused)} chunks in {elapsed*1000:.1f}ms "
            f"(dense={len(dense_results)}, sparse={len(sparse_results)})"
        )
        return fused

    # ── fusion ────────────────────────────────────────────────────────────
    def _fuse(
        self,
        dense: List[Tuple[Dict, float]],
        sparse: List[Tuple[Dict, float]],
    ) -> List[Dict]:
        """Weighted score fusion keyed by chunk 'id'."""
        scores: Dict[str, Dict] = {}

        for chunk, score in dense:
            cid = chunk["id"]
            if cid not in scores:
                scores[cid] = {"chunk": chunk, "dense": 0.0, "sparse": 0.0}
            scores[cid]["dense"] = score

        for chunk, score in sparse:
            cid = chunk["id"]
            if cid not in scores:
                scores[cid] = {"chunk": chunk, "dense": 0.0, "sparse": 0.0}
            scores[cid]["sparse"] = score

        fused: List[Dict] = []
        for cid, data in scores.items():
            hybrid = (
                settings.SEMANTIC_WEIGHT * data["dense"]
                + settings.BM25_WEIGHT * data["sparse"]
            )
            c = dict(data["chunk"])
            c["_score"] = round(hybrid, 6)
            c["_dense_score"] = round(data["dense"], 6)
            c["_sparse_score"] = round(data["sparse"], 6)
            fused.append(c)

        fused.sort(key=lambda x: x["_score"], reverse=True)
        return fused

    # ── reranking strategies ──────────────────────────────────────────────
    def _crossencoder_rerank(
        self, query: str, candidates: List[Dict], top_k: int
    ) -> List[Dict]:
        """Rerank with a cross-encoder; fallback to cosine on error."""
        try:
            pairs = [(query, c["text"]) for c in candidates]
            ce_scores = self._reranker.predict(pairs)
            for c, s in zip(candidates, ce_scores):
                c["_score"] = float(s)
            candidates.sort(key=lambda x: x["_score"], reverse=True)
            return candidates[:top_k]
        except Exception as exc:
            logger.warning(f"[Retriever] CrossEncoder predict failed ({exc}); using cosine.")
            return self._cosine_rerank(query, candidates, top_k)

    def _cosine_rerank(
        self, query: str, candidates: List[Dict], top_k: int
    ) -> List[Dict]:
        """Rerank by cosine similarity between query and chunk embeddings."""
        if not candidates:
            return []
        q_vec = self._embedder.embed(query)                   # (dim,)
        texts = [c["text"] for c in candidates]
        chunk_vecs = self._embedder.embed_batch(texts)        # (N, dim)
        # Vectors already L2-normalised → cosine = dot product
        sims = (chunk_vecs @ q_vec).tolist()
        for c, sim in zip(candidates, sims):
            c["_score"] = round(float(sim), 6)
        candidates.sort(key=lambda x: x["_score"], reverse=True)
        return candidates[:top_k]
