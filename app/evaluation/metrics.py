"""
RAG evaluation metrics (fully local – no external APIs required).

Metrics
───────
faithfulness     : fraction of answer sentences supported by retrieved context
context_precision: fraction of retrieved chunks that are relevant to the answer
context_recall   : fraction of answer information covered by retrieved context
answer_relevance : cosine similarity between question and answer embeddings
"""

import re
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _word_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word sets (case-insensitive)."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ── individual metrics ────────────────────────────────────────────────────────

def faithfulness(answer: str, context_chunks: List[str], threshold: float = 0.15) -> float:
    """
    Faithfulness = fraction of answer sentences that are 'supported' by
    at least one context chunk (word-overlap ≥ threshold).

    Range: [0, 1]  – 1 = fully grounded in context.
    """
    sents = _sentences(answer)
    if not sents or not context_chunks:
        return 0.0
    supported = 0
    context_blob = " ".join(context_chunks)
    for s in sents:
        if _word_overlap(s, context_blob) >= threshold:
            supported += 1
    return round(supported / len(sents), 4)


def context_precision(
    retrieved_chunks: List[str],
    reference_answer: str,
    threshold: float = 0.1,
) -> float:
    """
    Context precision = fraction of retrieved chunks relevant to the answer.

    Range: [0, 1]
    """
    if not retrieved_chunks:
        return 0.0
    relevant = sum(
        1 for c in retrieved_chunks
        if _word_overlap(c, reference_answer) >= threshold
    )
    return round(relevant / len(retrieved_chunks), 4)


def context_recall(
    answer: str,
    retrieved_chunks: List[str],
    threshold: float = 0.1,
) -> float:
    """
    Context recall = fraction of answer sentences that appear in retrieved context.

    Range: [0, 1]
    """
    sents = _sentences(answer)
    if not sents or not retrieved_chunks:
        return 0.0
    context_blob = " ".join(retrieved_chunks)
    covered = sum(
        1 for s in sents
        if _word_overlap(s, context_blob) >= threshold
    )
    return round(covered / len(sents), 4)


def answer_relevance(
    question: str,
    answer: str,
    embedder=None,
) -> float:
    """
    Answer relevance = cosine similarity between question and answer embeddings.
    Falls back to word-overlap Jaccard if no embedder is supplied.

    Range: [0, 1]
    """
    if embedder is not None:
        try:
            q_vec = embedder.embed(question).reshape(1, -1)
            a_vec = embedder.embed(answer).reshape(1, -1)
            sim = float(cosine_similarity(q_vec, a_vec)[0][0])
            return round(max(0.0, sim), 4)
        except Exception:
            pass
    # fallback
    return round(_word_overlap(question, answer), 4)


# ── composite scorer ──────────────────────────────────────────────────────────

def compute_all_metrics(
    question: str,
    answer: str,
    retrieved_chunks: List[str],
    reference_answer: Optional[str] = None,
    embedder=None,
) -> Dict[str, float]:
    """
    Compute all RAG metrics in one call.

    Parameters
    ----------
    question         : user question
    answer           : LLM-generated answer
    retrieved_chunks : list of chunk texts used as context
    reference_answer : ground-truth answer (if available, else answer is used)
    embedder         : optional Embedder instance for cosine-based relevance

    Returns
    -------
    Dict with keys: faithfulness, context_precision, context_recall,
                    answer_relevance, composite_score
    """
    ref = reference_answer if reference_answer else answer
    faith = faithfulness(answer, retrieved_chunks)
    cp = context_precision(retrieved_chunks, ref)
    cr = context_recall(ref, retrieved_chunks)
    ar = answer_relevance(question, answer, embedder)
    composite = round((faith + cp + cr + ar) / 4, 4)

    return {
        "faithfulness": faith,
        "context_precision": cp,
        "context_recall": cr,
        "answer_relevance": ar,
        "composite_score": composite,
    }
