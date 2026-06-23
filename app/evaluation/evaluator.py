"""
Batch evaluation pipeline.
─────────────────────────
Loads an eval dataset (JSON), runs retrieval + generation for each sample,
computes all RAG metrics, and writes a summary report.

Dataset format  (data/eval_dataset.json):
[
  { "question": "...", "reference_answer": "..." },
  ...
]

Usage
─────
python -m app.evaluation.evaluator
"""

import json
import os
import time
from typing import Dict, List

from app.core.config import get_settings
from app.core.logger import get_logger
from app.evaluation.metrics import compute_all_metrics

settings = get_settings()
logger = get_logger(__name__)

_EVAL_DATASET = "data/eval_dataset.json"
_REPORT_PATH = "data/eval_report.json"


def load_eval_dataset(path: str = _EVAL_DATASET) -> List[Dict]:
    if not os.path.exists(path):
        logger.warning(f"[Eval] Dataset not found at {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"[Eval] Loaded {len(data)} samples from {path}")
    return data


def run_evaluation(retriever, llm, embedder=None) -> Dict:
    """
    Run full batch evaluation.

    Parameters
    ----------
    retriever : HybridRetriever instance
    llm       : QwenLLM instance
    embedder  : optional Embedder for answer_relevance metric

    Returns
    -------
    Dict with per-sample results and aggregate statistics.
    """
    from app.llm.prompt_engine import build_rag_messages  # noqa: PLC0415

    dataset = load_eval_dataset()
    if not dataset:
        return {"error": "No evaluation data found."}

    results = []
    metric_totals: Dict[str, float] = {
        "faithfulness": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "answer_relevance": 0.0,
        "composite_score": 0.0,
    }

    logger.info(f"[Eval] Starting evaluation on {len(dataset)} samples ...")
    t_start = time.perf_counter()

    for i, sample in enumerate(dataset):
        question = sample.get("question", "")
        reference = sample.get("reference_answer", "")

        try:
            # 1. Retrieve context
            chunks = retriever.retrieve(question)
            chunk_texts = [c["text"] for c in chunks]

            # 2. Generate answer
            messages = build_rag_messages(question, chunk_texts)
            answer = llm.chat(messages)

            # 3. Compute metrics
            metrics = compute_all_metrics(
                question=question,
                answer=answer,
                retrieved_chunks=chunk_texts,
                reference_answer=reference or None,
                embedder=embedder,
            )

            result = {
                "id": i,
                "question": question,
                "reference_answer": reference,
                "generated_answer": answer,
                "retrieved_chunks": chunk_texts,
                "metrics": metrics,
            }
            results.append(result)

            for k in metric_totals:
                metric_totals[k] += metrics.get(k, 0.0)

            logger.info(
                f"[Eval] Sample {i+1}/{len(dataset)} – "
                f"composite={metrics['composite_score']:.4f}"
            )

        except Exception as exc:
            logger.error(f"[Eval] Sample {i+1} failed: {exc}")
            results.append({"id": i, "question": question, "error": str(exc)})

    n = len([r for r in results if "metrics" in r])
    elapsed = time.perf_counter() - t_start

    aggregates = {
        k: round(v / n, 4) if n else 0.0
        for k, v in metric_totals.items()
    }

    report = {
        "total_samples": len(dataset),
        "evaluated": n,
        "elapsed_seconds": round(elapsed, 2),
        "aggregate_metrics": aggregates,
        "per_sample": results,
    }

    # Persist
    os.makedirs("data", exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"[Eval] Report saved to {_REPORT_PATH}")
    logger.info(f"[Eval] Aggregate metrics: {aggregates}")

    return report
