"""
API route definitions.
"""

import os
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.api.dependencies import (
    get_bm25,
    get_cache,
    get_embedder,
    get_llm,
    get_memory_manager,
    get_retriever,
    get_vectorstore,
)
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    EvalResponse,
    HealthResponse,
    MemoryState,
    MetricsResponse,
    SourceItem,
    UploadResponse,
)
from app.core.config import get_settings
from app.core.logger import get_logger
from app.ingestion.chunker import chunk_documents
from app.ingestion.cv_loader import load_document
from app.llm.prompt_engine import build_chat_messages, build_rag_messages
from app.monitoring import metrics as prom
from app.utils.helpers import format_sources

settings = get_settings()
logger = get_logger(__name__)
router = APIRouter()


# ── POST /chat ────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    req: ChatRequest,
    llm=Depends(get_llm),
    retriever=Depends(get_retriever),
    cache=Depends(get_cache),
    mem_mgr=Depends(get_memory_manager),
    embedder=Depends(get_embedder),
):
    """
    Main chat endpoint. Supports 'rag' and 'chat' modes.
    Injects conversation memory, checks Redis cache, runs retrieval + LLM.
    """
    t_start = time.perf_counter()
    prom.requests_total.labels(endpoint="/chat", status="processing").inc()

    memory = mem_mgr.get_or_create(req.session_id)
    memory.set_llm(llm)
    memory_ctx = memory.get_context_string()

    # ── Cache check ───────────────────────────────────────────────────────
    cached = cache.get(req.query, memory_ctx, req.mode)
    if cached:
        prom.cache_hits_total.inc()
        prom.update_cache_ratio(**cache.stats)
        prom.requests_total.labels(endpoint="/chat", status="200").inc()
        return ChatResponse(**cached)

    prom.cache_misses_total.inc()

    # ── Retrieval (RAG mode) ───────────────────────────────────────────────
    sources: list = []
    chunk_texts: list[str] = []

    if req.mode == "rag":
        t_ret = time.perf_counter()
        chunks = retriever.retrieve(req.query, top_k=req.top_k)
        ret_ms = (time.perf_counter() - t_ret) * 1000
        prom.retrieval_latency.observe(ret_ms)

        chunk_texts = [c["text"] for c in chunks]
        sources = format_sources(chunks)
        messages = build_rag_messages(req.query, chunk_texts, memory_ctx)
    else:
        messages = build_chat_messages(req.query, memory_ctx)

    # ── LLM Inference ─────────────────────────────────────────────────────
    t_inf = time.perf_counter()
    answer = llm.chat(messages)
    inf_ms = (time.perf_counter() - t_inf) * 1000
    prom.inference_latency.observe(inf_ms)

    # ── Memory update ──────────────────────────────────────────────────────
    memory.add_interaction(req.query, answer)

    # ── Build response ─────────────────────────────────────────────────────
    total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    prom.request_latency.labels(endpoint="/chat").observe(total_ms / 1000)

    mem_state = MemoryState(**memory.to_dict())
    source_items = [SourceItem(**s) for s in sources]

    response_data = ChatResponse(
        answer=answer,
        mode=req.mode,
        session_id=req.session_id,
        sources=source_items,
        memory_state=mem_state,
        latency_ms=total_ms,
        cache_hit=False,
    )

    # ── Cache store ────────────────────────────────────────────────────────
    cache.set(req.query, response_data.model_dump(), memory_ctx, req.mode)
    prom.update_cache_ratio(**cache.stats)
    prom.requests_total.labels(endpoint="/chat", status="200").inc()

    logger.info(
        f"[/chat] session={req.session_id} mode={req.mode} "
        f"latency={total_ms}ms inf={inf_ms:.0f}ms"
    )
    return response_data


# ── POST /upload_cv ───────────────────────────────────────────────────────────

@router.post("/upload_cv", response_model=UploadResponse, tags=["Ingestion"])
async def upload_cv(
    file: UploadFile = File(...),
    vectorstore=Depends(get_vectorstore),
    bm25=Depends(get_bm25),
):
    """
    Upload a CV (PDF or TXT), chunk it, embed it, and index it in FAISS + BM25.
    Clears any previously indexed data before re-indexing.
    """
    prom.requests_total.labels(endpoint="/upload_cv", status="processing").inc()
    t_start = time.perf_counter()

    # Validate
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".pdf", ".txt"):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    # Save to disk
    os.makedirs(settings.CV_UPLOAD_PATH, exist_ok=True)
    save_path = os.path.join(settings.CV_UPLOAD_PATH, file.filename)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    logger.info(f"[/upload_cv] Saved {file.filename} ({len(content)} bytes)")

    # Load → chunk → index
    try:
        pages = load_document(save_path)
        chunks = chunk_documents(pages)

        vectorstore.clear()
        bm25.clear()
        vectorstore.add(chunks)
        bm25.index(chunks)
        vectorstore.save()

        prom.chunks_indexed.set(vectorstore.total_chunks)
    except Exception as exc:
        logger.error(f"[/upload_cv] Indexing failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
    prom.requests_total.labels(endpoint="/upload_cv", status="200").inc()
    prom.request_latency.labels(endpoint="/upload_cv").observe(elapsed_ms / 1000)

    return UploadResponse(
        message="CV indexed successfully.",
        filename=file.filename,
        chunks_indexed=vectorstore.total_chunks,
        processing_time_ms=elapsed_ms,
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health(
    llm=Depends(get_llm),
    embedder=Depends(get_embedder),
    vectorstore=Depends(get_vectorstore),
    cache=Depends(get_cache),
):
    prom.requests_total.labels(endpoint="/health", status="200").inc()
    return HealthResponse(
        status="ok",
        llm_loaded=llm.is_loaded,
        embedder_loaded=embedder.is_loaded,
        chunks_indexed=vectorstore.total_chunks,
        redis_connected=cache.is_connected,
        version=settings.APP_VERSION,
    )


# ── GET /metrics ──────────────────────────────────────────────────────────────

@router.get("/metrics", tags=["System"])
async def metrics_json(
    cache=Depends(get_cache),
    vectorstore=Depends(get_vectorstore),
    mem_mgr=Depends(get_memory_manager),
):
    """Human-readable JSON metrics endpoint."""
    prom.requests_total.labels(endpoint="/metrics", status="200").inc()
    return MetricsResponse(
        cache_stats=cache.stats,
        chunks_indexed=vectorstore.total_chunks,
        active_sessions=len(mem_mgr.list_sessions()),
        version=settings.APP_VERSION,
    )


@router.get("/metrics/prometheus", tags=["System"], include_in_schema=False)
async def metrics_prometheus():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── POST /evaluate ────────────────────────────────────────────────────────────

@router.post("/evaluate", response_model=EvalResponse, tags=["Evaluation"])
async def evaluate(
    llm=Depends(get_llm),
    retriever=Depends(get_retriever),
    embedder=Depends(get_embedder),
):
    """Run the batch RAG evaluation pipeline against data/eval_dataset.json."""
    from app.evaluation.evaluator import run_evaluation  # noqa: PLC0415

    prom.requests_total.labels(endpoint="/evaluate", status="processing").inc()
    report = run_evaluation(retriever=retriever, llm=llm, embedder=embedder)

    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])

    prom.requests_total.labels(endpoint="/evaluate", status="200").inc()
    return EvalResponse(
        total_samples=report["total_samples"],
        evaluated=report["evaluated"],
        elapsed_seconds=report["elapsed_seconds"],
        aggregate_metrics=report["aggregate_metrics"],
    )
