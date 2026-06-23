"""
FastAPI application entry point.
Handles startup model loading, middleware, and route registration.
"""

import os
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.dependencies import get_bm25, get_embedder, get_llm, get_vectorstore
from app.api.routes import router
from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger("main")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Production-grade RAG chatbot powered by Qwen2.5-1.5B-Instruct. "
            "Hybrid BM25 + FAISS retrieval with cross-encoder reranking, "
            "conversation memory, Redis cache, and Prometheus monitoring."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging middleware ─────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"{request.method} {request.url.path} "
            f"-> {response.status_code} [{elapsed}ms]"
        )
        return response

    # ── Global exception handler ───────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check logs for details."},
        )

    # ── Startup event: load models + restore index ─────────────────────────
    @app.on_event("startup")
    async def startup():
        logger.info("=" * 60)
        logger.info(f"  {settings.APP_NAME} v{settings.APP_VERSION} starting...")
        logger.info("=" * 60)

        # Load embedder first (needed by vectorstore)
        embedder = get_embedder()
        embedder.load()
        logger.info("[Startup] Embedder ready.")

        # Restore persisted FAISS index (if any)
        vs = get_vectorstore()
        loaded = vs.load()
        if loaded:
            bm25 = get_bm25()
            bm25.index(vs.all_chunks)
            logger.info(
                f"[Startup] Restored {vs.total_chunks} chunks from disk "
                "and rebuilt BM25 index."
            )
        else:
            logger.info("[Startup] No persisted index found – upload a CV to begin.")

        # Load LLM (heaviest operation)
        llm = get_llm()
        llm.load()
        logger.info("[Startup] LLM ready.")

        logger.info("[Startup] All systems operational. API ready.")

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("[Shutdown] Saving FAISS index ...")
        vs = get_vectorstore()
        vs.save()
        logger.info("[Shutdown] Done.")

    # ── Routes ────────────────────────────────────────────────────────────
    app.include_router(router, prefix="")

    # Root redirect
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": f"{settings.APP_NAME} is running.", "docs": "/docs"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
