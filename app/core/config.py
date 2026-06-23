"""
Application configuration via Pydantic Settings.
All values can be overridden through environment variables or .env file.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ────────────────────────────────────────────────────────
    APP_NAME: str = "RAG Chatbot API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── LLM ────────────────────────────────────────────────────────────────
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"
    LLM_MAX_NEW_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    LLM_DEVICE: str = "auto"          # auto | cpu | cuda | mps
    LLM_LOAD_IN_4BIT: bool = False    # 4-bit quant (CUDA only)

    # ── Embeddings ─────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DIMENSION: int = 384

    # ── Reranker ───────────────────────────────────────────────────────────
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    USE_RERANKER: bool = True
    RERANKER_TOP_K: int = 5

    # ── Retrieval ──────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5
    SEMANTIC_WEIGHT: float = 0.6
    BM25_WEIGHT: float = 0.4
    BM25_TOP_K: int = 10

    # ── Chunking ───────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 400             # tokens / words approximation
    CHUNK_OVERLAP: int = 50

    # ── Memory ─────────────────────────────────────────────────────────────
    MEMORY_MAX_INTERACTIONS: int = 10
    MEMORY_SUMMARY_TRIGGER: int = 5   # summarise after N interactions

    # ── Redis ──────────────────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL: int = 3600             # seconds (1 h)
    CACHE_ENABLED: bool = True

    # ── Storage paths ──────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = "data/faiss_index"
    CV_UPLOAD_PATH: str = "data/uploads"
    LOGS_PATH: str = "logs"

    # ── Prometheus ─────────────────────────────────────────────────────────
    METRICS_PORT: int = 8001

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",   # ignore unknown env vars like HF_HUB_OFFLINE
    }


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
