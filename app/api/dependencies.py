"""
FastAPI dependency providers.
All heavy objects (LLM, embedder, vectorstore …) are created once at startup
and injected via FastAPI's Depends() mechanism.
"""

from functools import lru_cache

from app.core.cache import RedisCache
from app.core.memory import MemoryManager
from app.llm.model import QwenLLM
from app.rag.bm25 import BM25Retriever
from app.rag.embedder import Embedder
from app.rag.retriever import HybridRetriever
from app.rag.vectorstore import VectorStore


@lru_cache()
def get_llm() -> QwenLLM:
    """Return the singleton LLM (loaded at startup)."""
    return QwenLLM()


@lru_cache()
def get_embedder() -> Embedder:
    """Return the singleton Embedder (loaded at startup)."""
    return Embedder()


@lru_cache()
def get_vectorstore() -> VectorStore:
    embedder = get_embedder()
    return VectorStore(embedder)


@lru_cache()
def get_bm25() -> BM25Retriever:
    return BM25Retriever()


@lru_cache()
def get_retriever() -> HybridRetriever:
    return HybridRetriever(
        embedder=get_embedder(),
        vectorstore=get_vectorstore(),
        bm25=get_bm25(),
    )


@lru_cache()
def get_cache() -> RedisCache:
    return RedisCache()


@lru_cache()
def get_memory_manager() -> MemoryManager:
    return MemoryManager()
