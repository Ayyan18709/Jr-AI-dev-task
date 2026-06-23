"""
Pydantic request / response schemas for the FastAPI layer.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── /chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="User question")
    session_id: str = Field("default", description="Session ID for conversation memory")
    mode: str = Field("rag", description="'rag' or 'chat'")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")

    model_config = {"json_schema_extra": {
        "example": {
            "query": "What is my educational background?",
            "session_id": "user-123",
            "mode": "rag",
            "top_k": 5,
        }
    }}


class SourceItem(BaseModel):
    id: str
    source: str
    section: str
    page: int
    score: float
    text_preview: str


class MemoryState(BaseModel):
    session_id: str
    interaction_count: int
    summary: str
    interactions: List[Dict[str, str]]


class ChatResponse(BaseModel):
    answer: str
    mode: str
    session_id: str
    sources: List[SourceItem] = Field(default_factory=list)
    memory_state: MemoryState
    latency_ms: float
    cache_hit: bool


# ── /upload_cv ────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_indexed: int
    processing_time_ms: float


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    llm_loaded: bool
    embedder_loaded: bool
    chunks_indexed: int
    redis_connected: bool
    version: str


# ── /metrics ──────────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    cache_stats: Dict[str, Any]
    chunks_indexed: int
    active_sessions: int
    version: str


# ── /evaluate ─────────────────────────────────────────────────────────────────

class EvalResponse(BaseModel):
    total_samples: int
    evaluated: int
    elapsed_seconds: float
    aggregate_metrics: Dict[str, float]
