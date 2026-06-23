"""
Prometheus metrics for production monitoring.

Metrics exposed
───────────────
rag_requests_total          Counter   – total HTTP requests, by endpoint
rag_request_latency_seconds Histogram – end-to-end request latency
rag_retrieval_latency_ms    Histogram – retrieval-only latency
rag_inference_latency_ms    Histogram – LLM inference latency
rag_cache_hits_total        Counter   – cache hits
rag_cache_misses_total      Counter   – cache misses
rag_cache_hit_ratio         Gauge     – rolling cache hit ratio
rag_chunks_indexed          Gauge     – total indexed chunks
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Request counters ──────────────────────────────────────────────────────────
requests_total = Counter(
    "rag_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)

# ── Latency histograms ────────────────────────────────────────────────────────
request_latency = Histogram(
    "rag_request_latency_seconds",
    "End-to-end request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

retrieval_latency = Histogram(
    "rag_retrieval_latency_ms",
    "Retrieval latency in milliseconds",
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000],
)

inference_latency = Histogram(
    "rag_inference_latency_ms",
    "LLM inference latency in milliseconds",
    buckets=[100, 500, 1000, 2000, 5000, 10000, 30000],
)

# ── Cache counters ────────────────────────────────────────────────────────────
cache_hits_total = Counter(
    "rag_cache_hits_total",
    "Total cache hits",
)

cache_misses_total = Counter(
    "rag_cache_misses_total",
    "Total cache misses",
)

cache_hit_ratio = Gauge(
    "rag_cache_hit_ratio",
    "Rolling cache hit ratio (0-1)",
)

# ── Index stats ───────────────────────────────────────────────────────────────
chunks_indexed = Gauge(
    "rag_chunks_indexed",
    "Total document chunks currently indexed",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def update_cache_ratio(hits: int, misses: int, **kwargs) -> None:
    """Recompute and update the cache hit ratio gauge."""
    total = hits + misses
    if total > 0:
        cache_hit_ratio.set(hits / total)
    else:
        cache_hit_ratio.set(0.0)
