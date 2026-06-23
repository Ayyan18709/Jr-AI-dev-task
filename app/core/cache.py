"""
Redis cache layer.
─────────────────
• MD5-keyed cache for (query, context, mode) triples.
• Graceful no-op fallback when Redis is unavailable.
• Hit/miss counters exposed for Prometheus metrics.
"""

import hashlib
import json
import threading
from typing import Any, Dict, Optional

import redis

from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class RedisCache:
    """
    Production Redis cache with connection pooling and graceful degradation.

    The cache is disabled automatically when:
      - ``CACHE_ENABLED`` is False in settings, or
      - Redis is unreachable (all operations become no-ops).
    """

    def __init__(self) -> None:
        self._client: Optional[redis.Redis] = None
        self._connected = False
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()
        self._connect()

    # ── connection ────────────────────────────────────────────────────────
    def _connect(self) -> None:
        try:
            pool = redis.ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                max_connections=20,
                decode_responses=True,
            )
            self._client = redis.Redis(
                connection_pool=pool,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            self._client.ping()
            self._connected = True
            logger.info(
                f"[Cache] Redis connected -> "
                f"{settings.REDIS_HOST}:{settings.REDIS_PORT}"
            )
        except Exception as exc:
            self._connected = False
            logger.warning(f"[Cache] Redis unavailable – cache disabled: {exc}")

    # ── key generation ────────────────────────────────────────────────────
    @staticmethod
    def _make_key(query: str, context: str, mode: str) -> str:
        raw = f"{query}||{context}||{mode}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return f"rag:{digest}"

    # ── public API ────────────────────────────────────────────────────────
    def get(
        self, query: str, context: str = "", mode: str = "chat"
    ) -> Optional[Dict[str, Any]]:
        """Return cached payload or None on miss / disabled."""
        if not self._connected or not settings.CACHE_ENABLED:
            return None
        key = self._make_key(query, context, mode)
        try:
            raw = self._client.get(key)
            with self._lock:
                if raw:
                    self._hits += 1
                    logger.info(f"[Cache] HIT  {key[:24]}...")
                    return json.loads(raw)
                self._misses += 1
                logger.debug(f"[Cache] MISS {key[:24]}...")
                return None
        except Exception as exc:
            logger.warning(f"[Cache] GET error: {exc}")
            return None

    def set(
        self,
        query: str,
        payload: Dict[str, Any],
        context: str = "",
        mode: str = "chat",
        ttl: Optional[int] = None,
    ) -> bool:
        """Persist payload in Redis with TTL. Returns True on success."""
        if not self._connected or not settings.CACHE_ENABLED:
            return False
        key = self._make_key(query, context, mode)
        ttl = ttl if ttl is not None else settings.CACHE_TTL
        try:
            self._client.setex(key, ttl, json.dumps(payload, ensure_ascii=False))
            logger.debug(f"[Cache] SET  {key[:24]}... TTL={ttl}s")
            return True
        except Exception as exc:
            logger.warning(f"[Cache] SET error: {exc}")
            return False

    def invalidate(
        self, query: str, context: str = "", mode: str = "chat"
    ) -> bool:
        """Delete a single cache entry."""
        if not self._connected:
            return False
        key = self._make_key(query, context, mode)
        try:
            self._client.delete(key)
            return True
        except Exception as exc:
            logger.warning(f"[Cache] DELETE error: {exc}")
            return False

    # ── stats ─────────────────────────────────────────────────────────────
    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "connected": self._connected,
                "hits": self._hits,
                "misses": self._misses,
                "total": total,
                "hit_rate": round(self._hits / total * 100, 2) if total else 0.0,
            }

    @property
    def is_connected(self) -> bool:
        return self._connected
