"""
Production-grade structured logging with rotating file + console handlers.
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone

from app.core.config import get_settings

settings = get_settings()


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line (for log aggregators)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Allow callers to pass structured extras via LoggerAdapter / extra={}
        for key, val in record.__dict__.items():
            if key.startswith("x_"):
                payload[key] = val
        return json.dumps(payload, ensure_ascii=False)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a named logger (idempotent)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── Console handler ────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # ── Rotating file handler (JSON) ───────────────────────────────────────
    os.makedirs(settings.LOGS_PATH, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        filename=os.path.join(settings.LOGS_PATH, "app.log"),
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(_JSONFormatter())

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Public helper – returns (or creates) a named logger."""
    return setup_logger(name, level=logging.DEBUG if settings.DEBUG else logging.INFO)
