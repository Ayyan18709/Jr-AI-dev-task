"""
Miscellaneous helpers: timing, safe file ops, JSON I/O.
"""

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict


@contextmanager
def timer(label: str = ""):
    """Context manager that yields elapsed milliseconds."""
    t0 = time.perf_counter()
    elapsed = {}
    try:
        yield elapsed
    finally:
        elapsed["ms"] = round((time.perf_counter() - t0) * 1000, 2)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Write JSON to file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Load JSON from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist; return path."""
    os.makedirs(path, exist_ok=True)
    return path


def format_sources(chunks: list) -> list:
    """
    Format retrieved chunks into a clean source list for API responses.
    """
    seen_ids = set()
    sources = []
    for c in chunks:
        cid = c.get("id", "")
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        sources.append({
            "id": cid,
            "source": c.get("source", ""),
            "section": c.get("section", ""),
            "page": c.get("page", 0),
            "score": round(c.get("_score", 0.0), 4),
            "text_preview": c.get("text", "")[:200] + "...",
        })
    return sources
