"""
Text chunker with configurable size, overlap, and section-title detection.
Produces chunk dicts ready for FAISS + BM25 indexing.
"""

import re
import uuid
from typing import Dict, List

from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Heuristic patterns that look like CV section headers
_SECTION_RE = re.compile(
    r"^(education|experience|skills|projects|certifications|awards|"
    r"publications|languages|interests|summary|objective|profile|"
    r"work history|employment|references|contact|about)\b",
    re.IGNORECASE,
)


def chunk_documents(pages: List[Dict]) -> List[Dict]:
    """
    Split page dicts into overlapping word-level chunks.

    Parameters
    ----------
    pages : output of cv_loader.load_document()

    Returns
    -------
    List of chunk dicts:
        id      : unique UUID string
        text    : chunk content
        source  : source filename
        page    : source page number
        section : detected section title or ""
        chunk_index : sequential index within the document
    """
    chunks: List[Dict] = []
    chunk_size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    chunk_index = 0

    for page in pages:
        words = page["text"].split()
        source = page["source"]
        page_num = page["page"]
        current_section = ""

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            window = words[start:end]
            text = " ".join(window)

            # Detect section heading in first few words
            first_line = " ".join(window[:6])
            m = _SECTION_RE.match(first_line.strip())
            if m:
                current_section = m.group(0).title()

            chunks.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "source": source,
                "page": page_num,
                "section": current_section,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

            if end == len(words):
                break
            start = end - overlap   # sliding overlap

    logger.info(
        f"[Chunker] Produced {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={overlap})"
    )
    return chunks
