"""
CV / document loader: supports PDF and TXT formats.
Returns a list of raw page/paragraph strings with source metadata.
"""

import os
from typing import Dict, List

from app.core.config import get_settings
from app.core.logger import get_logger
from app.utils.text_processing import clean_text

settings = get_settings()
logger = get_logger(__name__)


def load_document(file_path: str) -> List[Dict]:
    """
    Load a PDF or TXT file and return a list of page/block dicts.

    Each dict has:
        "text"   : str  – cleaned block content
        "page"   : int  – page number (0-based; 0 for TXT)
        "source" : str  – filename
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return _load_pdf(file_path)
    elif ext == ".txt":
        return _load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt")


# ── PDF ───────────────────────────────────────────────────────────────────────

def _load_pdf(path: str) -> List[Dict]:
    """Try pdfplumber first (better layout), fall back to PyPDF2."""
    try:
        return _load_pdf_pdfplumber(path)
    except Exception as exc:
        logger.warning(f"[Loader] pdfplumber failed ({exc}); trying PyPDF2 ...")
        return _load_pdf_pypdf2(path)


def _load_pdf_pdfplumber(path: str) -> List[Dict]:
    import pdfplumber  # noqa: PLC0415

    pages: List[Dict] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw = page.extract_text() or ""
            text = clean_text(raw)
            if text:
                pages.append({
                    "text": text,
                    "page": i,
                    "source": os.path.basename(path),
                })
    logger.info(f"[Loader] PDF loaded via pdfplumber: {len(pages)} pages from {path}")
    return pages


def _load_pdf_pypdf2(path: str) -> List[Dict]:
    import PyPDF2  # noqa: PLC0415

    pages: List[Dict] = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            text = clean_text(raw)
            if text:
                pages.append({
                    "text": text,
                    "page": i,
                    "source": os.path.basename(path),
                })
    logger.info(f"[Loader] PDF loaded via PyPDF2: {len(pages)} pages from {path}")
    return pages


# ── TXT ───────────────────────────────────────────────────────────────────────

def _load_txt(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # Split on double newlines (paragraphs)
    blocks = [clean_text(b) for b in raw.split("\n\n") if b.strip()]
    pages = [
        {"text": b, "page": 0, "source": os.path.basename(path)}
        for b in blocks if b
    ]
    logger.info(f"[Loader] TXT loaded: {len(pages)} blocks from {path}")
    return pages
