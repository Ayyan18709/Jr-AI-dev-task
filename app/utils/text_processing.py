"""
Text cleaning and pre-processing utilities.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Normalise unicode, remove control characters, collapse whitespace.
    Preserves sentence structure.
    """
    # Normalise unicode (NFC)
    text = unicodedata.normalize("NFC", text)
    # Replace non-breaking spaces and similar
    text = text.replace("\xa0", " ").replace("\t", " ")
    # Remove null bytes and other control chars (except newline)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    # Collapse multiple newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def truncate_text(text: str, max_words: int) -> str:
    """Truncate text to at most max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def word_count(text: str) -> int:
    return len(text.split())


def sentence_tokenize(text: str) -> list[str]:
    """Simple regex sentence splitter."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
