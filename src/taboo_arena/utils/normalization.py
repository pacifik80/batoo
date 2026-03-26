"""Text normalization helpers shared across the app."""

from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Iterable

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for comparison."""
    cleaned = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def strip_punctuation(text: str) -> str:
    """Remove punctuation characters from text."""
    table = str.maketrans("", "", string.punctuation)
    return normalize_text(text).translate(table).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize a string into simple word-like tokens."""
    return [token.lower() for token in TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text))]


def slugify(text: str) -> str:
    """Create a filesystem-friendly slug."""
    collapsed = strip_punctuation(text)
    collapsed = re.sub(r"[^a-z0-9]+", "-", collapsed)
    return collapsed.strip("-") or "item"


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    """Return items with duplicates removed while preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered

