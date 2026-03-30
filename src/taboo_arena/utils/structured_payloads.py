"""Helpers for detecting structured model payloads that should not become visible utterances."""

from __future__ import annotations

import json


def looks_like_structured_payload(text: str) -> bool:
    """Return whether a model string looks like JSON/structured controller output."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] in {"{", "["}:
        return True

    lowered = stripped.casefold()
    if any(marker in lowered for marker in ('"candidates"', "'candidates'", '"guesses"', "'guesses'")):
        return True

    if (":" in stripped and "{" in stripped and "}" in stripped) or (
        "," in stripped and "[" in stripped and "]" in stripped
    ):
        return True

    try:
        payload = json.loads(stripped)
    except Exception:
        return False
    return isinstance(payload, (dict, list))
