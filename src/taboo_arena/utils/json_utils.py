"""JSON parsing helpers."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_first_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a text response."""
    text = text.strip()
    if not text:
        raise ValueError("Empty response, expected JSON object.")
    if text.startswith("{") and text.endswith("}"):
        value = _load_json_object(text)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Could not locate a JSON object in model output.")
    value = _load_json_object(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def _load_json_object(text: str) -> Any:
    """Load a JSON object, repairing common model escaping quirks when safe."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        repaired = _repair_commonly_escaped_json(text)
        if repaired is None:
            raise
        value = json.loads(repaired)

    if isinstance(value, str):
        nested = value.strip()
        if nested.startswith("{") and nested.endswith("}"):
            return _load_json_object(nested)
    return value


def _repair_commonly_escaped_json(text: str) -> str | None:
    """Fix common backslash-escaped quote patterns emitted by chat models."""
    stripped = text.strip()
    if r"\"" not in stripped:
        return None
    repaired = stripped.replace(r"\"", '"')
    return repaired if repaired != stripped else None
