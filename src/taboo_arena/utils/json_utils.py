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
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Could not locate a JSON object in model output.")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value

