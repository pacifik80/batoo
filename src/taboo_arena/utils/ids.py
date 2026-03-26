"""Identifiers used for runs and rounds."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4


def new_run_id() -> str:
    """Create a stable-looking run identifier."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{stamp}_{uuid4().hex[:8]}"


def new_round_id() -> str:
    """Create a round identifier."""
    return f"round_{uuid4().hex[:10]}"


def new_batch_id() -> str:
    """Create a batch identifier."""
    return f"batch_{uuid4().hex[:10]}"

