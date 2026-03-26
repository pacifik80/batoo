from __future__ import annotations

from pathlib import Path

from taboo_arena.app.main import (
    _active_transcript_placeholder,
    _compact_prompt_preview,
    _live_logger_events,
)
from taboo_arena.logging.run_logger import RunLogger


def test_live_logger_events_prefers_in_memory_when_file_lags(tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    logger.emit("app_started", state="idle")
    logger.events.append(
        {
            "event_type": "round_started",
            "run_id": logger.run_id,
            "round_id": "round_live",
            "card_id": "animals:alpaca:0001",
            "state": "generating_clue",
        }
    )

    live_events = _live_logger_events(logger)

    assert [event["event_type"] for event in live_events] == ["app_started", "round_started"]


def test_active_transcript_placeholder_tracks_live_phase(tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    logger.emit("model_load_started", model_id="phi-4-mini-instruct", state="loading_model")

    assert _active_transcript_placeholder(logger) == "Loading model weights into memory..."


def test_compact_prompt_preview_trims_and_collapses_excessive_whitespace() -> None:
    preview = _compact_prompt_preview(
        "  You are the Cluer.  \n\n\n  Target: Alpaca. \n   \n  Output only the clue.   "
    )

    assert preview == "You are the Cluer.\n\nTarget: Alpaca.\n\nOutput only the clue."
