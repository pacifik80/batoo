"""Logging-related schemas."""

from __future__ import annotations

from pydantic import BaseModel


class RoundSummaryRecord(BaseModel):
    """Per-round summary row."""

    run_id: str
    round_id: str
    card_id: str
    target: str
    solved: bool
    solved_on_attempt: int | None = None
    total_guess_attempts_used: int
    total_visible_guesses_made: int = 0
    total_clue_repairs: int
    first_clue_passed_without_repair: bool
    clue_repaired_successfully: bool
    clue_not_repaired: bool
    terminal_reason: str
    cluer_model_id: str
    guesser_model_id: str
    judge_model_id: str
    total_latency_ms: float
