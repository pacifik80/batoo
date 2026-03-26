"""Shared engine result types."""

from __future__ import annotations

from dataclasses import dataclass, field

from taboo_arena.judge.llm import GuessJudgeResult, LLMJudgeResult
from taboo_arena.judge.logical import LogicalValidationResult
from taboo_arena.judge.merge import MergedJudgeResult


@dataclass(slots=True)
class RepairTrace:
    """One clue draft and its validation trace."""

    repair_no: int
    clue_text_raw: str
    clue_text_normalized: str
    logical_result: LogicalValidationResult
    llm_result: LLMJudgeResult
    merged_result: MergedJudgeResult
    latency_ms: float
    prompt_template_id: str
    selected_angle: str = ""
    candidate_clues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GuessTrace:
    """One guess trace."""

    attempt_no: int
    guess_text_raw: str
    guess_text_normalized: str
    latency_ms: float
    prompt_template_id: str
    match_status: str = "incorrect"
    match_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    hidden_retry_count: int = 0
    shortlist_candidates: list[str] = field(default_factory=list)
    guess_judge_result: GuessJudgeResult | None = None
    judge_disagreement: bool = False


@dataclass(slots=True)
class AttemptTrace:
    """All events for a single guess attempt."""

    attempt_no: int
    repairs: list[RepairTrace] = field(default_factory=list)
    accepted_clue: str | None = None
    accepted_repair_no: int | None = None
    guess: GuessTrace | None = None
