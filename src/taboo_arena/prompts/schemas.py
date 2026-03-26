"""Strict prompt-output schemas shared by controller parsing."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ClueAngleLiteral = Literal[
    "type",
    "use",
    "context",
    "effect",
    "part_whole",
    "historical_association",
]


class CluerCandidatePayload(BaseModel):
    """One angle-tagged clue candidate from the cluer."""

    angle: ClueAngleLiteral
    clue: str = Field(min_length=1)


class CluerCandidatesPayload(BaseModel):
    """Structured candidate set returned by the cluer model."""

    candidates: list[CluerCandidatePayload] = Field(default_factory=list)


class CluerRepairFeedbackPayload(BaseModel):
    """Machine-friendly repair feedback sent back to the cluer."""

    blocked_terms: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    blocked_angles: list[ClueAngleLiteral] = Field(default_factory=list)
    allowed_angles: list[ClueAngleLiteral] = Field(default_factory=list)


class ClueJudgePayload(BaseModel):
    """Structured clue-judge verdict."""

    allow: bool
    block_reason_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    matched_surface_forms: list[str] = Field(default_factory=list)
    judge_version: str = "clue_judge_v1"


class GuessJudgePayload(BaseModel):
    """Structured guess-judge verdict."""

    correct: bool
    reason_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    matched_surface_forms: list[str] = Field(default_factory=list)
    judge_version: str = "guess_judge_v1"


class GuesserCandidatesPayload(BaseModel):
    """Structured shortlist returned by the guesser model."""

    guesses: list[str] = Field(default_factory=list)
