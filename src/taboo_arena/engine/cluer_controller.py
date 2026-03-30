"""Deterministic controller helpers for structured clue generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.judge.logical import LogicalValidationResult, LogicalValidator
from taboo_arena.prompts.schemas import (
    CluerCandidatePayload,
    CluerCandidatesPayload,
    CluerRepairFeedbackPayload,
)
from taboo_arena.utils.json_utils import extract_first_json_object
from taboo_arena.utils.normalization import dedupe_preserve_order, normalize_text, tokenize
from taboo_arena.utils.structured_payloads import looks_like_structured_payload


class ClueAngle(StrEnum):
    """Closed set of controller-owned clue angles."""

    TYPE = "type"
    USE = "use"
    CONTEXT = "context"
    EFFECT = "effect"
    PART_WHOLE = "part_whole"
    HISTORICAL_ASSOCIATION = "historical_association"


@dataclass(slots=True)
class ClueCandidateEvaluation:
    """Deterministic evaluation result for one clue candidate."""

    angle: ClueAngle
    clue_text_raw: str
    clue_text_normalized: str
    logical_result: LogicalValidationResult
    score: int
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.logical_result.verdict == "pass"


def select_allowed_angles(
    *,
    attempt_no: int,
    used_angles: list[str],
    blocked_angles: list[str],
    count: int = 3,
) -> list[ClueAngle]:
    """Choose a deterministic set of angles for the next visible attempt cycle."""
    all_angles = list(ClueAngle)
    start_index = (max(attempt_no, 1) - 1) % len(all_angles)
    rotated = all_angles[start_index:] + all_angles[:start_index]
    blocked = {item for item in blocked_angles if item}
    used = {item for item in used_angles if item}

    preferred = [angle for angle in rotated if angle.value not in blocked and angle.value not in used]
    fallback = [angle for angle in rotated if angle.value not in blocked and angle not in preferred]
    selected = (preferred + fallback)[:count]
    if len(selected) < count:
        selected.extend(angle for angle in rotated if angle not in selected)
    return selected[:count]


def parse_cluer_candidates(
    text: str,
    *,
    allowed_angles: list[ClueAngle],
) -> tuple[list[CluerCandidatePayload], str]:
    """Parse strict JSON clue candidates without promoting raw fallback text."""
    allowed_values = {angle.value for angle in allowed_angles}
    try:
        payload = extract_first_json_object(text)
        structured = CluerCandidatesPayload.model_validate(payload)
        filtered: list[CluerCandidatePayload] = []
        seen_angles: set[str] = set()
        for candidate in structured.candidates:
            if candidate.angle not in allowed_values:
                continue
            if candidate.angle in seen_angles:
                continue
            clue_text = candidate.clue.strip()
            if not clue_text:
                continue
            seen_angles.add(candidate.angle)
            filtered.append(
                CluerCandidatePayload(angle=candidate.angle, clue=clue_text)
            )
        if filtered:
            return filtered, "json"
    except (ValueError, ValidationError):
        return [], "parse_failure"

    return [], "structured_payload_rejected" if looks_like_structured_payload(text) else "unstructured_output"


def evaluate_clue_candidates(
    *,
    candidates: list[CluerCandidatePayload],
    validator: LogicalValidator,
    card: CardRecord,
    previous_accepted_clues: list[str],
    previous_rejected_clues: list[str],
    used_angles: list[str],
) -> list[ClueCandidateEvaluation]:
    """Run deterministic validation and scoring over a candidate batch."""
    evaluations: list[ClueCandidateEvaluation] = []
    used = {item for item in used_angles if item}
    for candidate in candidates:
        if looks_like_structured_payload(candidate.clue):
            logical_result = LogicalValidationResult(
                verdict="fail",
                normalized_text=normalize_text(candidate.clue),
                violations=["structured_payload_detected"],
                matched_terms=[],
            )
            evaluations.append(
                ClueCandidateEvaluation(
                    angle=ClueAngle(candidate.angle),
                    clue_text_raw=candidate.clue,
                    clue_text_normalized=logical_result.normalized_text,
                    logical_result=logical_result,
                    score=-1000,
                    warnings=[],
                )
            )
            continue
        logical_result = validator.validate(
            candidate.clue,
            card=card,
            previous_accepted_clues=previous_accepted_clues,
            previous_rejected_clues=previous_rejected_clues,
        )
        evaluations.append(
            ClueCandidateEvaluation(
                angle=ClueAngle(candidate.angle),
                clue_text_raw=candidate.clue,
                clue_text_normalized=logical_result.normalized_text,
                logical_result=logical_result,
                score=_score_candidate(
                    candidate=candidate,
                    logical_result=logical_result,
                    used_angles=used,
                ),
                warnings=[],
            )
        )
    return evaluations


def select_best_candidate(evaluations: list[ClueCandidateEvaluation]) -> ClueCandidateEvaluation | None:
    """Choose the best surviving candidate deterministically."""
    valid = [item for item in evaluations if item.is_valid]
    if not valid:
        return None
    return max(
        valid,
        key=lambda item: (
            item.score,
            -len(item.logical_result.violations),
            item.angle.value,
            item.clue_text_normalized,
        ),
    )


def build_repair_feedback(
    *,
    evaluations: list[ClueCandidateEvaluation],
    allowed_angles: list[ClueAngle],
    blocked_angles: list[ClueAngle],
    llm_result: Any | None = None,
) -> CluerRepairFeedbackPayload:
    """Collapse controller/judge feedback into machine-friendly repair codes."""
    blocked_terms = dedupe_preserve_order(
        [
            term
            for evaluation in evaluations
            for term in evaluation.logical_result.matched_terms
        ]
    )
    reason_codes = dedupe_preserve_order(
        [
            code
            for evaluation in evaluations
            for code in evaluation.logical_result.violations
        ]
    )
    if llm_result is not None:
        for reason in getattr(llm_result, "reasons", []):
            normalized_reason = normalize_text(str(reason))
            if normalized_reason:
                reason_codes.append(normalized_reason.replace(" ", "_"))
    return CluerRepairFeedbackPayload(
        blocked_terms=blocked_terms,
        reason_codes=dedupe_preserve_order(reason_codes),
        blocked_angles=[angle.value for angle in blocked_angles],
        allowed_angles=[angle.value for angle in allowed_angles],
    )


def blocked_angles_from_evaluations(evaluations: list[ClueCandidateEvaluation]) -> list[ClueAngle]:
    """Return angles that should be avoided after this cycle."""
    blocked: list[ClueAngle] = []
    seen: set[ClueAngle] = set()
    for evaluation in evaluations:
        if evaluation.is_valid or evaluation.angle in seen:
            continue
        seen.add(evaluation.angle)
        blocked.append(evaluation.angle)
    return blocked


def _score_candidate(
    *,
    candidate: CluerCandidatePayload,
    logical_result: LogicalValidationResult,
    used_angles: set[str],
) -> int:
    if logical_result.verdict != "pass":
        return -1000 - len(logical_result.violations)
    token_count = len(tokenize(candidate.clue))
    target_tokens = 9
    score = 100 - abs(token_count - target_tokens)
    if candidate.angle not in used_angles:
        score += 10
    if token_count >= 4:
        score += 5
    return score
