"""LLM judge orchestration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.models.registry import ModelEntry
from taboo_arena.prompts.schemas import ClueJudgePayload, GuessJudgePayload
from taboo_arena.prompts.tasks import (
    build_clue_judge_messages,
    build_guess_judge_messages,
)
from taboo_arena.utils.json_utils import extract_first_json_object
from taboo_arena.utils.normalization import dedupe_preserve_order


class LLMJudgeResult(BaseModel):
    """Structured output from the LLM judge."""

    verdict: str
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suspicious_terms: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""
    judge_version: str = "llm_judge_v1"


class GuessJudgeResult(BaseModel):
    """Structured output from the visible-guess judge."""

    correct: bool
    reason_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    matched_surface_forms: list[str] = Field(default_factory=list)
    judge_version: str = "guess_judge_v1"


class NormalizedLLMJudge:
    """Run the LLM judge and normalize the response into a strict schema."""

    def evaluate_clue(
        self,
        *,
        model_manager: Any,
        model_entry: ModelEntry,
        card: CardRecord,
        clue_draft: str,
        accepted_clues: list[str],
        rejected_clues: list[str],
        attempt_no: int,
        generation_params: Any,
        runtime_policy: str,
        device_preference: str,
    ) -> tuple[LLMJudgeResult, Any]:
        """Evaluate a final clue candidate using the clue-judge prompt."""
        messages = build_clue_judge_messages(
            card=card,
            clue_draft=clue_draft,
            accepted_clues=accepted_clues,
            rejected_clues=rejected_clues,
            attempt_no=attempt_no,
        )
        response = model_manager.generate(
            model_entry=model_entry,
            messages=messages,
            generation_params=generation_params,
            runtime_policy=runtime_policy,
            device_preference=device_preference,
            trace_role="judge",
        )
        try:
            payload = extract_first_json_object(response.text)
            structured = ClueJudgePayload.model_validate(payload)
            block_reason_codes, warnings = _normalize_clue_judge_codes(
                structured.block_reason_codes,
                structured.warnings,
            )
            result = LLMJudgeResult(
                verdict="fail" if block_reason_codes else "pass",
                reasons=block_reason_codes,
                warnings=warnings,
                suspicious_terms=list(structured.matched_surface_forms),
                confidence=1.0,
                summary="; ".join([*block_reason_codes, *warnings]),
                judge_version=structured.judge_version,
            )
        except (ValueError, ValidationError) as exc:
            result = LLMJudgeResult(
                verdict="uncertain",
                reasons=[f"judge_output_parse_error: {exc}"],
                warnings=["judge_output_parse_error"],
                suspicious_terms=[],
                confidence=0.0,
                summary="Judge response could not be parsed as strict JSON.",
                judge_version="clue_judge_v1",
            )
        return result, response

    def evaluate_guess(
        self,
        *,
        model_manager: Any,
        model_entry: ModelEntry,
        card: CardRecord,
        guess_text: str,
        attempt_no: int,
        match_status: str,
        match_reason: str,
        candidate_spans: list[str],
        warnings: list[str],
        generation_params: Any,
        runtime_policy: str,
        device_preference: str,
    ) -> tuple[GuessJudgeResult, Any]:
        """Evaluate the final visible guess using the guess-judge prompt."""
        messages = build_guess_judge_messages(
            card=card,
            guess_text=guess_text,
            attempt_no=attempt_no,
            match_status=match_status,
            match_reason=match_reason,
            candidate_spans=candidate_spans,
            warnings=warnings,
        )
        response = model_manager.generate(
            model_entry=model_entry,
            messages=messages,
            generation_params=generation_params,
            runtime_policy=runtime_policy,
            device_preference=device_preference,
            trace_role="judge",
        )
        try:
            payload = extract_first_json_object(response.text)
            result = GuessJudgeResult.model_validate(GuessJudgePayload.model_validate(payload).model_dump())
        except (ValueError, ValidationError) as exc:
            result = GuessJudgeResult(
                correct=False,
                reason_codes=[f"judge_output_parse_error: {exc}"],
                warnings=[],
                matched_surface_forms=[],
                judge_version="guess_judge_v1",
            )
        return result, response

    def evaluate(
        self,
        *,
        model_manager: Any,
        model_entry: ModelEntry,
        card: CardRecord,
        clue_draft: str,
        accepted_clues: list[str],
        wrong_guesses: list[str],
        attempt_no: int,
        generation_params: Any,
        runtime_policy: str,
        device_preference: str,
    ) -> tuple[LLMJudgeResult, Any]:
        """Compatibility wrapper for the clue-judge evaluation path."""
        return self.evaluate_clue(
            model_manager=model_manager,
            model_entry=model_entry,
            card=card,
            clue_draft=clue_draft,
            accepted_clues=accepted_clues,
            rejected_clues=[],
            attempt_no=attempt_no,
            generation_params=generation_params,
            runtime_policy=runtime_policy,
            device_preference=device_preference,
        )


def _normalize_clue_judge_codes(
    block_reason_codes: list[str],
    warnings: list[str],
) -> tuple[list[str], list[str]]:
    """Demote warning-only analytic codes so they never hard-block clue acceptance."""
    warning_only_codes = {"near_explicit_paraphrase"}
    normalized_blocks: list[str] = []
    normalized_warnings: list[str] = [item for item in warnings if item]
    for reason in block_reason_codes:
        code = str(reason).strip()
        if not code:
            continue
        if code in warning_only_codes:
            normalized_warnings.append(code)
            continue
        normalized_blocks.append(code)
    return (
        dedupe_preserve_order(normalized_blocks),
        dedupe_preserve_order(normalized_warnings),
    )
