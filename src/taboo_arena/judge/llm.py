"""LLM judge orchestration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.models.registry import ModelEntry
from taboo_arena.prompts.tasks import build_judge_messages
from taboo_arena.utils.json_utils import extract_first_json_object


class LLMJudgeResult(BaseModel):
    """Structured output from the LLM judge."""

    verdict: str
    reasons: list[str] = Field(default_factory=list)
    suspicious_terms: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""
    judge_version: str = "llm_judge_v1"


class NormalizedLLMJudge:
    """Run the LLM judge and normalize the response into a strict schema."""

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
        """Evaluate a clue using a loaded text generation model."""
        messages = build_judge_messages(
            card=card,
            clue_draft=clue_draft,
            accepted_clues=accepted_clues,
            wrong_guesses=wrong_guesses,
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
            result = LLMJudgeResult.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            result = LLMJudgeResult(
                verdict="uncertain",
                reasons=[f"judge_output_parse_error: {exc}"],
                suspicious_terms=[],
                confidence=0.0,
                summary="Judge response could not be parsed as strict JSON.",
                judge_version="llm_judge_v1",
            )
        return result, response
