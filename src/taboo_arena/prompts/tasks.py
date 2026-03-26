"""Prompt builders for the three model roles."""

from __future__ import annotations

import json

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.prompts.store import render_prompt_messages
from taboo_arena.prompts.templates import PromptMessage


def build_cluer_messages(
    card: CardRecord,
    accepted_clues: list[str],
    rejected_clues: list[str],
    wrong_guesses: list[str],
    attempt_no: int,
    repair_no: int,
    allowed_angles: list[str],
    blocked_terms: list[str],
    blocked_prior_clues: list[str],
    blocked_angles: list[str],
    repair_feedback_json: str | None = None,
) -> list[PromptMessage]:
    """Build prompt messages for structured clue-candidate generation."""
    prompt_id = "cluer_repair" if repair_feedback_json else "cluer_candidates"
    return render_prompt_messages(
        prompt_id,
        {
            "target": card.target,
            "forbidden_words_json": _json_list(card.taboo_hard),
            "allowed_angles_json": _json_list(allowed_angles),
            "blocked_terms_json": _json_list(blocked_terms),
            "blocked_prior_clues_json": _json_list(blocked_prior_clues),
            "blocked_angles_json": _json_list(blocked_angles),
            "previous_accepted_clues_json": _json_list(accepted_clues),
            "previous_rejected_clues_json": _json_list(rejected_clues),
            "previous_wrong_guesses_json": _json_list(wrong_guesses),
            "attempt_no": attempt_no,
            "repair_no": repair_no,
            "repair_feedback_json": repair_feedback_json or "{}",
            "output_schema_json": (
                '{"candidates":['
                '{"angle":"type|use|context|effect|part_whole|historical_association","clue":"string"}'
                "]}"
            ),
        },
    )


def build_guesser_messages(
    card: CardRecord,
    current_clue: str,
    accepted_clues: list[str],
    wrong_guesses: list[str],
    attempt_no: int,
) -> list[PromptMessage]:
    """Build prompt messages for shortlist-style guess generation."""
    prior_clues = accepted_clues[:-1] if accepted_clues else []
    return render_prompt_messages(
        "guesser_candidates",
        {
            "current_clue": current_clue,
            "prior_accepted_clues_json": _json_list(prior_clues or ["<none>"]),
            "prior_wrong_guesses_json": _json_list(wrong_guesses or ["<none>"]),
            "attempt_no": attempt_no,
            "output_schema_json": '{"guesses":["string","string","string"]}',
        },
    )


def build_clue_judge_messages(
    card: CardRecord,
    clue_draft: str,
    accepted_clues: list[str],
    rejected_clues: list[str],
    attempt_no: int,
) -> list[PromptMessage]:
    """Build prompt messages for clue-rule arbitration."""
    return render_prompt_messages(
        "judge_clue",
        {
            "target": card.target,
            "forbidden_words_json": _json_list(card.taboo_hard),
            "clue_draft": clue_draft,
            "previous_accepted_clues_json": _json_list(accepted_clues or []),
            "previous_rejected_clues_json": _json_list(rejected_clues or []),
            "attempt_no": attempt_no,
            "output_schema_json": (
                '{"allow":true,"block_reason_codes":["string"],'
                '"warnings":["string"],"matched_surface_forms":["string"],'
                '"judge_version":"clue_judge_v1"}'
            ),
        },
    )


def build_guess_judge_messages(
    card: CardRecord,
    guess_text: str,
    attempt_no: int,
    match_status: str,
    match_reason: str,
    candidate_spans: list[str],
    warnings: list[str],
) -> list[PromptMessage]:
    """Build prompt messages for final visible-guess arbitration."""
    return render_prompt_messages(
        "judge_guess",
        {
            "target": card.target,
            "guess_text": guess_text,
            "attempt_no": attempt_no,
            "deterministic_match_status": match_status,
            "deterministic_match_reason": match_reason,
            "candidate_spans_json": _json_list(candidate_spans),
            "deterministic_warnings_json": _json_list(warnings),
            "output_schema_json": (
                '{"correct":false,"reason_codes":["string"],'
                '"warnings":["string"],"matched_surface_forms":["string"],'
                '"judge_version":"guess_judge_v1"}'
            ),
        },
    )


def build_judge_messages(
    card: CardRecord,
    clue_draft: str,
    accepted_clues: list[str],
    wrong_guesses: list[str],
    attempt_no: int,
) -> list[PromptMessage]:
    """Compatibility wrapper for the clue-judge prompt builder."""
    return build_clue_judge_messages(
        card=card,
        clue_draft=clue_draft,
        accepted_clues=accepted_clues,
        rejected_clues=[],
        attempt_no=attempt_no,
    )


def _json_list(value: list[str]) -> str:
    return json.dumps(value)
