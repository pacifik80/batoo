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
    last_rejection_feedback: str | None = None,
) -> list[PromptMessage]:
    """Build prompt messages for the cluer role."""
    return render_prompt_messages(
        "cluer",
        {
            "target": card.target,
            "forbidden_words_csv": ", ".join(card.taboo_hard),
            "aliases_csv": ", ".join(card.aliases or ["<none>"]),
            "previous_accepted_clues_json": _json_list(accepted_clues or ["<none>"]),
            "previous_rejected_clues_json": _json_list(rejected_clues or ["<none>"]),
            "previous_wrong_guesses_json": _json_list(wrong_guesses or ["<none>"]),
            "attempt_no": attempt_no,
            "repair_no": repair_no,
            "last_rejection_feedback": last_rejection_feedback or "<none>",
        },
    )


def build_guesser_messages(
    card: CardRecord,
    current_clue: str,
    accepted_clues: list[str],
    wrong_guesses: list[str],
    attempt_no: int,
) -> list[PromptMessage]:
    """Build prompt messages for the guesser role."""
    prior_clues = accepted_clues[:-1] if accepted_clues else []
    return render_prompt_messages(
        "guesser",
        {
            "current_clue": current_clue,
            "prior_accepted_clues_json": _json_list(prior_clues or ["<none>"]),
            "prior_wrong_guesses_json": _json_list(wrong_guesses or ["<none>"]),
            "attempt_no": attempt_no,
            "target": card.target,
        },
    )


def build_judge_messages(
    card: CardRecord,
    clue_draft: str,
    accepted_clues: list[str],
    wrong_guesses: list[str],
    attempt_no: int,
) -> list[PromptMessage]:
    """Build prompt messages for the LLM judge."""
    return render_prompt_messages(
        "judge",
        {
            "target": card.target,
            "forbidden_words_json": _json_list(card.taboo_hard),
            "aliases_json": _json_list(card.aliases or []),
            "clue_draft": clue_draft,
            "previous_accepted_clues_json": _json_list(accepted_clues or []),
            "previous_wrong_guesses_json": _json_list(wrong_guesses or []),
            "attempt_no": attempt_no,
            "output_schema_json": (
                '{"verdict":"pass|fail|uncertain","reasons":["string"],'
                '"suspicious_terms":["string"],"confidence":0.0,"summary":"string","judge_version":"v1"}'
            ),
        },
    )


def _json_list(value: list[str]) -> str:
    return json.dumps(value)
