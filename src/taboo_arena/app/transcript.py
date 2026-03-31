"""Helpers for rendering a messenger-like transcript from logger events."""

from __future__ import annotations

import json
import re
from typing import Any

from taboo_arena.app.transcript_view_models import (
    BubbleDebugField,
    BubbleDebugSection,
    BubblePublicLine,
    BubbleRawArtifact,
    TranscriptMessage,
)

TranscriptDebugEntry = BubbleDebugField


def merge_transcript_event_sources(
    *,
    history_events: list[dict[str, Any]],
    current_events: list[dict[str, Any]],
    archived_run_ids: list[str] | None = None,
    current_run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Combine archived and current events without replaying an already archived run."""
    combined = list(history_events)
    if not (current_run_id and archived_run_ids and current_run_id in set(archived_run_ids)):
        combined.extend(current_events)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in combined:
        fingerprint = json.dumps(event, sort_keys=True, default=str)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(event)
    return deduped


def latest_round_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the newest round from an event stream."""
    effective_round_ids = _effective_round_ids(events)
    latest_round_id = next((round_id for round_id in reversed(effective_round_ids) if round_id), None)
    if latest_round_id is None:
        return list(events)
    return [
        event
        for event, effective_round_id in zip(events, effective_round_ids, strict=True)
        if effective_round_id == latest_round_id
    ]


def build_transcript_messages(events: list[dict[str, Any]]) -> list[TranscriptMessage]:
    """Project controller events into public dialogue bubbles plus hidden debug details."""
    messages: list[TranscriptMessage | None] = []
    clue_indexes: dict[int, int] = {}
    clue_line_indexes: dict[int, dict[int, int]] = {}
    clue_judge_indexes: dict[int, int] = {}
    guess_indexes: dict[int, int] = {}
    guess_judge_indexes: dict[int, int] = {}
    pending_clue_public_text: dict[tuple[int, int], str] = {}
    active_round_id: str | None = None
    last_clue_key: int | None = None

    for event in events:
        round_id = _event_round_id(event, active_round_id)
        if round_id != active_round_id:
            active_round_id = round_id
            clue_indexes = {}
            clue_line_indexes = {}
            clue_judge_indexes = {}
            guess_indexes = {}
            guess_judge_indexes = {}
            pending_clue_public_text = {}
            last_clue_key = None
            messages.append(
                TranscriptMessage(
                    role="meta",
                    label="Round",
                    public_text=_round_separator_text(event),
                    tone="meta",
                    alignment="center",
                    message_id=f"{round_id}:meta",
                )
            )

        attempt_no = int(event.get("attempt_no", 0) or 0)
        repair_no = int(event.get("clue_repair_no", 0) or 0)
        clue_key = attempt_no
        repair_key = (attempt_no, repair_no if repair_no > 0 else 1)
        event_type = str(event.get("event_type", ""))
        if attempt_no > 0 and event_type.startswith("clue_"):
            last_clue_key = clue_key

        if event_type == "clue_draft_started":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no)
            clue_message.status_label = _cluer_status(event_type, repair_no)
            _append_timeline(clue_message, clue_message.status_label)
            continue

        if event_type == "clue_candidate_cycle_started":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no)
            clue_message.status_label = _cluer_status(event_type, repair_no)
            _append_timeline(clue_message, clue_message.status_label)
            _set_debug_section(
                clue_message,
                "Planning",
                [
                    _debug_field("Allowed angles", _comma_join(event.get("allowed_angles"))),
                    _debug_field("Blocked terms", _comma_join(event.get("blocked_terms"))),
                    _debug_field("Blocked prior clues", _comma_join(event.get("blocked_prior_clues"))),
                    _debug_field("Blocked angles", _comma_join(event.get("blocked_angles"))),
                ],
                summary="Controller planning inputs.",
            )
            continue

        if event_type == "clue_candidates_generated":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no)
            clue_message.status_label = _cluer_status(event_type, repair_no)
            preview_text = _candidate_preview_text(event)
            if preview_text:
                pending_clue_public_text[repair_key] = preview_text
                _set_clue_public_line(
                    clue_message,
                    clue_line_indexes,
                    attempt_no,
                    repair_no,
                    preview_text,
                    is_struck_out=False,
                )
            _append_timeline(clue_message, "drafting candidates")
            _append_timeline(
                clue_message,
                "candidate batch parsed"
                if _optional_text(event.get("clue_candidate_parse_mode")) == "json"
                else f"candidate batch {_optional_text(event.get('clue_candidate_parse_mode')) or 'rejected'}",
            )
            generated_candidates = _string_list(event.get("clue_candidate_clues"))
            _set_debug_section(
                clue_message,
                "Candidates",
                [
                    _debug_field(
                        "Internal clue candidates",
                        _paired_lines(
                            event.get("clue_candidate_angles"),
                            event.get("clue_candidate_clues"),
                        ),
                    )
                ],
                summary=f"{len(generated_candidates)} candidate(s) generated.",
            )
            _set_raw_artifact(
                clue_message,
                "Cluer parse mode",
                _optional_text(event.get("clue_candidate_parse_mode")),
            )
            _set_raw_artifact(
                clue_message,
                "Cluer raw model output",
                _optional_text(event.get("raw_model_output")),
            )
            continue

        if event_type == "clue_candidate_validation_completed":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no)
            clue_message.status_label = _cluer_status(event_type, repair_no)
            _append_timeline(clue_message, "hard filter completed")
            _set_debug_section(
                clue_message,
                "Validation",
                [
                    _debug_field(
                        "Candidate validation",
                        _format_candidate_validation(event.get("candidate_results")),
                    ),
                    _debug_field("Validator version", _optional_text(event.get("validator_version"))),
                ],
                summary="Hard rule filtering over the candidate batch.",
            )
            continue

        if event_type == "clue_candidate_selected":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no)
            clue_message.status_label = _cluer_status(event_type, repair_no)
            _append_timeline(clue_message, "selected candidate")
            _set_debug_section(
                clue_message,
                "Summary",
                [
                    _debug_field("Selected angle", _optional_text(event.get("selected_angle"))),
                ],
                summary="One candidate selected for judge review.",
            )
            _set_debug_section(
                clue_message,
                "Candidates",
                [
                    _debug_field("Internal clue candidates", _line_list(event.get("clue_candidate_clues"))),
                ],
            )
            continue

        if event_type == "clue_draft_generated":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no)
            clue_message.status_label = _cluer_status(event_type, repair_no)
            pending_text = _optional_text(event.get("visible_clue_text"))
            if pending_text:
                pending_clue_public_text[repair_key] = pending_text
                _set_clue_public_line(
                    clue_message,
                    clue_line_indexes,
                    attempt_no,
                    repair_no,
                    pending_text,
                    is_struck_out=False,
                )
                clue_message.public_text = pending_text
                clue_message.pending_public_text = pending_text
                clue_message.is_struck_out = False
                clue_message.status_label = None
                _append_timeline(clue_message, "visible clue candidate prepared")
            else:
                clue_message.pending_public_text = None
                _append_timeline(clue_message, "candidate prepared")
            _set_prompt_metadata(clue_message, event)
            _set_debug_section(
                clue_message,
                "Summary",
                [
                    _debug_field("Selected angle", _optional_text(event.get("selected_angle"))),
                    _debug_field("Selected visible clue", pending_text),
                ],
            )
            _set_debug_section(
                clue_message,
                "Candidates",
                [
                    _debug_field("Internal clue candidates", _line_list(event.get("clue_candidate_clues"))),
                ],
            )
            continue

        if event_type == "clue_internal_retry_requested":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no)
            clue_message.status_label = _cluer_status(event_type, repair_no)
            retry_text = pending_clue_public_text.get(repair_key)
            if retry_text:
                _set_clue_public_line(
                    clue_message,
                    clue_line_indexes,
                    attempt_no,
                    repair_no,
                    retry_text,
                    is_struck_out=True,
                )
            _append_timeline(clue_message, f"hidden repair {repair_no} requested")
            _set_debug_section(
                clue_message,
                "Retries",
                [
                    _debug_field("Reason codes", _comma_join(event.get("reason_codes"))),
                    _debug_field("Blocked terms", _comma_join(event.get("blocked_terms"))),
                    _debug_field("Blocked angles", _comma_join(event.get("blocked_angles"))),
                    _debug_field("Allowed angles", _comma_join(event.get("allowed_angles"))),
                ],
                summary="Hidden repair requested before any public clue is shown.",
            )
            continue

        if event_type == "clue_review_started":
            judge_message = _ensure_clue_judge_message(
                messages,
                clue_judge_indexes,
                round_id,
                attempt_no,
            )
            judge_message.status_label = "checking rules"
            _append_timeline(judge_message, "checking rules")
            continue

        if event_type == "llm_validation_completed":
            existing_clue_message = _lookup_message(messages, clue_indexes, clue_key)
            if existing_clue_message is not None:
                verdict = str(event.get("final_judge_verdict", "")).strip()
                if verdict == "fail":
                    existing_clue_message.pending_public_text = None
                    existing_clue_message.status_label = None
                    existing_clue_message.tone = "rejected"
                    existing_clue_message.is_struck_out = bool(existing_clue_message.public_text.strip())
                    _mark_clue_public_line(
                        existing_clue_message,
                        clue_line_indexes,
                        attempt_no,
                        repair_no,
                        is_struck_out=True,
                    )
                    _append_timeline(existing_clue_message, "selected candidate rejected")
                else:
                    existing_clue_message.public_text = pending_clue_public_text.get(repair_key, "")
                    existing_clue_message.pending_public_text = None
                    existing_clue_message.status_label = None
                    existing_clue_message.tone = "accepted"
                    existing_clue_message.is_struck_out = False
                    _mark_clue_public_line(
                        existing_clue_message,
                        clue_line_indexes,
                        attempt_no,
                        repair_no,
                        is_struck_out=False,
                    )
                    _append_timeline(existing_clue_message, "selected candidate accepted")

            judge_message = _ensure_clue_judge_message(
                messages,
                clue_judge_indexes,
                round_id,
                attempt_no,
            )
            judge_message.status_label = None
            judge_message.public_text = _clue_judge_public_text(event)
            _append_timeline(
                judge_message,
                "approved" if str(event.get("final_judge_verdict", "")).strip() != "fail" else "rejected",
            )
            _set_prompt_metadata(judge_message, event)
            _set_debug_section(
                judge_message,
                "Summary",
                [
                    _debug_field("Final verdict", _optional_text(event.get("final_judge_verdict"))),
                    _debug_field("LLM verdict", _optional_text(event.get("llm_judge_verdict"))),
                    _debug_field(
                        "Judge disagreement",
                        "yes" if bool(event.get("judge_disagreement", False)) else "no",
                    ),
                ],
                summary="Final clue arbitration result.",
            )
            _set_debug_section(
                judge_message,
                "Details",
                [
                    _debug_field("Judge warnings", _comma_join(event.get("llm_judge_warnings"))),
                    _debug_field("Judge reasons", _comma_join(event.get("llm_judge_reasons"))),
                    _debug_field(
                        "Matched surface forms",
                        _comma_join(event.get("llm_judge_suspicious_terms")),
                    ),
                ],
            )
            continue

        if event_type == "clue_accepted":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no)
            clue_message.public_text = (
                _optional_text(event.get("visible_clue_text"))
                or pending_clue_public_text.get(repair_key, "")
            )
            clue_message.pending_public_text = None
            clue_message.status_label = None
            clue_message.tone = "accepted"
            clue_message.is_struck_out = False
            _mark_clue_public_line(
                clue_message,
                clue_line_indexes,
                attempt_no,
                repair_no,
                is_struck_out=False,
            )
            _append_timeline(clue_message, "accepted visible clue")
            _set_debug_section(
                clue_message,
                "Summary",
                [
                    _debug_field("Selected angle", _optional_text(event.get("selected_angle"))),
                    _debug_field("Judge warnings", _comma_join(event.get("judge_warning_flags"))),
                ],
                summary="Final accepted clue shown to the guesser.",
            )
            continue

        if event_type == "clue_repair_requested":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no)
            clue_message.tone = "rejected"
            clue_message.status_label = None
            clue_message.pending_public_text = None
            clue_message.is_struck_out = bool(clue_message.public_text.strip())
            _mark_clue_public_line(
                clue_message,
                clue_line_indexes,
                attempt_no,
                repair_no,
                is_struck_out=True,
            )
            _append_timeline(clue_message, "hidden clue rejected")
            _set_debug_section(
                clue_message,
                "Retries",
                [
                    _debug_field("Hidden repair reasons", _format_hidden_repair_debug(event)),
                    _debug_field("Selected angle", _optional_text(event.get("selected_angle"))),
                ],
                summary="Judge rejected the hidden clue candidate.",
            )

            judge_message = _ensure_clue_judge_message(
                messages,
                clue_judge_indexes,
                round_id,
                attempt_no,
            )
            judge_message.status_label = None
            judge_message.public_text = (
                "Judge failed to return a valid decision."
                if _contains_parse_failure(event.get("llm_judge_reasons"))
                else "Rejected clue."
            )
            _append_timeline(judge_message, "rejected")
            _set_prompt_metadata(judge_message, event)
            _set_debug_section(
                judge_message,
                "Summary",
                [
                    _debug_field("Final verdict", _optional_text(event.get("final_judge_verdict"))),
                    _debug_field("LLM verdict", _optional_text(event.get("llm_judge_verdict"))),
                    _debug_field(
                        "Judge disagreement",
                        "yes" if bool(event.get("judge_disagreement", False)) else "no",
                    ),
                ],
                summary="Rejected clue remains hidden from the public dialogue.",
            )
            _set_debug_section(
                judge_message,
                "Details",
                [
                    _debug_field("Logical violations", _comma_join(event.get("logical_violations"))),
                    _debug_field("Matched terms", _comma_join(event.get("logical_matched_terms"))),
                    _debug_field("Judge reasons", _comma_join(event.get("llm_judge_reasons"))),
                ],
            )
            continue

        if event_type == "guess_started":
            guess_message = _ensure_guess_message(messages, guess_indexes, round_id, attempt_no)
            guess_message.status_label = "forming hypotheses"
            _append_timeline(guess_message, "forming hypotheses")
            continue

        if event_type == "guess_shortlist_generated":
            guess_message = _ensure_guess_message(messages, guess_indexes, round_id, attempt_no)
            guess_message.status_label = "deduping"
            _append_timeline(
                guess_message,
                "shortlist parsed"
                if _optional_text(event.get("guess_parse_mode")) == "json"
                else f"shortlist {_optional_text(event.get('guess_parse_mode')) or 'rejected'}",
            )
            shortlist = _string_list(event.get("guess_shortlist_candidates"))
            _set_debug_section(
                guess_message,
                "Summary",
                [
                    _debug_field("Shortlist size", str(len(shortlist))),
                ],
                summary="Shortlist built before visible guess selection.",
            )
            _set_debug_section(
                guess_message,
                "Shortlist",
                [
                    _debug_field("Guess shortlist", _line_list(event.get("guess_shortlist_candidates"))),
                    _debug_field(
                        "Candidate notes",
                        _format_guess_candidate_notes(
                            event.get("guess_shortlist_candidate_keys"),
                            event.get("guess_shortlist_invalid_reasons"),
                            event.get("guess_shortlist_repeated_against"),
                        ),
                    ),
                ],
            )
            _set_raw_artifact(
                guess_message,
                "Guesser parse mode",
                _optional_text(event.get("guess_parse_mode")),
            )
            _set_raw_artifact(
                guess_message,
                "Guesser raw model output",
                _optional_text(event.get("raw_model_output")),
            )
            continue

        if event_type == "guess_hidden_retry_requested":
            guess_message = _ensure_guess_message(messages, guess_indexes, round_id, attempt_no)
            guess_message.status_label = "forming hypotheses"
            _append_timeline(guess_message, f"hidden retry {int(event.get('guess_hidden_retry_no', 0) or 0)} requested")
            _set_debug_section(
                guess_message,
                "Retries",
                [
                    _debug_field("Hidden retry reasons", _comma_join(event.get("reason_codes"))),
                    _debug_field("Hidden retry count", _optional_text(event.get("guess_hidden_retry_no"))),
                ],
                summary="Controller asked the guesser for another hidden shortlist.",
            )
            continue

        if event_type == "guess_review_started":
            guess_message = _ensure_guess_message(messages, guess_indexes, round_id, attempt_no)
            review_guess_visible = _optional_text(event.get("visible_guess_text"))
            review_guess_raw = _optional_text(event.get("guess_text_raw"))
            if review_guess_visible or review_guess_raw:
                review_guess_text = _guess_public_text(event)
                guess_message.public_text = review_guess_text
                guess_message.status_label = None
                guess_message.tone = "guess"
                _append_timeline(guess_message, "visible guess selected")
                _set_debug_section(
                    guess_message,
                    "Summary",
                    [
                        _debug_field("Selected visible guess", _optional_text(event.get("visible_guess_text"))),
                        _debug_field("Match status", _optional_text(event.get("guess_match_status"))),
                        _debug_field("Match reason", _optional_text(event.get("guess_match_reason"))),
                        _debug_field("Hidden retry count", _optional_text(event.get("guess_hidden_retry_count"))),
                    ],
                    summary="Selected guess submitted for judge verification.",
                )
                _set_debug_section(
                    guess_message,
                    "Shortlist",
                    [
                        _debug_field("Guess shortlist", _line_list(event.get("guess_shortlist_candidates"))),
                    ],
                )
            judge_message = _ensure_guess_judge_message(
                messages,
                guess_judge_indexes,
                round_id,
                attempt_no,
            )
            judge_message.status_label = "verifying guess"
            _append_timeline(judge_message, "verifying guess")
            continue

        if event_type == "guess_validation_completed":
            guess_message = _ensure_guess_message(messages, guess_indexes, round_id, attempt_no)
            completed_guess_visible = _optional_text(event.get("visible_guess_text"))
            completed_guess_raw = _optional_text(event.get("guess_text_raw"))
            if completed_guess_visible or completed_guess_raw:
                completed_guess_text = _guess_public_text(event)
                guess_message.public_text = completed_guess_text
                guess_message.tone = "success" if bool(event.get("final_guess_correct", False)) else "guess"
                guess_message.status_label = None
                _append_timeline(guess_message, "visible guess selected")
                _append_timeline(guess_message, "canonicalization complete")
                _set_debug_section(
                    guess_message,
                    "Summary",
                    [
                        _debug_field("Selected visible guess", _optional_text(event.get("visible_guess_text"))),
                        _debug_field("Match status", _optional_text(event.get("guess_match_status"))),
                        _debug_field("Match reason", _optional_text(event.get("guess_match_reason"))),
                        _debug_field("Hidden retry count", _optional_text(event.get("guess_hidden_retry_count"))),
                    ],
                    summary="Visible guess committed before the round advances.",
                )
                _set_debug_section(
                    guess_message,
                    "Shortlist",
                    [
                        _debug_field("Guess shortlist", _line_list(event.get("guess_shortlist_candidates"))),
                    ],
                )
                _set_debug_section(
                    guess_message,
                    "Canonicalization",
                    [
                        _debug_field(
                            "Canonicalization",
                            _format_canonicalization_debug(event),
                        ),
                    ],
                )
            judge_message = _ensure_guess_judge_message(
                messages,
                guess_judge_indexes,
                round_id,
                attempt_no,
            )
            judge_message.status_label = None
            judge_message.public_text = _guess_judge_public_text(event)
            _append_timeline(judge_message, "final verdict recorded")
            _set_prompt_metadata(judge_message, event)
            _set_debug_section(
                judge_message,
                "Summary",
                [
                    _debug_field("Final guess correct", "yes" if bool(event.get("final_guess_correct", False)) else "no"),
                    _debug_field(
                        "Judge disagreement",
                        "yes" if bool(event.get("judge_disagreement", False)) else "no",
                    ),
                ],
                summary="Visible guess arbitration result.",
            )
            _set_debug_section(
                judge_message,
                "Details",
                [
                    _debug_field("Judge reason codes", _comma_join(event.get("guess_judge_reason_codes"))),
                    _debug_field("Judge warnings", _comma_join(event.get("guess_judge_warnings"))),
                    _debug_field(
                        "Matched surface forms",
                        _comma_join(event.get("guess_judge_matched_surface_forms")),
                    ),
                ],
            )
            continue

        if event_type == "guess_generated":
            guess_message = _ensure_guess_message(messages, guess_indexes, round_id, attempt_no)
            final_correct = _guess_solved(event)
            guess_message.public_text = _guess_public_text(event)
            guess_message.tone = "success" if final_correct else "guess"
            guess_message.status_label = None
            _append_timeline(guess_message, "visible guess selected")
            _append_timeline(guess_message, "canonicalization complete")
            _set_prompt_metadata(guess_message, event)
            _set_debug_section(
                guess_message,
                "Summary",
                [
                    _debug_field("Selected visible guess", _optional_text(event.get("visible_guess_text"))),
                    _debug_field("Match status", _optional_text(event.get("guess_match_status"))),
                    _debug_field("Match reason", _optional_text(event.get("guess_match_reason"))),
                    _debug_field("Hidden retry count", _optional_text(event.get("guess_hidden_retry_count"))),
                ],
                summary="Final visible guess committed to the transcript.",
            )
            _set_debug_section(
                guess_message,
                "Shortlist",
                [
                    _debug_field("Guess shortlist", _line_list(event.get("guess_shortlist_candidates"))),
                ],
            )
            _set_debug_section(
                guess_message,
                "Canonicalization",
                [
                    _debug_field(
                        "Canonicalization",
                        _format_canonicalization_debug(event),
                    ),
                ],
            )
            continue

        if event_type == "round_finished" and str(event.get("terminal_reason", "")).strip() == "clue_not_repaired":
            if last_clue_key is None:
                continue
            final_clue_message = _lookup_message(messages, clue_indexes, last_clue_key)
            if final_clue_message is None:
                continue
            if not final_clue_message.public_lines and not final_clue_message.public_text:
                final_clue_message.public_text = "Cluer failed to produce an acceptable clue."
            final_clue_message.pending_public_text = None
            final_clue_message.status_label = None
            final_clue_message.tone = "rejected"
            final_clue_message.is_struck_out = bool(final_clue_message.public_text.strip()) and not bool(
                final_clue_message.public_lines
            )

    for message in messages:
        if message is None or message.role != "cluer":
            continue
        if message.public_text or not message.pending_public_text:
            continue
        if any(section.title == "Retries" for section in message.debug_sections):
            continue
        message.public_text = message.pending_public_text
        message.pending_public_text = None
        message.status_label = None
        message.tone = "accepted"
        message.is_struck_out = False

    for index, message in enumerate(messages):
        if message is None:
            continue
        if message.public_text.strip() or message.pending_public_text:
            continue
        if not message.status_label:
            continue
        stream_key = _message_stream_key(message)
        next_message = next(
            (
                later
                for later in messages[index + 1 :]
                if later is not None and _message_stream_key(later) == stream_key
            ),
            None,
        )
        if next_message is not None:
            _merge_message_debug(message, next_message)
            message.status_label = None
            message.is_public_turn = False

    return [message for message in messages if message is not None and _should_render_message(message)]


def _ensure_clue_message(
    messages: list[TranscriptMessage | None],
    clue_indexes: dict[int, int],
    round_id: str,
    attempt_no: int,
) -> TranscriptMessage:
    clue_key = attempt_no
    clue_message = _lookup_message(messages, clue_indexes, clue_key)
    if clue_message is not None:
        return clue_message
    clue_indexes[clue_key] = len(messages)
    clue_message = TranscriptMessage(
        role="cluer",
        label=_cluer_label(attempt_no, 1),
        public_text="",
        tone="pending",
        alignment="left",
        message_id=f"{round_id}:clue:{attempt_no}",
    )
    messages.append(clue_message)
    return clue_message


def _ensure_clue_judge_message(
    messages: list[TranscriptMessage | None],
    judge_indexes: dict[int, int],
    round_id: str,
    attempt_no: int,
) -> TranscriptMessage:
    clue_key = attempt_no
    judge_message = _lookup_message(messages, judge_indexes, clue_key)
    if judge_message is not None:
        return judge_message
    judge_indexes[clue_key] = len(messages)
    judge_message = TranscriptMessage(
        role="judge",
        label="Judge",
        public_text="",
        tone="judge",
        alignment="center",
        message_id=f"{round_id}:judge-review:{attempt_no}",
    )
    messages.append(judge_message)
    return judge_message


def _ensure_guess_message(
    messages: list[TranscriptMessage | None],
    guess_indexes: dict[int, int],
    round_id: str,
    attempt_no: int,
) -> TranscriptMessage:
    guess_message = _lookup_message(messages, guess_indexes, attempt_no)
    if guess_message is not None:
        return guess_message
    guess_indexes[attempt_no] = len(messages)
    guess_message = TranscriptMessage(
        role="guesser",
        label=f"Guesser - attempt {attempt_no}",
        public_text="",
        tone="guess",
        alignment="right",
        message_id=f"{round_id}:guess:{attempt_no}",
    )
    messages.append(guess_message)
    return guess_message


def _ensure_guess_judge_message(
    messages: list[TranscriptMessage | None],
    judge_indexes: dict[int, int],
    round_id: str,
    attempt_no: int,
) -> TranscriptMessage:
    judge_message = _lookup_message(messages, judge_indexes, attempt_no)
    if judge_message is not None:
        return judge_message
    judge_indexes[attempt_no] = len(messages)
    judge_message = TranscriptMessage(
        role="judge",
        label="Judge",
        public_text="",
        tone="judge",
        alignment="center",
        message_id=f"{round_id}:judge-guess:{attempt_no}",
    )
    messages.append(judge_message)
    return judge_message


def _lookup_message(
    messages: list[TranscriptMessage | None],
    indexes: dict[Any, int],
    key: Any,
) -> TranscriptMessage | None:
    index = indexes.get(key)
    if index is None or index >= len(messages):
        return None
    return messages[index]


def _set_prompt_metadata(message: TranscriptMessage, event: dict[str, Any]) -> None:
    message.prompt_text = _prompt_text(event)
    message.prompt_model_id = _optional_text(event.get("prompt_model_id"))
    message.prompt_template_id = _prompt_template_id(event)


def _message_stream_key(message: TranscriptMessage) -> str:
    if message.role == "judge":
        if "judge-review:" in message.message_id:
            return "judge-review"
        if "judge-guess:" in message.message_id:
            return "judge-guess"
    return message.role


def _merge_message_debug(source: TranscriptMessage, target: TranscriptMessage) -> None:
    if source.debug_timeline:
        merged_timeline = list(source.debug_timeline)
        for step in target.debug_timeline:
            if not merged_timeline or merged_timeline[-1] != step:
                merged_timeline.append(step)
        target.debug_timeline = merged_timeline

    if source.debug_sections:
        merged_sections: dict[str, BubbleDebugSection] = {
            section.title: BubbleDebugSection(
                title=section.title,
                summary=section.summary,
                fields=list(section.fields),
            )
            for section in source.debug_sections
        }
        for section in target.debug_sections:
            current = merged_sections.get(
                section.title,
                BubbleDebugSection(title=section.title),
            )
            if section.summary:
                current.summary = section.summary
            merged_fields = {field.label: field for field in current.fields}
            for field in section.fields:
                merged_fields[field.label] = field
            current.fields = list(merged_fields.values())
            merged_sections[section.title] = current
        target.debug_sections = list(merged_sections.values())

    if source.raw_artifacts:
        merged_artifacts = {artifact.label: artifact for artifact in source.raw_artifacts}
        for artifact in target.raw_artifacts:
            merged_artifacts[artifact.label] = artifact
        target.raw_artifacts = list(merged_artifacts.values())


def _should_render_message(message: TranscriptMessage) -> bool:
    if message.tone == "meta":
        return True
    if not message.is_public_turn:
        return False
    return bool(message.public_text.strip()) or bool(message.status_label)


def _append_timeline(message: TranscriptMessage, step: str | None) -> None:
    if step is None:
        return
    text = step.strip()
    if not text:
        return
    if message.debug_timeline and message.debug_timeline[-1] == text:
        return
    message.debug_timeline.append(text)


def _set_debug_section(
    message: TranscriptMessage,
    title: str,
    fields: list[BubbleDebugField | None],
    *,
    summary: str | None = None,
) -> None:
    merged = {section.title: section for section in message.debug_sections}
    current = merged.get(title, BubbleDebugSection(title=title))
    if summary is not None:
        current.summary = summary.strip() or None
    merged_fields = {field.label: field for field in current.fields}
    for field in fields:
        if field is None:
            continue
        merged_fields[field.label] = field
    current.fields = list(merged_fields.values())
    merged[title] = current
    message.debug_sections = list(merged.values())


def _set_raw_artifact(message: TranscriptMessage, label: str, value: str | None) -> None:
    artifact = _raw_artifact(label, value)
    if artifact is None:
        return
    merged = {item.label: item for item in message.raw_artifacts}
    merged[artifact.label] = artifact
    message.raw_artifacts = list(merged.values())


def _set_clue_public_line(
    message: TranscriptMessage,
    clue_line_indexes: dict[int, dict[int, int]],
    attempt_no: int,
    repair_no: int,
    text: str,
    *,
    is_struck_out: bool,
) -> None:
    normalized_repair_no = repair_no if repair_no > 0 else 1
    per_attempt = clue_line_indexes.setdefault(attempt_no, {})
    line_index = per_attempt.get(normalized_repair_no)
    line = BubblePublicLine(text=text, is_struck_out=is_struck_out)
    if line_index is None or line_index >= len(message.public_lines):
        per_attempt[normalized_repair_no] = len(message.public_lines)
        message.public_lines.append(line)
    else:
        message.public_lines[line_index] = line
    message.public_text = _last_public_line_text(message)


def _mark_clue_public_line(
    message: TranscriptMessage,
    clue_line_indexes: dict[int, dict[int, int]],
    attempt_no: int,
    repair_no: int,
    *,
    is_struck_out: bool,
) -> None:
    normalized_repair_no = repair_no if repair_no > 0 else 1
    line_index = clue_line_indexes.get(attempt_no, {}).get(normalized_repair_no)
    if line_index is None or line_index >= len(message.public_lines):
        return
    line = message.public_lines[line_index]
    message.public_lines[line_index] = BubblePublicLine(text=line.text, is_struck_out=is_struck_out)
    message.public_text = _last_public_line_text(message)


def _last_public_line_text(message: TranscriptMessage) -> str:
    for line in reversed(message.public_lines):
        if line.text.strip():
            return line.text
    return message.public_text


def _debug_field(label: str, value: str | None) -> BubbleDebugField | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return BubbleDebugField(label=label, value=text)


def _raw_artifact(label: str, value: str | None) -> BubbleRawArtifact | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return BubbleRawArtifact(label=label, value=text)


def _cluer_label(attempt_no: int, repair_no: int) -> str:
    return f"Cluer - attempt {attempt_no}"


def _cluer_status(event_type: str, repair_no: int) -> str:
    if event_type in {"clue_draft_started", "clue_candidate_cycle_started"}:
        return f"repair {repair_no}" if repair_no > 1 else "planning"
    if event_type == "clue_candidates_generated":
        return "drafting candidates"
    if event_type == "clue_candidate_validation_completed":
        return "hard filtering"
    if event_type in {"clue_candidate_selected", "clue_draft_generated"}:
        return "selected"
    if event_type in {"clue_internal_retry_requested", "clue_repair_requested"}:
        return f"repair {repair_no + 1}"
    return "planning"


def _effective_round_ids(events: list[dict[str, Any]]) -> list[str | None]:
    current_round_id: str | None = None
    effective_round_ids: list[str | None] = []
    for event in events:
        explicit_round_id = _optional_text(event.get("round_id"))
        if explicit_round_id:
            current_round_id = explicit_round_id
        effective_round_ids.append(current_round_id)
    return effective_round_ids


def _event_round_id(event: dict[str, Any], active_round_id: str | None) -> str:
    round_id = _optional_text(event.get("round_id"))
    if round_id:
        return round_id
    if active_round_id:
        return active_round_id
    card_id = _optional_text(event.get("card_id"))
    if card_id:
        return f"live:{card_id}"
    run_id = _optional_text(event.get("run_id"))
    if run_id:
        return f"live:{run_id}"
    return "live:current"


def _round_separator_text(event: dict[str, Any]) -> str:
    card_id = str(event.get("card_id", "")).strip()
    target = str(event.get("target", "")).strip()
    if card_id:
        return f"Round • {card_id} • {target}" if target else f"Round • {card_id}"
    round_id = str(event.get("round_id", "")).strip()
    return f"Round • {round_id}" if round_id else "Round"


def _clue_judge_public_text(event: dict[str, Any]) -> str:
    if _contains_parse_failure(event.get("llm_judge_reasons")):
        return "Judge failed to return a valid decision."
    verdict = str(event.get("final_judge_verdict", "")).strip()
    if verdict == "fail":
        return "Rejected clue."
    warnings = _string_list(event.get("llm_judge_warnings"))
    if verdict == "pass_with_warning" or warnings:
        return "Approved with warning."
    return "Approved."


def _guess_judge_public_text(event: dict[str, Any]) -> str:
    if _contains_parse_failure(event.get("guess_judge_reason_codes")):
        return "Judge failed to return a valid decision."
    if bool(event.get("final_guess_correct", False)):
        return "Correct guess confirmed."
    return "Incorrect guess confirmed."


def _guess_public_text(event: dict[str, Any]) -> str:
    guess_text = _optional_text(event.get("visible_guess_text"))
    if not guess_text:
        return "Guesser failed to produce a usable guess."
    return guess_text


def _format_hidden_repair_debug(event: dict[str, Any]) -> str | None:
    parts: list[str] = []
    reason_codes = _comma_join(event.get("logical_violations")) or _comma_join(event.get("llm_judge_reasons"))
    if reason_codes:
        parts.append(reason_codes)
    matched_terms = _comma_join(event.get("logical_matched_terms"))
    if matched_terms:
        parts.append(f"matched: {matched_terms}")
    return " | ".join(parts) or None


def _candidate_preview_text(event: dict[str, Any]) -> str | None:
    visible_text = _optional_text(event.get("visible_clue_text"))
    if visible_text:
        return visible_text
    parsed_candidates = _string_list(event.get("clue_candidate_clues"))
    if parsed_candidates:
        return parsed_candidates[0]
    raw_output = _optional_text(event.get("raw_model_output"))
    if not raw_output:
        return None
    matches = re.findall(r'"clue"\s*:\s*"([^"]+)"', raw_output)
    for match in matches:
        text = match.strip()
        if text:
            return text
    return None


def _format_candidate_validation(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    lines: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        angle = _optional_text(item.get("angle")) or "candidate"
        verdict = _optional_text(item.get("logical_verdict")) or "unknown"
        violations = _comma_join(item.get("logical_violations"))
        line = f"{angle}: {verdict}"
        if violations:
            line += f" [{violations}]"
        lines.append(line)
    return "\n".join(lines) or None


def _format_guess_candidate_notes(
    candidate_keys: Any,
    invalid_reasons: Any,
    repeated_against: Any,
) -> str | None:
    if not isinstance(candidate_keys, list):
        return None
    invalid_items = invalid_reasons if isinstance(invalid_reasons, list) else []
    repeated_items = repeated_against if isinstance(repeated_against, list) else []
    lines: list[str] = []
    for index, keys in enumerate(candidate_keys):
        key_text = _comma_join(keys)
        invalid_reason = ""
        if index < len(invalid_items):
            invalid_reason = _optional_text(invalid_items[index]) or ""
        repeated_text = ""
        if index < len(repeated_items):
            repeated_text = _comma_join(repeated_items[index]) or ""
        line = key_text or f"candidate {index + 1}"
        if invalid_reason:
            line += f" -> {invalid_reason}"
        if repeated_text:
            line += f" ({repeated_text})"
        lines.append(line)
    return "\n".join(lines) or None


def _format_canonicalization_debug(event: dict[str, Any]) -> str | None:
    parts: list[str] = []
    match_status = _optional_text(event.get("guess_match_status"))
    if match_status:
        parts.append(f"status: {match_status}")
    match_reason = _optional_text(event.get("guess_match_reason"))
    if match_reason:
        parts.append(f"reason: {match_reason}")
    warnings = _comma_join(event.get("guess_match_warnings"))
    if warnings:
        parts.append(f"warnings: {warnings}")
    candidates = _comma_join(event.get("guess_match_candidates"))
    if candidates:
        parts.append(f"candidates: {candidates}")
    disagreement = event.get("guess_judge_disagreement")
    if disagreement is not None:
        parts.append(f"judge disagreement: {'yes' if bool(disagreement) else 'no'}")
    return "\n".join(parts) or None


def _guess_solved(event: dict[str, Any]) -> bool:
    if "final_guess_correct" in event:
        return bool(event.get("final_guess_correct"))
    return str(event.get("state", "")).strip() == "round_finished"


def _paired_lines(left: Any, right: Any) -> str | None:
    if not isinstance(left, list) or not isinstance(right, list):
        return None
    lines: list[str] = []
    for item_left, item_right in zip(left, right, strict=False):
        left_text = _optional_text(item_left)
        right_text = _optional_text(item_right)
        if left_text and right_text:
            lines.append(f"{left_text}: {right_text}")
        elif right_text:
            lines.append(right_text)
    return "\n".join(lines) or None


def _line_list(value: Any) -> str | None:
    items = _string_list(value)
    return "\n".join(items) or None


def _comma_join(value: Any) -> str | None:
    items = _string_list(value)
    return ", ".join(items) or None


def _contains_parse_failure(value: Any) -> bool:
    return any("judge_output_parse_error" in item for item in _string_list(value))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _prompt_text(event: dict[str, Any]) -> str | None:
    prompt = _optional_text(event.get("prompt_text"))
    return prompt or None


def _prompt_template_id(event: dict[str, Any]) -> str | None:
    prompt_template_id = _optional_text(event.get("prompt_template_id"))
    if prompt_template_id:
        return prompt_template_id
    return _optional_text(event.get("prompt_trace_template_id"))


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
