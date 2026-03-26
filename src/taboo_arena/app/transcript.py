"""Helpers for rendering a messenger-like transcript from logger events."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

TranscriptTone = Literal["pending", "rejected", "accepted", "judge", "guess", "success", "meta"]
TranscriptAlignment = Literal["left", "center", "right"]


@dataclass(slots=True)
class TranscriptDebugEntry:
    """One expandable debug detail inside a transcript bubble."""

    label: str
    value: str


@dataclass(slots=True)
class TranscriptMessage:
    """One visible message bubble in the transcript panel."""

    role: str
    label: str
    text: str
    tone: TranscriptTone
    alignment: TranscriptAlignment
    message_id: str = ""
    status_text: str | None = None
    debug_entries: list[TranscriptDebugEntry] = field(default_factory=list)
    prompt_text: str | None = None
    prompt_model_id: str | None = None
    prompt_template_id: str | None = None


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
    """Build transcript bubbles across session rounds in chronological order."""
    messages: list[TranscriptMessage | None] = []
    clue_indexes: dict[tuple[int, int], int] = {}
    clue_judge_indexes: dict[tuple[int, int], int] = {}
    guess_indexes: dict[int, int] = {}
    guess_judge_indexes: dict[int, int] = {}
    active_round_id: str | None = None

    for event in events:
        round_id = _event_round_id(event, active_round_id)
        if round_id != active_round_id:
            active_round_id = round_id
            clue_indexes = {}
            clue_judge_indexes = {}
            guess_indexes = {}
            guess_judge_indexes = {}
            messages.append(
                TranscriptMessage(
                    role="meta",
                    label="Round",
                    text=_round_separator_text(event),
                    tone="meta",
                    alignment="center",
                    message_id=f"{round_id}:meta",
                )
            )

        attempt_no = int(event.get("attempt_no", 0) or 0)
        repair_no = int(event.get("clue_repair_no", 0) or 0)
        clue_key = (attempt_no, repair_no)
        event_type = str(event.get("event_type", ""))

        if event_type == "clue_draft_started":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no, repair_no)
            clue_message.status_text = _cluer_status(event_type, repair_no)
            continue

        if event_type == "clue_candidate_cycle_started":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no, repair_no)
            clue_message.status_text = _cluer_status(event_type, repair_no)
            _merge_debug_entries(
                clue_message,
                [
                    _debug_entry("Allowed angles", _comma_join(event.get("allowed_angles"))),
                    _debug_entry("Blocked terms", _comma_join(event.get("blocked_terms"))),
                    _debug_entry("Blocked prior clues", _comma_join(event.get("blocked_prior_clues"))),
                    _debug_entry("Blocked angles", _comma_join(event.get("blocked_angles"))),
                ],
            )
            continue

        if event_type == "clue_candidates_generated":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no, repair_no)
            clue_message.status_text = _cluer_status(event_type, repair_no)
            _merge_debug_entries(
                clue_message,
                [
                    _debug_entry(
                        "Internal clue candidates",
                        _paired_lines(
                            event.get("clue_candidate_angles"),
                            event.get("clue_candidate_clues"),
                        ),
                    )
                ],
            )
            continue

        if event_type == "clue_candidate_validation_completed":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no, repair_no)
            clue_message.status_text = _cluer_status(event_type, repair_no)
            _merge_debug_entries(
                clue_message,
                [
                    _debug_entry(
                        "Candidate validation",
                        _format_candidate_validation(event.get("candidate_results")),
                    )
                ],
            )
            continue

        if event_type == "clue_candidate_selected":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no, repair_no)
            clue_message.status_text = _cluer_status(event_type, repair_no)
            _merge_debug_entries(
                clue_message,
                [
                    _debug_entry("Selected angle", _optional_text(event.get("selected_angle"))),
                    _debug_entry("Internal clue candidates", _line_list(event.get("clue_candidate_clues"))),
                ],
            )
            continue

        if event_type == "clue_draft_generated":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no, repair_no)
            clue_message.status_text = _cluer_status(event_type, repair_no)
            clue_message.text = str(event.get("clue_text_raw", "")).strip() or "(empty clue)"
            _set_prompt_metadata(clue_message, event)
            _merge_debug_entries(
                clue_message,
                [
                    _debug_entry("Selected angle", _optional_text(event.get("selected_angle"))),
                    _debug_entry("Internal clue candidates", _line_list(event.get("clue_candidate_clues"))),
                ],
            )
            continue

        if event_type == "clue_internal_retry_requested":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no, repair_no)
            clue_message.status_text = _cluer_status(event_type, repair_no)
            clue_message.tone = "rejected"
            _merge_debug_entries(
                clue_message,
                [
                    _debug_entry("Hidden repair reasons", _comma_join(event.get("reason_codes"))),
                    _debug_entry("Blocked terms", _comma_join(event.get("blocked_terms"))),
                    _debug_entry("Blocked angles", _comma_join(event.get("blocked_angles"))),
                ],
            )
            continue

        if event_type == "clue_review_started":
            judge_message = _ensure_clue_judge_message(
                messages,
                clue_judge_indexes,
                round_id,
                attempt_no,
                repair_no,
            )
            judge_message.status_text = "verifying final clue"
            continue

        if event_type == "llm_validation_completed":
            existing_clue_message = _lookup_message(messages, clue_indexes, clue_key)
            if existing_clue_message is not None:
                verdict = str(event.get("final_judge_verdict", "")).strip()
                existing_clue_message.tone = "rejected" if verdict == "fail" else "accepted"

            judge_message = _ensure_clue_judge_message(
                messages,
                clue_judge_indexes,
                round_id,
                attempt_no,
                repair_no,
            )
            verdict = str(event.get("final_judge_verdict", "")).strip()
            judge_message.status_text = "accepted" if verdict != "fail" else "rejected"
            judge_message.text = _format_clue_judge_text(event)
            _set_prompt_metadata(judge_message, event)
            _merge_debug_entries(
                judge_message,
                [
                    _debug_entry("Judge warnings", _comma_join(event.get("llm_judge_warnings"))),
                    _debug_entry("Judge reasons", _comma_join(event.get("llm_judge_reasons"))),
                    _debug_entry(
                        "Matched surface forms",
                        _comma_join(event.get("llm_judge_suspicious_terms")),
                    ),
                ],
            )
            continue

        if event_type == "clue_repair_requested":
            clue_message = _ensure_clue_message(messages, clue_indexes, round_id, attempt_no, repair_no)
            clue_message.tone = "rejected"
            clue_message.status_text = _cluer_status(event_type, repair_no)
            _merge_debug_entries(
                clue_message,
                [
                    _debug_entry("Hidden repair reasons", _format_hidden_repair_debug(event)),
                    _debug_entry("Selected angle", _optional_text(event.get("selected_angle"))),
                ],
            )

            judge_message = _ensure_clue_judge_message(
                messages,
                clue_judge_indexes,
                round_id,
                attempt_no,
                repair_no,
            )
            judge_message.status_text = "rejected"
            judge_message.text = _format_rejection_reason(event)
            _set_prompt_metadata(judge_message, event)
            _merge_debug_entries(
                judge_message,
                [
                    _debug_entry("Logical violations", _comma_join(event.get("logical_violations"))),
                    _debug_entry("Matched terms", _comma_join(event.get("logical_matched_terms"))),
                    _debug_entry("Judge reasons", _comma_join(event.get("llm_judge_reasons"))),
                ],
            )
            continue

        if event_type == "guess_started":
            guess_message = _ensure_guess_message(messages, guess_indexes, round_id, attempt_no)
            guess_message.status_text = "forming hypotheses"
            continue

        if event_type == "guess_shortlist_generated":
            guess_message = _ensure_guess_message(messages, guess_indexes, round_id, attempt_no)
            guess_message.status_text = "deduping"
            _merge_debug_entries(
                guess_message,
                [
                    _debug_entry("Guess shortlist", _line_list(event.get("guess_shortlist_candidates"))),
                    _debug_entry(
                        "Candidate notes",
                        _format_guess_candidate_notes(
                            event.get("guess_shortlist_candidate_keys"),
                            event.get("guess_shortlist_invalid_reasons"),
                            event.get("guess_shortlist_repeated_against"),
                        ),
                    ),
                ],
            )
            continue

        if event_type == "guess_hidden_retry_requested":
            guess_message = _ensure_guess_message(messages, guess_indexes, round_id, attempt_no)
            guess_message.status_text = "forming hypotheses"
            _merge_debug_entries(
                guess_message,
                [
                    _debug_entry("Hidden retry reasons", _comma_join(event.get("reason_codes"))),
                ],
            )
            continue

        if event_type == "guess_review_started":
            judge_message = _ensure_guess_judge_message(
                messages,
                guess_judge_indexes,
                round_id,
                attempt_no,
            )
            judge_message.status_text = "verifying guess"
            continue

        if event_type == "guess_validation_completed":
            judge_message = _ensure_guess_judge_message(
                messages,
                guess_judge_indexes,
                round_id,
                attempt_no,
            )
            final_correct = bool(event.get("final_guess_correct", False))
            judge_message.status_text = "accepted" if final_correct else "rejected"
            judge_message.text = "Correct guess confirmed." if final_correct else "Incorrect guess confirmed."
            _set_prompt_metadata(judge_message, event)
            _merge_debug_entries(
                judge_message,
                [
                    _debug_entry("Judge reason codes", _comma_join(event.get("guess_judge_reason_codes"))),
                    _debug_entry("Judge warnings", _comma_join(event.get("guess_judge_warnings"))),
                    _debug_entry(
                        "Matched surface forms",
                        _comma_join(event.get("guess_judge_matched_surface_forms")),
                    ),
                    _debug_entry(
                        "Judge disagreement",
                        "yes" if bool(event.get("judge_disagreement", False)) else "no",
                    ),
                ],
            )
            continue

        if event_type == "guess_generated":
            guess_message = _ensure_guess_message(messages, guess_indexes, round_id, attempt_no)
            final_correct = _guess_solved(event)
            guess_message.text = str(event.get("guess_text_raw", "")).strip() or "(empty guess)"
            guess_message.tone = "success" if final_correct else "guess"
            guess_message.status_text = "finalizing guess"
            _set_prompt_metadata(guess_message, event)
            _merge_debug_entries(
                guess_message,
                [
                    _debug_entry("Guess shortlist", _line_list(event.get("guess_shortlist_candidates"))),
                    _debug_entry(
                        "Canonicalization",
                        _format_canonicalization_debug(event),
                    ),
                ],
            )
            continue

    return [message for message in messages if message is not None]


def _ensure_clue_message(
    messages: list[TranscriptMessage | None],
    clue_indexes: dict[tuple[int, int], int],
    round_id: str,
    attempt_no: int,
    repair_no: int,
) -> TranscriptMessage:
    clue_key = (attempt_no, repair_no)
    clue_message = _lookup_message(messages, clue_indexes, clue_key)
    if clue_message is not None:
        return clue_message
    clue_indexes[clue_key] = len(messages)
    clue_message = TranscriptMessage(
        role="cluer",
        label=_cluer_label(attempt_no, repair_no),
        text="",
        tone="pending",
        alignment="left",
        message_id=f"{round_id}:clue:{attempt_no}:{repair_no}",
    )
    messages.append(clue_message)
    return clue_message


def _ensure_clue_judge_message(
    messages: list[TranscriptMessage | None],
    judge_indexes: dict[tuple[int, int], int],
    round_id: str,
    attempt_no: int,
    repair_no: int,
) -> TranscriptMessage:
    clue_key = (attempt_no, repair_no)
    judge_message = _lookup_message(messages, judge_indexes, clue_key)
    if judge_message is not None:
        return judge_message
    judge_indexes[clue_key] = len(messages)
    judge_message = TranscriptMessage(
        role="judge",
        label="Judge",
        text="",
        tone="judge",
        alignment="center",
        message_id=f"{round_id}:judge-review:{attempt_no}:{repair_no}",
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
        text="",
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
        text="",
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


def _merge_debug_entries(
    message: TranscriptMessage,
    entries: list[TranscriptDebugEntry | None],
) -> None:
    merged = {entry.label: entry for entry in message.debug_entries}
    for entry in entries:
        if entry is None:
            continue
        merged[entry.label] = entry
    message.debug_entries = list(merged.values())


def _debug_entry(label: str, value: str | None) -> TranscriptDebugEntry | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return TranscriptDebugEntry(label=label, value=text)


def _cluer_label(attempt_no: int, repair_no: int) -> str:
    if repair_no <= 1:
        return f"Cluer - attempt {attempt_no}"
    return f"Cluer - repair {repair_no}"


def _cluer_status(event_type: str, repair_no: int) -> str:
    if event_type in {"clue_draft_started", "clue_candidate_cycle_started"}:
        return "repair" if repair_no > 1 else "planning"
    if event_type == "clue_candidates_generated":
        return "drafting candidates"
    if event_type == "clue_candidate_validation_completed":
        return "hard filtering"
    if event_type in {"clue_candidate_selected", "clue_draft_generated"}:
        return "selected"
    if event_type in {"clue_internal_retry_requested", "clue_repair_requested"}:
        return "repair"
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
    if card_id:
        return f"Round • {card_id}"
    round_id = str(event.get("round_id", "")).strip()
    return f"Round • {round_id}" if round_id else "Round"


def _format_rejection_reason(event: dict[str, Any]) -> str:
    logical_violations = _string_list(event.get("logical_violations"))
    matched_terms = _string_list(event.get("logical_matched_terms"))
    llm_verdict = str(event.get("llm_judge_verdict", "")).strip()
    llm_reasons = _string_list(event.get("llm_judge_reasons"))
    judge_disagreement = bool(event.get("judge_disagreement", False))
    parts: list[str] = ["Rejected clue."]
    if logical_violations:
        parts.append(f"Rules: {', '.join(logical_violations)}.")
    if matched_terms:
        parts.append(f"Matched terms: {', '.join(matched_terms)}.")
    if judge_disagreement and llm_verdict:
        parts.append(
            f"Layer disagreement: deterministic validator rejected it while the LLM judge returned {llm_verdict}."
        )
    if llm_reasons:
        parts.append(f"Judge: {'; '.join(llm_reasons)}.")
    return " ".join(parts)


def _format_clue_judge_text(event: dict[str, Any]) -> str:
    verdict = str(event.get("final_judge_verdict", "")).strip()
    if verdict == "fail":
        return "Rejected clue."
    warnings = _string_list(event.get("llm_judge_warnings")) or _string_list(event.get("llm_judge_reasons"))
    if verdict == "pass_with_warning" or warnings:
        return _join_reason_parts(["Approved with warning."], warnings)
    return "Approved."


def _format_hidden_repair_debug(event: dict[str, Any]) -> str | None:
    parts: list[str] = []
    reason_codes = _comma_join(event.get("logical_violations")) or _comma_join(event.get("llm_judge_reasons"))
    if reason_codes:
        parts.append(reason_codes)
    matched_terms = _comma_join(event.get("logical_matched_terms"))
    if matched_terms:
        parts.append(f"matched: {matched_terms}")
    return " | ".join(parts) or None


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


def _join_reason_parts(prefix_parts: list[str], reasons: list[str]) -> str:
    parts = list(prefix_parts)
    if reasons:
        parts.append(f"Judge: {'; '.join(reasons)}.")
    return " ".join(parts)


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
