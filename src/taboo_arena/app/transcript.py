"""Helpers for rendering a messenger-like transcript from logger events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

TranscriptTone = Literal["pending", "rejected", "accepted", "judge", "guess", "success", "meta"]
TranscriptAlignment = Literal["left", "center", "right"]


@dataclass(slots=True)
class TranscriptMessage:
    """One visible message bubble in the transcript panel."""

    role: str
    label: str
    text: str
    tone: TranscriptTone
    alignment: TranscriptAlignment
    message_id: str = ""
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
    judge_indexes: dict[tuple[int, int], int] = {}
    guess_indexes: dict[int, int] = {}
    active_round_id: str | None = None

    for event in events:
        round_id = _event_round_id(event, active_round_id)
        if round_id != active_round_id:
            active_round_id = round_id
            clue_indexes = {}
            judge_indexes = {}
            guess_indexes = {}
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
            clue_indexes[clue_key] = len(messages)
            messages.append(
                TranscriptMessage(
                    role="cluer",
                    label=_cluer_label(attempt_no, repair_no),
                    text="Thinking...",
                    tone="pending",
                    alignment="left",
                    message_id=f"{round_id}:clue:{attempt_no}:{repair_no}",
                )
            )
            continue

        if event_type == "clue_draft_generated":
            clue_message = _lookup_clue_message(messages, clue_indexes, clue_key)
            clue_text = str(event.get("clue_text_raw", "")).strip() or "(empty clue)"
            if clue_message is not None:
                clue_message.text = clue_text
                clue_message.prompt_text = _prompt_text(event)
                clue_message.prompt_model_id = _optional_text(event.get("prompt_model_id"))
                clue_message.prompt_template_id = _prompt_template_id(event)
            else:
                clue_indexes[clue_key] = len(messages)
                messages.append(
                    TranscriptMessage(
                        role="cluer",
                        label=_cluer_label(attempt_no, repair_no),
                        text=clue_text,
                        tone="pending",
                        alignment="left",
                        message_id=f"{round_id}:clue:{attempt_no}:{repair_no}",
                        prompt_text=_prompt_text(event),
                        prompt_model_id=_optional_text(event.get("prompt_model_id")),
                        prompt_template_id=_prompt_template_id(event),
                    )
                )
            continue

        if event_type == "clue_review_started":
            judge_indexes[clue_key] = len(messages)
            messages.append(
                TranscriptMessage(
                    role="judge",
                    label="Judge",
                    text="Reviewing clue...",
                    tone="judge",
                    alignment="center",
                    message_id=f"{round_id}:judge-review:{attempt_no}:{repair_no}",
                )
            )
            continue

        if event_type == "llm_validation_completed":
            clue_message = _lookup_clue_message(messages, clue_indexes, clue_key)
            if clue_message is None:
                continue
            verdict = str(event.get("final_judge_verdict", ""))
            clue_message.tone = "rejected" if verdict == "fail" else "accepted"
            judge_index = judge_indexes.pop(clue_key, None)
            if verdict == "pass_with_warning":
                warning_text = _join_reason_parts(
                    ["Approved with warning."],
                    _string_list(event.get("llm_judge_reasons")),
                )
                if judge_index is not None and judge_index < len(messages):
                    messages[judge_index] = TranscriptMessage(
                        role="judge",
                        label="Judge",
                        text=warning_text,
                        tone="judge",
                        alignment="center",
                        message_id=f"{round_id}:judge-warning:{attempt_no}:{repair_no}",
                        prompt_text=_prompt_text(event),
                        prompt_model_id=_optional_text(event.get("prompt_model_id")),
                        prompt_template_id=_prompt_template_id(event),
                    )
                else:
                    messages.append(
                        TranscriptMessage(
                            role="judge",
                            label="Judge",
                            text=warning_text,
                            tone="judge",
                            alignment="center",
                            message_id=f"{round_id}:judge-warning:{attempt_no}:{repair_no}",
                            prompt_text=_prompt_text(event),
                            prompt_model_id=_optional_text(event.get("prompt_model_id")),
                            prompt_template_id=_prompt_template_id(event),
                        )
                    )
            elif judge_index is not None and judge_index < len(messages):
                messages[judge_index] = None
            continue

        if event_type == "clue_repair_requested":
            rejection_message = TranscriptMessage(
                role="judge",
                label="Judge",
                text=_format_rejection_reason(event),
                tone="judge",
                alignment="center",
                message_id=f"{round_id}:judge-reject:{attempt_no}:{repair_no}",
                prompt_text=_prompt_text(event),
                prompt_model_id=_optional_text(event.get("prompt_model_id")),
                prompt_template_id=_prompt_template_id(event),
            )
            judge_index = judge_indexes.pop(clue_key, None)
            if judge_index is not None and judge_index < len(messages):
                messages[judge_index] = rejection_message
            else:
                messages.append(
                    rejection_message
                )
            continue

        if event_type == "guess_started":
            guess_indexes[attempt_no] = len(messages)
            messages.append(
                TranscriptMessage(
                    role="guesser",
                    label=f"Guesser - attempt {attempt_no}",
                    text="Thinking...",
                    tone="guess",
                    alignment="right",
                    message_id=f"{round_id}:guess:{attempt_no}",
                )
            )
            continue

        if event_type == "guess_generated":
            solved = str(event.get("state", "")) == "round_finished"
            guess_text = str(event.get("guess_text_raw", "")).strip() or "(empty guess)"
            guess_index = guess_indexes.get(attempt_no)
            if guess_index is not None and guess_index < len(messages):
                guess_message = messages[guess_index]
                if guess_message is not None:
                    guess_message.text = guess_text
                    guess_message.tone = "success" if solved else "guess"
                    guess_message.prompt_text = _prompt_text(event)
                    guess_message.prompt_model_id = _optional_text(event.get("prompt_model_id"))
                    guess_message.prompt_template_id = _prompt_template_id(event)
                    continue
            messages.append(
                TranscriptMessage(
                    role="guesser",
                    label=f"Guesser - attempt {attempt_no}",
                    text=guess_text,
                    tone="success" if solved else "guess",
                    alignment="right",
                    message_id=f"{round_id}:guess:{attempt_no}",
                    prompt_text=_prompt_text(event),
                    prompt_model_id=_optional_text(event.get("prompt_model_id")),
                    prompt_template_id=_prompt_template_id(event),
                )
            )

    return [message for message in messages if message is not None]


def _lookup_clue_message(
    messages: list[TranscriptMessage | None],
    clue_indexes: dict[tuple[int, int], int],
    clue_key: tuple[int, int],
) -> TranscriptMessage | None:
    index = clue_indexes.get(clue_key)
    if index is None:
        return None
    if index >= len(messages):
        return None
    return messages[index]


def _cluer_label(attempt_no: int, repair_no: int) -> str:
    if repair_no <= 1:
        return f"Cluer - attempt {attempt_no}"
    return f"Cluer - repair {repair_no}"


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
