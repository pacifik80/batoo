"""Transcript panel rendering helpers."""

from __future__ import annotations

import html
import json
from typing import Any

import streamlit as st

from taboo_arena.app.session_facade import SessionFacade
from taboo_arena.app.transcript import (
    TranscriptMessage,
    build_transcript_messages,
    latest_round_events,
    merge_transcript_event_sources,
)
from taboo_arena.logging.run_logger import RunLogger


def live_logger_events(current_logger: RunLogger) -> list[dict[str, Any]]:
    """Read the freshest event stream for the active logger."""
    in_memory_events = current_logger.snapshot_events()
    try:
        if current_logger.events_path.exists():
            file_events: list[dict[str, Any]] = []
            with current_logger.events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        file_events.append(payload)
            if len(file_events) > len(in_memory_events):
                return file_events
            if in_memory_events:
                return in_memory_events
            if file_events:
                return file_events
    except OSError:
        pass
    return in_memory_events


def render_transcript_panel_content(
    *,
    session: SessionFacade,
    logger: RunLogger | None,
    current_result: Any,
    active_run_present: bool,
) -> None:
    """Render the current transcript panel contents."""
    transcript_messages = build_transcript_messages(
        transcript_source_events(session=session, current_logger=logger)
    )
    if current_result is not None:
        render_result_banner(current_result)
    if not transcript_messages:
        if active_run_present:
            st.info(active_transcript_placeholder(logger))
        elif current_result is None:
            st.info("Press Start to run the selected card here.")
        return
    render_transcript_messages(transcript_messages)


def transcript_source_events(
    *,
    session: SessionFacade,
    current_logger: RunLogger | None,
) -> list[dict[str, Any]]:
    """Return transcript events from the newest round only."""
    if current_logger is not None:
        current_events = live_logger_events(current_logger)
        current_round_events = latest_round_events(current_events)
        if any(str(event.get("round_id", "")).strip() for event in current_round_events):
            return current_round_events

    merged_events = merge_transcript_event_sources(
        history_events=session.transcript_history_events,
        current_events=[] if current_logger is None else live_logger_events(current_logger),
        archived_run_ids=session.transcript_history_run_ids,
        current_run_id=None if current_logger is None else current_logger.run_id,
    )
    return latest_round_events(merged_events)


def active_transcript_placeholder(current_logger: RunLogger | None) -> str:
    """Return a user-facing placeholder for the current live phase."""
    if current_logger is None:
        return "Starting the round..."
    logger_events = current_logger.snapshot_events()
    if not logger_events:
        return "Starting the round..."

    last_event = logger_events[-1]
    event_type = str(last_event.get("event_type", "")).strip()
    if event_type == "app_started":
        return "Starting the round..."
    if event_type == "round_started":
        return "Round started. Preparing the first clue..."
    if event_type == "model_download_started":
        return "Downloading model files..."
    if event_type == "model_load_started":
        return "Loading model weights into memory..."
    if event_type == "clue_draft_started":
        return "Cluer is drafting a clue..."
    if event_type in {"logical_validation_completed", "clue_review_started"}:
        return "Judge is reviewing the clue..."
    if event_type == "guess_started":
        return "Guesser is preparing an answer..."
    if event_type == "clue_repair_requested":
        return "Cluer is repairing the rejected clue..."
    return "Round is in progress..."


def archive_current_logger_for_transcript(session: SessionFacade) -> None:
    """Move the current logger into session transcript history once a run is superseded."""
    current_logger = session.current_logger
    if current_logger is None:
        return
    archived_run_ids = session.transcript_history_run_ids
    if current_logger.run_id in archived_run_ids:
        return
    session.transcript_history_events = [
        *session.transcript_history_events,
        *current_logger.snapshot_events(),
    ]
    session.session_history_round_summaries = [
        *session.session_history_round_summaries,
        *current_logger.snapshot_round_summaries(),
    ]
    session.transcript_history_run_ids = [*archived_run_ids, current_logger.run_id]


def render_result_banner(result: Any) -> None:
    """Render the final result summary above the transcript."""
    if result.solved:
        text = f"Solved on attempt {result.solved_on_attempt}"
        css_class = "result-banner result-success"
    elif result.terminal_reason == "clue_not_repaired":
        text = "Round ended: clue_not_repaired"
        css_class = "result-banner result-fail"
    else:
        text = f"Failed after {int(result.total_guess_attempts_used)} guesses"
        css_class = "result-banner result-fail"
    st.markdown(f"<div class='{css_class}'>{text}</div>", unsafe_allow_html=True)


def render_transcript_messages(messages: list[TranscriptMessage]) -> None:
    """Render transcript messages using the shared chat bubble layout."""
    if not messages:
        return
    transcript_html = "".join(transcript_message_html(message) for message in messages)
    st.markdown(f"<div class='transcript-wrap'>{transcript_html}</div>", unsafe_allow_html=True)


def transcript_message_html(message: TranscriptMessage) -> str:
    """Render one transcript bubble to HTML."""
    bubble_class = "transcript-meta" if message.tone == "meta" else f"transcript-{message.tone}"
    label_html = (
        f"<span class='transcript-label'>{html.escape(message.label)}</span>"
        if message.tone != "meta"
        else ""
    )
    status_html = (
        f"<div class='transcript-status'>{html.escape(message.status_text)}</div>"
        if message.status_text
        else ""
    )
    text_html = (
        f"<div class='transcript-text'>{html.escape(message.text)}</div>"
        if message.text
        else ""
    )
    debug_html = _transcript_debug_html(message)
    bubble_html = (
        f"<div class='transcript-bubble {bubble_class} transcript-inline-bubble'>"
        f"{label_html}{status_html}{text_html}{debug_html}</div>"
    )
    if message.tone == "meta":
        return f"<div class='transcript-row center'>{bubble_html}</div>"
    return f"<div class='transcript-row {message.alignment}'>{bubble_html}</div>"


def _transcript_debug_html(message: TranscriptMessage) -> str:
    if not message.debug_entries:
        return ""
    rows = "".join(
        (
            "<div class='transcript-debug-item'>"
            f"<div class='transcript-debug-label'>{html.escape(entry.label)}</div>"
            f"<div class='transcript-debug-value'>{html.escape(entry.value).replace(chr(10), '<br/>')}</div>"
            "</div>"
        )
        for entry in message.debug_entries
    )
    return (
        "<details class='transcript-debug'>"
        "<summary>Debug</summary>"
        f"<div class='transcript-debug-grid'>{rows}</div>"
        "</details>"
    )
