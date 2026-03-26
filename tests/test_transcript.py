from __future__ import annotations

from typing import Any

from taboo_arena.app.transcript import (
    build_transcript_messages,
    latest_round_events,
    merge_transcript_event_sources,
)


def test_build_transcript_messages_renders_clue_judge_and_guess_flow() -> None:
    messages = build_transcript_messages(
        [
            {"event_type": "round_started", "round_id": "round_1"},
            {
                "event_type": "clue_draft_generated",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "clue_text_raw": "forest giant",
            },
            {
                "event_type": "llm_validation_completed",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "final_judge_verdict": "fail",
            },
            {
                "event_type": "clue_repair_requested",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "logical_violations": ["taboo overlap"],
                "logical_matched_terms": ["Star Wars"],
                "llm_judge_verdict": "pass",
                "judge_disagreement": True,
                "llm_judge_reasons": ["too close to the target"],
            },
            {
                "event_type": "clue_draft_generated",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 2,
                "clue_text_raw": "winter sleeper",
            },
            {
                "event_type": "llm_validation_completed",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 2,
                "final_judge_verdict": "pass",
            },
            {
                "event_type": "guess_generated",
                "round_id": "round_1",
                "attempt_no": 1,
                "guess_text_raw": "bear",
                "state": "round_finished",
            },
        ]
    )

    assert [message.role for message in messages] == ["meta", "cluer", "judge", "cluer", "guesser"]
    assert [message.tone for message in messages] == ["meta", "rejected", "judge", "accepted", "success"]
    assert messages[0].text == "Round • round_1"
    assert "Rejected clue." in messages[2].text
    assert "Matched terms: Star Wars." in messages[2].text
    assert "deterministic validator rejected it while the LLM judge returned pass" in messages[2].text
    assert messages[4].alignment == "right"


def test_build_transcript_messages_keeps_round_history_with_round_separators() -> None:
    messages = build_transcript_messages(
        [
            {
                "event_type": "round_started",
                "round_id": "round_older",
                "card_id": "animals:bear:0001",
            },
            {
                "event_type": "clue_draft_generated",
                "round_id": "round_older",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "clue_text_raw": "old clue",
            },
            {
                "event_type": "round_started",
                "round_id": "round_newer",
                "card_id": "places:papua-new-guinea:0001",
            },
            {
                "event_type": "clue_draft_generated",
                "round_id": "round_newer",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "clue_text_raw": "fresh clue",
            },
        ]
    )

    assert len(messages) == 4
    assert messages[0].text == "Round • animals:bear:0001"
    assert messages[1].text == "old clue"
    assert messages[2].text == "Round • places:papua-new-guinea:0001"
    assert messages[3].text == "fresh clue"


def test_build_transcript_messages_carries_prompt_metadata() -> None:
    messages = build_transcript_messages(
        [
            {
                "event_type": "round_started",
                "round_id": "round_1",
                "card_id": "animals:bear:0001",
            },
            {
                "event_type": "clue_draft_generated",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "clue_text_raw": "forest giant",
                "prompt_text": "You are the cluer.",
                "prompt_model_id": "phi-4-mini-instruct",
                "prompt_template_id": "phi_chat",
            },
        ]
    )

    assert messages[1].prompt_text == "You are the cluer."
    assert messages[1].prompt_model_id == "phi-4-mini-instruct"
    assert messages[1].prompt_template_id == "phi_chat"


def test_build_transcript_messages_handles_live_events_without_round_id() -> None:
    messages = build_transcript_messages(
        [
            {
                "event_type": "clue_draft_generated",
                "run_id": "run_live",
                "card_id": "food:soup:0112",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "clue_text_raw": "comforting meal",
            },
            {
                "event_type": "clue_repair_requested",
                "run_id": "run_live",
                "card_id": "food:soup:0112",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "llm_judge_verdict": "fail",
                "judge_disagreement": True,
                "llm_judge_reasons": ["too direct"],
            },
            {
                "event_type": "guess_generated",
                "run_id": "run_live",
                "card_id": "food:soup:0112",
                "attempt_no": 1,
                "guess_text_raw": "stew",
                "state": "generating_clue",
            },
        ]
    )

    assert [message.role for message in messages] == ["meta", "cluer", "judge", "guesser"]
    assert messages[0].text == "Round • food:soup:0112"
    assert messages[1].text == "comforting meal"
    assert "Rejected clue." in messages[2].text
    assert messages[3].text == "stew"


def test_merge_transcript_optional_text_treats_none_as_missing() -> None:
    messages = build_transcript_messages(
        [
            {
                "event_type": "round_started",
                "round_id": "round_1",
            },
            {
                "event_type": "clue_draft_generated",
                "round_id": None,
                "attempt_no": 1,
                "clue_repair_no": 1,
                "clue_text_raw": "forest giant",
            },
        ]
    )

    assert [message.text for message in messages] == ["Round • round_1", "forest giant"]


def test_merge_transcript_event_sources_skips_current_run_when_already_archived() -> None:
    archived_event = {
        "event_type": "clue_draft_generated",
        "run_id": "run_1",
        "round_id": "round_1",
        "attempt_no": 1,
        "clue_repair_no": 1,
        "clue_text_raw": "old clue",
    }

    merged = merge_transcript_event_sources(
        history_events=[archived_event],
        current_events=[archived_event],
        archived_run_ids=["run_1"],
        current_run_id="run_1",
    )

    assert merged == [archived_event]


def test_merge_transcript_event_sources_appends_new_run_without_replaying_history() -> None:
    history_event = {
        "event_type": "clue_draft_generated",
        "run_id": "run_1",
        "round_id": "round_1",
        "attempt_no": 1,
        "clue_repair_no": 1,
        "clue_text_raw": "old clue",
    }
    current_events: list[dict[str, Any]] = [
        {
            "event_type": "round_started",
            "run_id": "run_2",
            "round_id": "round_2",
            "card_id": "animals:alpaca:0001",
        },
        {
            "event_type": "clue_draft_generated",
            "run_id": "run_2",
            "round_id": "round_2",
            "attempt_no": 1,
            "clue_repair_no": 1,
            "clue_text_raw": "fresh clue",
        },
    ]

    merged = merge_transcript_event_sources(
        history_events=[history_event],
        current_events=current_events,
        archived_run_ids=["run_1"],
        current_run_id="run_2",
    )

    messages = build_transcript_messages(merged)

    assert [message.text for message in messages] == [
        "Round • round_1",
        "old clue",
        "Round • animals:alpaca:0001",
        "fresh clue",
    ]


def test_latest_round_events_keeps_only_newest_round() -> None:
    events = [
        {
            "event_type": "clue_draft_generated",
            "round_id": "round_older",
            "card_id": "cars:isuzu:0031",
            "clue_text_raw": "old clue",
        },
        {
            "event_type": "clue_draft_generated",
            "round_id": "round_newer",
            "card_id": "cars:isuzu:0031",
            "clue_text_raw": "fresh clue",
        },
        {
            "event_type": "guess_generated",
            "round_id": "round_newer",
            "card_id": "cars:isuzu:0031",
            "guess_text_raw": "jeep",
        },
    ]

    latest = latest_round_events(events)

    assert [event["event_type"] for event in latest] == [
        "clue_draft_generated",
        "guess_generated",
    ]
    assert all(event["round_id"] == "round_newer" for event in latest)


def test_latest_round_events_keeps_live_events_without_round_id_after_round_start() -> None:
    events: list[dict[str, Any]] = [
        {
            "event_type": "round_started",
            "round_id": "round_live",
            "card_id": "food:soup:0112",
        },
        {
            "event_type": "clue_draft_generated",
            "card_id": "food:soup:0112",
            "attempt_no": 1,
            "clue_repair_no": 1,
            "clue_text_raw": "comforting meal",
        },
        {
            "event_type": "guess_generated",
            "attempt_no": 1,
            "guess_text_raw": "stew",
        },
    ]

    latest = latest_round_events(events)

    assert [event["event_type"] for event in latest] == [
        "round_started",
        "clue_draft_generated",
        "guess_generated",
    ]
