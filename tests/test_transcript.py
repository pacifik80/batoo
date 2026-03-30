from __future__ import annotations

from typing import Any

from taboo_arena.app.transcript import (
    BubbleDebugSection,
    TranscriptDebugEntry,
    TranscriptMessage,
    build_transcript_messages,
    latest_round_events,
    merge_transcript_event_sources,
)
from taboo_arena.app.ui_transcript_panel import transcript_message_html


def test_build_transcript_messages_shows_only_final_visible_turns_per_attempt() -> None:
    messages = build_transcript_messages(
        [
            {"event_type": "round_started", "round_id": "round_1"},
            {
                "event_type": "clue_draft_generated",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "visible_clue_text": "forest giant",
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
                "logical_matched_terms": ["Bear"],
                "llm_judge_verdict": "fail",
                "judge_disagreement": False,
                "llm_judge_reasons": ["too close to the target"],
            },
            {
                "event_type": "clue_draft_generated",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 2,
                "visible_clue_text": "winter sleeper",
            },
            {
                "event_type": "llm_validation_completed",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 2,
                "final_judge_verdict": "pass",
            },
            {
                "event_type": "clue_accepted",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 2,
                "visible_clue_text": "winter sleeper",
            },
            {
                "event_type": "guess_generated",
                "round_id": "round_1",
                "attempt_no": 1,
                "visible_guess_text": "bear",
                "guess_match_status": "correct",
                "guess_match_reason": "exact_match",
                "state": "round_finished",
            },
        ]
    )

    assert [message.role for message in messages] == ["meta", "cluer", "judge", "guesser"]
    assert messages[0].text == "Round • round_1"
    assert messages[1].text == "winter sleeper"
    assert messages[2].text == "Approved."
    assert messages[3].text == "bear"
    assert messages[1].alignment == "left"
    assert messages[2].alignment == "center"
    assert messages[3].alignment == "right"
    assert any("hidden clue rejected" in step for step in messages[1].debug_timeline)


def test_build_transcript_messages_uses_visible_fields_not_raw_payloads() -> None:
    messages = build_transcript_messages(
        [
            {"event_type": "round_started", "round_id": "round_1"},
            {
                "event_type": "clue_draft_generated",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "visible_clue_text": "winter sleeper",
                "clue_text_raw": '{"candidates":[{"angle":"use","clue":"json blob"}]}',
            },
            {
                "event_type": "llm_validation_completed",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "final_judge_verdict": "pass",
            },
            {
                "event_type": "guess_generated",
                "round_id": "round_1",
                "attempt_no": 1,
                "visible_guess_text": "bear",
                "guess_text_raw": '{"guesses":["bear","wolf","fox"]}',
                "state": "round_finished",
            },
        ]
    )

    assert messages[1].text == "winter sleeper"
    assert messages[2].text == "Approved."
    assert messages[3].text == "bear"
    assert all("{" not in message.text for message in messages if message.role != "meta")


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
                "visible_clue_text": "old clue",
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
                "visible_clue_text": "fresh clue",
            },
        ]
    )

    assert [message.text for message in messages] == [
        "Round • animals:bear:0001",
        "old clue",
        "Round • places:papua-new-guinea:0001",
        "fresh clue",
    ]


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
                "visible_clue_text": "forest giant",
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
                "visible_clue_text": "comforting meal",
            },
            {
                "event_type": "guess_generated",
                "run_id": "run_live",
                "card_id": "food:soup:0112",
                "attempt_no": 1,
                "visible_guess_text": "stew",
                "state": "generating_clue",
            },
        ]
    )

    assert [message.role for message in messages] == ["meta", "cluer", "guesser"]
    assert messages[0].text == "Round • food:soup:0112"
    assert messages[1].text == "comforting meal"
    assert messages[2].text == "stew"


def test_build_transcript_messages_tracks_live_status_progression_and_debug_details() -> None:
    messages = build_transcript_messages(
        [
            {"event_type": "round_started", "round_id": "round_live"},
            {
                "event_type": "clue_draft_started",
                "round_id": "round_live",
                "attempt_no": 1,
                "clue_repair_no": 1,
            },
            {
                "event_type": "clue_candidate_cycle_started",
                "round_id": "round_live",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "allowed_angles": ["type", "use", "context"],
                "blocked_terms": ["bear"],
            },
            {
                "event_type": "clue_candidates_generated",
                "round_id": "round_live",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "clue_candidate_angles": ["type", "use", "context"],
                "clue_candidate_clues": ["forest mammal", "winter sleeper", "deep woods"],
                "clue_candidate_parse_mode": "json",
            },
            {
                "event_type": "clue_candidate_validation_completed",
                "round_id": "round_live",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "candidate_results": [
                    {
                        "angle": "type",
                        "logical_verdict": "fail",
                        "logical_violations": ["whole_word_match"],
                    },
                    {
                        "angle": "use",
                        "logical_verdict": "pass",
                        "logical_violations": [],
                    },
                ],
            },
            {
                "event_type": "clue_candidate_selected",
                "round_id": "round_live",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "selected_angle": "use",
                "clue_candidate_clues": ["forest mammal", "winter sleeper", "deep woods"],
            },
            {
                "event_type": "clue_draft_generated",
                "round_id": "round_live",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "visible_clue_text": "winter sleeper",
            },
            {
                "event_type": "guess_started",
                "round_id": "round_live",
                "attempt_no": 1,
            },
            {
                "event_type": "guess_shortlist_generated",
                "round_id": "round_live",
                "attempt_no": 1,
                "guess_shortlist_candidates": ["wolf", "bear", "fox"],
                "guess_shortlist_candidate_keys": [["wolf"], ["bear"], ["fox"]],
                "guess_shortlist_invalid_reasons": ["", "", ""],
                "guess_shortlist_repeated_against": [[], [], []],
                "guess_parse_mode": "json",
            },
            {
                "event_type": "guess_generated",
                "round_id": "round_live",
                "attempt_no": 1,
                "visible_guess_text": "bear",
                "guess_match_status": "correct",
                "guess_match_reason": "exact_match",
                "guess_match_warnings": ["normalized_wrapper_match"],
                "guess_match_candidates": ["bear"],
                "guess_shortlist_candidates": ["wolf", "bear", "fox"],
                "state": "round_finished",
            },
        ]
    )

    cluer_message = messages[1]
    guesser_message = messages[2]

    assert cluer_message.text == "winter sleeper"
    assert cluer_message.status_text is None
    assert cluer_message.debug_timeline[:3] == ["planning", "drafting candidates", "candidate batch parsed"]
    assert any(entry.label == "Allowed angles" for entry in cluer_message.debug_entries)
    assert any(entry.label == "Internal clue candidates" for entry in cluer_message.debug_entries)
    assert any(entry.label == "Candidate validation" for entry in cluer_message.debug_entries)
    assert guesser_message.text == "bear"
    assert guesser_message.status_text is None
    assert any(entry.label == "Guess shortlist" for entry in guesser_message.debug_entries)
    assert any(entry.label == "Canonicalization" for entry in guesser_message.debug_entries)


def test_build_transcript_messages_shows_visible_clue_during_judge_review() -> None:
    messages = build_transcript_messages(
        [
            {"event_type": "round_started", "round_id": "round_live"},
            {
                "event_type": "clue_draft_generated",
                "round_id": "round_live",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "visible_clue_text": "winter sleeper",
            },
            {
                "event_type": "clue_review_started",
                "round_id": "round_live",
                "attempt_no": 1,
                "clue_repair_no": 1,
            },
        ]
    )

    assert [message.role for message in messages] == ["meta", "cluer", "judge"]
    assert messages[1].text == "winter sleeper"
    assert messages[1].status_text is None
    assert messages[2].status_text == "checking rules"


def test_build_transcript_messages_preserves_raw_payloads_only_in_debug() -> None:
    messages = build_transcript_messages(
        [
            {"event_type": "round_started", "round_id": "round_debug"},
            {
                "event_type": "clue_candidates_generated",
                "round_id": "round_debug",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "clue_candidate_parse_mode": "parse_failure",
                "raw_model_output": '{"candidates":[{"angle":"type","clue":"oops"}',
            },
            {
                "event_type": "clue_internal_retry_requested",
                "round_id": "round_debug",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "reason_codes": ["parse_failure"],
            },
            {
                "event_type": "clue_draft_generated",
                "round_id": "round_debug",
                "attempt_no": 1,
                "clue_repair_no": 2,
                "visible_clue_text": "winter sleeper",
            },
            {
                "event_type": "llm_validation_completed",
                "round_id": "round_debug",
                "attempt_no": 1,
                "clue_repair_no": 2,
                "final_judge_verdict": "pass",
            },
        ]
    )

    cluer_message = messages[1]
    assert cluer_message.text == "winter sleeper"
    assert any("hidden repair 1 requested" in step for step in cluer_message.debug_timeline)
    assert any(artifact.label == "Cluer raw model output" for artifact in cluer_message.raw_artifacts)
    assert all("{" not in message.text for message in messages if message.role != "meta")


def test_build_transcript_messages_hidden_repairs_do_not_create_extra_public_bubbles() -> None:
    messages = build_transcript_messages(
        [
            {"event_type": "round_started", "round_id": "round_1"},
            {
                "event_type": "clue_draft_generated",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 1,
                "visible_clue_text": "forest giant",
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
                "llm_judge_verdict": "fail",
            },
            {
                "event_type": "clue_draft_generated",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 2,
                "visible_clue_text": "winter sleeper",
            },
            {
                "event_type": "llm_validation_completed",
                "round_id": "round_1",
                "attempt_no": 1,
                "clue_repair_no": 2,
                "final_judge_verdict": "pass",
            },
            {
                "event_type": "guess_review_started",
                "round_id": "round_1",
                "attempt_no": 1,
            },
            {
                "event_type": "guess_validation_completed",
                "round_id": "round_1",
                "attempt_no": 1,
                "final_guess_correct": True,
            },
        ]
    )

    assert [message.role for message in messages] == ["meta", "cluer", "judge", "judge"]
    assert len([message for message in messages if message.role == "cluer"]) == 1
    assert len([message for message in messages if message.message_id.endswith("judge-review:1")]) == 1
    assert len([message for message in messages if message.message_id.endswith("judge-guess:1")]) == 1


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
                "visible_clue_text": "forest giant",
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
        "visible_clue_text": "old clue",
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
        "visible_clue_text": "old clue",
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
            "visible_clue_text": "fresh clue",
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
            "visible_clue_text": "old clue",
        },
        {
            "event_type": "clue_draft_generated",
            "round_id": "round_newer",
            "card_id": "cars:isuzu:0031",
            "visible_clue_text": "fresh clue",
        },
        {
            "event_type": "guess_generated",
            "round_id": "round_newer",
            "card_id": "cars:isuzu:0031",
            "visible_guess_text": "jeep",
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
            "visible_clue_text": "comforting meal",
        },
        {
            "event_type": "guess_generated",
            "attempt_no": 1,
            "visible_guess_text": "stew",
        },
    ]

    latest = latest_round_events(events)

    assert [event["event_type"] for event in latest] == [
        "round_started",
        "clue_draft_generated",
        "guess_generated",
    ]


def test_transcript_message_html_renders_status_timeline_and_debug_details() -> None:
    html = transcript_message_html(
        TranscriptMessage(
            role="cluer",
            label="Cluer - attempt 1",
            public_text="winter sleeper",
            tone="accepted",
            alignment="left",
            status_label="selected",
            debug_timeline=["planning", "selected candidate accepted"],
            debug_sections=[
                BubbleDebugSection(
                    title="Summary",
                    fields=[
                        TranscriptDebugEntry(label="Selected angle", value="use"),
                        TranscriptDebugEntry(label="Internal clue candidates", value="type: forest mammal"),
                    ],
                )
            ],
        )
    )

    assert "selected" in html
    assert "winter sleeper" in html
    assert "Debug" in html
    assert "Timeline" in html
    assert "Selected angle" in html
