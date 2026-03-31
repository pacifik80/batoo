from __future__ import annotations

import json
from pathlib import Path

from taboo_arena.config import RunSettings
from taboo_arena.engine import BatchRunner, BatchSpec, BenchmarkRunner, RoundEngine, create_benchmark_plan
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.models.registry import ModelEntry
from tests.conftest import FakeModelManager

CLUE_ALLOW = (
    '{"allow":true,"block_reason_codes":[],"warnings":[],"matched_surface_forms":[],'
    '"judge_version":"clue_judge_v1"}'
)
GUESS_CORRECT = (
    '{"correct":true,"reason_codes":["exact_target_present"],"warnings":[],'
    '"matched_surface_forms":["Bear"],"judge_version":"guess_judge_v1"}'
)
GUESS_INCORRECT = (
    '{"correct":false,"reason_codes":["target_absent"],"warnings":[],'
    '"matched_surface_forms":[],"judge_version":"guess_judge_v1"}'
)


def _entry(model_id: str) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        display_name=model_id,
        backend="transformers_safetensors",
        repo_id=f"fake/{model_id}",
        architecture_family="fake",
        chat_template_id="generic_completion",
        supports_system_prompt=True,
        roles_supported=["cluer", "guesser", "judge"],
        languages=["en"],
    )


def _cluer_candidates(*candidates: tuple[str, str]) -> str:
    return json.dumps(
        {
            "candidates": [
                {"angle": angle, "clue": clue}
                for angle, clue in candidates
            ]
        }
    )


def _guesser_candidates(*guesses: str) -> str:
    return json.dumps({"guesses": list(guesses)})


def test_round_engine_solves_after_hidden_repair(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": [
                _cluer_candidates(("type", "Bear clue")),
                _cluer_candidates(("use", "forest giant")),
            ],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [_guesser_candidates("Bear", "Wolf", "Fox")],
        }
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)
    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )
    assert result.solved is True
    assert result.total_guess_attempts_used == 1
    assert result.total_clue_repairs == 1
    assert result.clue_repaired_successfully is True
    assert logger.events_path.exists()
    assert logger.rounds_csv_path.exists()
    assert logger.rounds_parquet_path.exists()
    assert logger.summary_csv_path.exists()


def test_round_engine_selects_best_valid_clue_from_candidate_batch(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": [
                _cluer_candidates(
                    ("type", "Bear animal"),
                    ("use", "hibernates through winter"),
                    ("context", "forest giant"),
                )
            ],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [_guesser_candidates("Bear", "Wolf", "Fox")],
        }
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is True
    clue_event = next(event for event in logger.events if event["event_type"] == "clue_draft_generated")
    assert clue_event["visible_clue_text"] == "hibernates through winter"
    assert clue_event["selected_angle"] == "use"
    assert clue_event["clue_text_raw"] == "hibernates through winter"


def test_round_engine_logs_internal_clue_retry_when_candidate_batch_fails(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": [
                _cluer_candidates(
                    ("type", "Bear animal"),
                    ("use", "grizzly sleeper"),
                    ("context", "pooh friend"),
                ),
                _cluer_candidates(
                    ("effect", "hibernates through winter"),
                    ("part_whole", "large pawed mammal"),
                    ("historical_association", "seen in wilderness stories"),
                ),
            ],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [_guesser_candidates("Bear", "Wolf", "Fox")],
        }
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is True
    assert result.total_clue_repairs == 1
    retry_event = next(event for event in logger.events if event["event_type"] == "clue_internal_retry_requested")
    assert retry_event["blocked_angles"] == ["type", "use", "context"]
    assert retry_event["clue_internal_cycle_no"] == 1
    selected_event = next(event for event in logger.events if event["event_type"] == "clue_candidate_selected")
    assert selected_event["clue_internal_cycle_no"] == 2


def test_round_engine_hides_malformed_structured_cluer_output_until_retry(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    malformed_payload = '{"candidates":[{"angle":"use","clue":"forest giant"}'
    manager = FakeModelManager(
        responses={
            "cluer": [
                malformed_payload,
                _cluer_candidates(("use", "forest giant")),
            ],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [_guesser_candidates("Bear", "Wolf", "Fox")],
        }
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is True
    clue_generation_events = [
        event for event in logger.events if event["event_type"] == "clue_candidates_generated"
    ]
    assert clue_generation_events[0]["clue_candidate_parse_mode"] == "parse_failure"
    assert clue_generation_events[0]["raw_model_output"] == malformed_payload
    retry_event = next(event for event in logger.events if event["event_type"] == "clue_internal_retry_requested")
    assert "parse_failure" in retry_event["reason_codes"]
    clue_event = next(event for event in logger.events if event["event_type"] == "clue_draft_generated")
    assert clue_event["visible_clue_text"] == "forest giant"


def test_round_engine_accepts_backslash_escaped_structured_cluer_output(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": [
                '{\\"candidates\\":[{"angle":"type","clue":"Bear animal"},{"angle":"use","clue":"hibernates through winter"},{"angle":"context","clue":"forest giant"}]}'
            ],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [_guesser_candidates("Bear", "Wolf", "Fox")],
        }
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is True
    clue_event = next(event for event in logger.events if event["event_type"] == "clue_draft_generated")
    assert clue_event["visible_clue_text"] == "hibernates through winter"
    candidate_event = next(event for event in logger.events if event["event_type"] == "clue_candidates_generated")
    assert candidate_event["clue_candidate_parse_mode"] == "json"


def test_batch_runner_smoke(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": [_cluer_candidates(("use", "forest giant"))],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [_guesser_candidates("Bear", "Wolf", "Fox")],
        }
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)
    batch = BatchRunner(engine, logger)
    results = batch.run(
        BatchSpec(
            cluer_entries=[_entry("cluer")],
            guesser_entries=[_entry("guesser")],
            judge_entries=[_entry("judge")],
            cards=[sample_card],
            repeats_per_card=1,
            seed=7,
        )
    )
    assert len(results) == 1
    assert results[0].solved is True


def test_benchmark_runner_emits_progress_without_name_errors(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": [_cluer_candidates(("use", "forest giant"))],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [_guesser_candidates("Bear", "Wolf", "Fox")],
        }
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)
    runner = BenchmarkRunner(engine, logger)
    plan = create_benchmark_plan(
        eligible_cards=[sample_card],
        selected_categories=[str(sample_card.category_label or sample_card.source_category)],
        cluer_model_id="cluer",
        guesser_model_id="guesser",
        judge_model_id="judge",
        seed=7,
        card_selection_mode="all_eligible",
    )
    seen_progress = []

    summary = runner.run_plan(
        plan=plan,
        cards_by_id={sample_card.id: sample_card},
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
        progress_callback=lambda progress: seen_progress.append(progress),
    )

    assert summary.cards_played_total == 1
    assert summary.cards_solved_total == 1
    assert len(seen_progress) == 3
    assert seen_progress[-1].completed_cards == 1
    assert seen_progress[-1].card_solve_rate_so_far == 1.0


def test_round_engine_passes_cluer_guard_phrases(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    calls: list[dict[str, object]] = []
    manager = FakeModelManager(
        responses={
            "cluer": [_cluer_candidates(("use", "forest giant"))],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [_guesser_candidates("Bear", "Wolf", "Fox")],
        },
        calls=calls,
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    cluer_call = next(call for call in calls if call["trace_role"] == "cluer")
    banned_phrases = cluer_call["banned_phrases"]
    assert isinstance(banned_phrases, list)
    assert "Bear" in banned_phrases
    assert "grizzly" in banned_phrases


def test_round_engine_passes_only_visible_clue_text_to_guesser(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    calls: list[dict[str, object]] = []
    raw_cluer_payload = _cluer_candidates(("use", "forest giant"))
    manager = FakeModelManager(
        responses={
            "cluer": [raw_cluer_payload],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [_guesser_candidates("Bear", "Wolf", "Fox")],
        },
        calls=calls,
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is True
    guesser_call = next(call for call in calls if call["trace_role"] == "guesser")
    messages = guesser_call["messages"]
    assert isinstance(messages, list)
    message_text = "\n".join(str(getattr(message, "content", "")) for message in messages)
    assert "forest giant" in message_text
    assert raw_cluer_payload not in message_text


def test_round_engine_emits_selected_guess_before_judge_verdict(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": [_cluer_candidates(("use", "forest giant"))],
            "judge": [CLUE_ALLOW, GUESS_INCORRECT],
            "guesser": [_guesser_candidates("Wolf", "Fox")],
        }
    )
    settings = RunSettings(max_guess_attempts=1)
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is False
    review_event = next(event for event in logger.events if event["event_type"] == "guess_review_started")
    assert review_event["visible_guess_text"] == "Wolf"
    assert review_event["guess_text_raw"] == "Wolf"
    assert review_event["guess_hidden_retry_count"] == 0


def test_round_engine_passes_previous_wrong_guesses_as_guesser_guard_phrases(
    sample_card,
    tmp_path: Path,
) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    calls: list[dict[str, object]] = []
    manager = FakeModelManager(
        responses={
            "cluer": [
                _cluer_candidates(("use", "forest giant")),
                _cluer_candidates(("effect", "winter sleeper")),
            ],
            "judge": [CLUE_ALLOW, GUESS_INCORRECT, CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [
                _guesser_candidates("Wolf", "Fox", "Animal"),
                _guesser_candidates("Bear", "Wolf", "Fox"),
            ],
        },
        calls=calls,
    )
    settings = RunSettings(max_guess_attempts=2)
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is True
    guesser_calls = [call for call in calls if call["trace_role"] == "guesser"]
    assert len(guesser_calls) == 2
    assert guesser_calls[0]["banned_phrases"] == []
    assert guesser_calls[1]["banned_phrases"] == ["Wolf"]


def test_round_engine_records_prompt_trace_fields_without_event_field_collision(
    sample_card,
    tmp_path: Path,
) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    logger.latest_prompt_by_role = {
        "cluer": {
            "prompt": "Cluer prompt",
            "model_id": "cluer",
            "prompt_template_id": "generic_completion",
        },
        "judge": {
            "prompt": "Judge prompt",
            "model_id": "judge",
            "prompt_template_id": "generic_completion",
        },
        "guesser": {
            "prompt": "Guesser prompt",
            "model_id": "guesser",
            "prompt_template_id": "generic_completion",
        },
    }
    manager = FakeModelManager(
        responses={
            "cluer": [_cluer_candidates(("use", "forest giant"))],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [_guesser_candidates("Bear", "Wolf", "Fox")],
        }
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is True
    clue_event = next(event for event in logger.events if event["event_type"] == "clue_draft_generated")
    guess_event = next(event for event in logger.events if event["event_type"] == "guess_generated")
    assert clue_event["prompt_template_id"] == "generic_completion"
    assert clue_event["prompt_trace_template_id"] == "generic_completion"
    assert guess_event["prompt_trace_template_id"] == "generic_completion"


def test_round_engine_honors_configured_max_guess_attempts(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": [
                _cluer_candidates(("use", "forest giant")),
                _cluer_candidates(("effect", "forest giant again")),
            ],
            "judge": [CLUE_ALLOW, GUESS_INCORRECT, CLUE_ALLOW, GUESS_INCORRECT],
            "guesser": [
                _guesser_candidates("Wolf", "Animal", "Fox"),
                _guesser_candidates("Fox", "Animal", "Wolf"),
            ],
        }
    )
    settings = RunSettings(max_guess_attempts=2)
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is False
    assert result.total_guess_attempts_used == 2
    assert result.terminal_reason == "max_attempts_reached"


def test_round_engine_hides_malformed_structured_guesser_output_until_retry(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    malformed_payload = '{"guesses":["Bear","Wolf","Fox"]'
    manager = FakeModelManager(
        responses={
            "cluer": [_cluer_candidates(("use", "forest giant"))],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [
                malformed_payload,
                _guesser_candidates("Bear", "Wolf", "Fox"),
            ],
        }
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is True
    shortlist_events = [
        event for event in logger.events if event["event_type"] == "guess_shortlist_generated"
    ]
    assert shortlist_events[0]["guess_parse_mode"] == "parse_failure"
    assert shortlist_events[0]["raw_model_output"] == malformed_payload
    retry_event = next(event for event in logger.events if event["event_type"] == "guess_hidden_retry_requested")
    assert "parse_failure" in retry_event["reason_codes"]
    guess_event = next(event for event in logger.events if event["event_type"] == "guess_generated")
    assert guess_event["visible_guess_text"] == "Bear"


def test_round_engine_accepts_backslash_escaped_structured_guesser_output(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": [_cluer_candidates(("use", "forest giant"))],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [
                '{\\"guesses\\":["Bear","Wolf","Fox"]}',
            ],
        }
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is True
    guess_event = next(event for event in logger.events if event["event_type"] == "guess_generated")
    assert guess_event["visible_guess_text"] == "Bear"
    shortlist_event = next(event for event in logger.events if event["event_type"] == "guess_shortlist_generated")
    assert shortlist_event["guess_parse_mode"] == "json"


def test_round_engine_uses_hidden_guess_retry_for_repeated_wrapper_variants(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": [
                _cluer_candidates(("use", "forest giant")),
                _cluer_candidates(("effect", "winter sleeper")),
            ],
            "judge": [CLUE_ALLOW, GUESS_INCORRECT, CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [
                _guesser_candidates("wolf", "animal", "fox"),
                _guesser_candidates("my guess is wolf", "the wolf", "wolf"),
                _guesser_candidates("Bear", "Fox", "Animal"),
            ],
        }
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is True
    assert result.solved_on_attempt == 2
    guess_events = [event for event in logger.events if event["event_type"] == "guess_generated"]
    assert len(guess_events) == 2
    assert guess_events[1]["guess_text_raw"] == "Bear"
    assert guess_events[1]["guess_hidden_retry_count"] == 1
    assert guess_events[1]["guess_internal_cycle_no"] == 2
    retry_events = [event for event in logger.events if event["event_type"] == "guess_hidden_retry_requested"]
    assert len(retry_events) == 1
    assert retry_events[0]["guess_internal_cycle_no"] == 1


def test_round_engine_keeps_deterministic_correct_guess_when_guess_judge_disagrees(
    sample_card,
    tmp_path: Path,
) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": [_cluer_candidates(("use", "forest giant"))],
            "judge": [
                CLUE_ALLOW,
                (
                    '{"correct":false,"reason_codes":["judge_disagrees"],"warnings":["strict_read"],'
                    '"matched_surface_forms":[],"judge_version":"guess_judge_v1"}'
                ),
            ],
            "guesser": [_guesser_candidates("Bear", "Wolf", "Fox")],
        }
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is True
    guess_validation_event = next(
        event for event in logger.events if event["event_type"] == "guess_validation_completed"
    )
    assert guess_validation_event["guess_judge_correct"] is False
    assert guess_validation_event["final_guess_correct"] is True
    assert guess_validation_event["judge_disagreement"] is True


def test_round_engine_logs_specific_wrapper_match_reason(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": [_cluer_candidates(("use", "forest giant"))],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [_guesser_candidates("my guess is Bear", "Wolf", "Fox")],
        }
    )
    settings = RunSettings()
    engine = RoundEngine(model_manager=manager, logger=logger, settings=settings)

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is True
    guess_event = next(event for event in logger.events if event["event_type"] == "guess_generated")
    assert guess_event["visible_guess_text"] == "my guess is Bear"
    assert guess_event["guess_match_reason"] == "normalized_wrapper_match"
    assert "normalized_wrapper_match" in guess_event["guess_match_warnings"]


def test_run_logger_invokes_event_callback(tmp_path: Path) -> None:
    seen_events: list[dict[str, object]] = []
    logger = RunLogger(
        log_root=tmp_path,
        console_trace=False,
        event_callback=lambda event: seen_events.append(event),
    )

    logger.emit("round_started", state="generating_clue", round_id="round_1")

    assert len(seen_events) == 1
    assert seen_events[0]["event_type"] == "round_started"


def test_run_logger_snapshot_events_returns_stable_copy(tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    logger.emit("round_started", state="generating_clue", round_id="round_1")

    snapshot = logger.snapshot_events()
    snapshot.append({"event_type": "mutated"})

    assert [event["event_type"] for event in logger.snapshot_events()] == ["round_started"]
