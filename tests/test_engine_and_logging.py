from __future__ import annotations

from pathlib import Path

from taboo_arena.config import RunSettings
from taboo_arena.engine import BatchRunner, BatchSpec, RoundEngine
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.models.registry import ModelEntry
from tests.conftest import FakeModelManager


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


def test_round_engine_solves_after_hidden_repair(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": ["Bear clue", "forest giant"],
            "judge": [
                '{"verdict":"pass","reasons":[],"suspicious_terms":[],"confidence":0.9,"summary":"ok","judge_version":"v1"}',
                '{"verdict":"pass","reasons":[],"suspicious_terms":[],"confidence":0.9,"summary":"ok","judge_version":"v1"}',
            ],
            "guesser": ["Bear"],
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
    assert result.total_clue_repairs == 2
    assert result.clue_repaired_successfully is True
    assert logger.events_path.exists()
    assert logger.rounds_csv_path.exists()
    assert logger.rounds_parquet_path.exists()
    assert logger.summary_csv_path.exists()


def test_batch_runner_smoke(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": ["forest giant"],
            "judge": [
                '{"verdict":"pass","reasons":[],"suspicious_terms":[],"confidence":0.9,"summary":"ok","judge_version":"v1"}'
            ],
            "guesser": ["Bear"],
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


def test_round_engine_passes_cluer_guard_phrases(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    calls: list[dict[str, object]] = []
    manager = FakeModelManager(
        responses={
            "cluer": ["forest giant"],
            "judge": [
                '{"verdict":"pass","reasons":[],"suspicious_terms":[],"confidence":0.9,"summary":"ok","judge_version":"v1"}'
            ],
            "guesser": ["Bear"],
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
    assert "Grizzly Bear" in banned_phrases


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
            "cluer": ["forest giant"],
            "judge": [
                '{"verdict":"pass","reasons":[],"suspicious_terms":[],"confidence":0.9,"summary":"ok","judge_version":"v1"}'
            ],
            "guesser": ["Bear"],
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
            "cluer": ["forest giant", "forest giant again"],
            "judge": [
                '{"verdict":"pass","reasons":[],"suspicious_terms":[],"confidence":0.9,"summary":"ok","judge_version":"v1"}',
                '{"verdict":"pass","reasons":[],"suspicious_terms":[],"confidence":0.9,"summary":"ok","judge_version":"v1"}',
            ],
            "guesser": ["Wolf", "Fox"],
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
    assert result.terminal_reason == "max_guess_attempts_reached"


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
