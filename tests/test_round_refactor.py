from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from taboo_arena.app.jobs import _benchmark_process_target
from taboo_arena.app.live_round import advance_live_round, start_live_round
from taboo_arena.app.session_facade import SessionFacade
from taboo_arena.config import AppSettings, RunSettings
from taboo_arena.engine import BenchmarkPlan, BenchmarkProgress, BenchmarkSummary
from taboo_arena.engine.round_engine import RoundEngine
from taboo_arena.engine.round_session import RoundPhase, RoundStepper
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.logging.schemas import RoundSummaryRecord
from taboo_arena.models.manager import ModelManager
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


def _entry(
    model_id: str,
    backend: Literal["transformers_safetensors", "llama_cpp_gguf"] = "transformers_safetensors",
) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        display_name=model_id,
        backend=backend,
        repo_id=f"fake/{model_id}",
        architecture_family="fake",
        chat_template_id="generic_completion",
        supports_system_prompt=True,
        roles_supported=["cluer", "guesser", "judge"],
        languages=["en"],
        filename="model.gguf" if backend == "llama_cpp_gguf" else None,
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


def test_round_stepper_matches_round_engine_behavior(sample_card, tmp_path: Path) -> None:
    responses = {
        "cluer": [
            _cluer_candidates(("type", "Bear clue")),
            _cluer_candidates(("use", "forest giant")),
        ],
        "judge": [CLUE_ALLOW, GUESS_CORRECT],
        "guesser": [_guesser_candidates("Bear", "Wolf", "Fox")],
    }

    engine_logger = RunLogger(log_root=tmp_path / "engine", console_trace=False)
    stepper_logger = RunLogger(log_root=tmp_path / "stepper", console_trace=False)
    engine = RoundEngine(
        model_manager=FakeModelManager(responses={key: list(value) for key, value in responses.items()}),
        logger=engine_logger,
        settings=RunSettings(),
    )
    stepper = RoundStepper(
        model_manager=FakeModelManager(responses={key: list(value) for key, value in responses.items()}),
        logger=stepper_logger,
        settings=RunSettings(),
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    engine_result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )
    stepper_result = stepper.run_to_completion()

    assert stepper_result.solved == engine_result.solved
    assert stepper_result.solved_on_attempt == engine_result.solved_on_attempt
    assert stepper_result.total_guess_attempts_used == engine_result.total_guess_attempts_used
    assert stepper_result.total_clue_repairs == engine_result.total_clue_repairs
    assert stepper_result.terminal_reason == engine_result.terminal_reason
    assert [event["event_type"] for event in stepper_logger.snapshot_events()] == [
        event["event_type"] for event in engine_logger.snapshot_events()
    ]


def test_live_round_adapter_delegates_to_canonical_stepper(sample_card, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": [_cluer_candidates(("use", "forest giant"))],
            "judge": [CLUE_ALLOW, GUESS_CORRECT],
            "guesser": [_guesser_candidates("Bear", "Wolf", "Fox")],
        }
    )
    controller = start_live_round(
        settings=RunSettings(),
        model_manager=manager,
        logger=logger,
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert isinstance(controller.stepper, RoundStepper)
    assert controller.phase == "clue_prepare"

    while controller.result is None and controller.error_message is None:
        advance_live_round(controller, model_manager=manager)

    assert controller.result is not None
    assert controller.result.solved is True
    assert controller.stepper.state.phase is RoundPhase.FINISHED


def test_jobs_benchmark_target_uses_canonical_benchmark_runner(
    monkeypatch,
    sample_card,
    tmp_path: Path,
) -> None:
    settings = AppSettings()
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    plan = BenchmarkPlan(
        benchmark_id="benchmark_test",
        cluer_model_id="cluer",
        guesser_model_id="guesser",
        judge_model_id="judge",
        seed=settings.run.random_seed,
        card_selection_mode="all_eligible",
        eligible_card_count=1,
        eligible_card_ids=[sample_card.id],
        played_card_ids=[sample_card.id],
        selected_categories=[str(sample_card.category_label or sample_card.source_category)],
    )
    seen: dict[str, Any] = {}
    updates: list[dict[str, Any]] = []

    class _Queue:
        def put(self, item: dict[str, Any]) -> None:
            updates.append(item)

    class _FakeBenchmarkRunner:
        def __init__(self, engine: Any, logger: RunLogger) -> None:
            seen["engine"] = engine
            seen["logger"] = logger

        def run_plan(
            self,
            *,
            plan: BenchmarkPlan,
            cards_by_id: dict[str, Any],
            cluer_entry: Any,
            guesser_entry: Any,
            judge_entry: Any,
            stop_requested: Any = None,
            progress_callback: Any = None,
            round_completed_callback: Any = None,
        ) -> BenchmarkSummary:
            del round_completed_callback
            seen["plan"] = plan
            seen["cards_by_id"] = cards_by_id
            seen["cluer_entry"] = cluer_entry
            seen["guesser_entry"] = guesser_entry
            seen["judge_entry"] = judge_entry
            seen["stop_requested"] = stop_requested
            if progress_callback is not None:
                progress_callback(
                    BenchmarkProgress(
                        benchmark_id=plan.benchmark_id,
                        completed_cards=1,
                        total_cards=1,
                        current_card_id=sample_card.id,
                        current_target=sample_card.target,
                        solved_so_far=1,
                        card_solve_rate_so_far=1.0,
                        elapsed_seconds=0.01,
                    )
                )
            return BenchmarkSummary(
                benchmark_id=plan.benchmark_id,
                cluer_model_id=plan.cluer_model_id,
                guesser_model_id=plan.guesser_model_id,
                judge_model_id=plan.judge_model_id,
                seed=plan.seed,
                card_selection_mode=plan.card_selection_mode,
                requested_sample_size=plan.requested_sample_size,
                eligible_card_count=plan.eligible_card_count,
                played_card_ids=list(plan.played_card_ids),
                selected_categories=list(plan.selected_categories),
                cards_played_total=1,
                cards_solved_total=1,
                card_solve_rate=1.0,
                solved_on_attempt_1_count=1,
                solved_on_attempt_2_count=0,
                solved_on_attempt_3_count=0,
                clue_not_repaired_count=0,
                max_guess_attempts_reached_count=0,
                average_repairs_per_card=0.0,
                average_latency_ms=12.5,
                failure_reason_breakdown={},
                elapsed_seconds=0.01,
            )

    monkeypatch.setattr("taboo_arena.app.jobs.BenchmarkRunner", _FakeBenchmarkRunner)
    _benchmark_process_target(
        update_queue=_Queue(),
        settings_payload=settings.model_dump(mode="json"),
        run_id=logger.run_id,
        log_root=str(logger.log_root),
        plan_payload=plan.model_dump(mode="json"),
        cards_payload=[sample_card.model_dump(mode="json")],
        cluer_payload=_entry("cluer").model_dump(mode="json"),
        guesser_payload=_entry("guesser").model_dump(mode="json"),
        judge_payload=_entry("judge").model_dump(mode="json"),
    )

    assert seen["plan"].benchmark_id == "benchmark_test"
    assert list(seen["cards_by_id"]) == [sample_card.id]
    assert seen["cluer_entry"].id == "cluer"
    assert seen["guesser_entry"].id == "guesser"
    assert seen["judge_entry"].id == "judge"
    assert any(update["type"] == "benchmark_progress" for update in updates)
    progress_update = next(update for update in updates if update["type"] == "benchmark_progress")
    assert progress_update["progress"]["current_target"] == sample_card.target
    assert any(update["type"] == "benchmark_summary" for update in updates)
    assert updates[-1]["type"] == "completed"


def test_run_logger_buffered_flush_preserves_output_files(tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    logger.emit("round_started", state="generating_clue", round_id="round_1")
    logger.record_round_summary(
        RoundSummaryRecord(
            run_id=logger.run_id,
            round_id="round_1",
            card_id="animals:bear:0001",
            target="Bear",
            solved=True,
            solved_on_attempt=1,
            total_guess_attempts_used=1,
            total_clue_repairs=1,
            first_clue_passed_without_repair=True,
            clue_repaired_successfully=False,
            clue_not_repaired=False,
            terminal_reason="solved",
            cluer_model_id="cluer",
            guesser_model_id="guesser",
            judge_model_id="judge",
            total_latency_ms=10.0,
        ),
        flush=False,
    )

    assert not logger.rounds_csv_path.exists()
    logger.flush()
    assert logger.rounds_csv_path.exists()
    assert logger.rounds_parquet_path.exists()
    assert logger.summary_csv_path.exists()


def test_backend_runtime_capabilities_are_exposed() -> None:
    manager = ModelManager()

    transformers_caps = manager.runtime_capabilities(
        _entry("tx"),
        device_preference="auto",
        runtime_policy="keep_cpu_offloaded_if_possible",
    )
    gguf_caps = manager.runtime_capabilities(
        _entry("gguf", backend="llama_cpp_gguf"),
        device_preference="auto",
        runtime_policy="keep_cpu_offloaded_if_possible",
    )

    assert transformers_caps.supports_banned_phrase_enforcement is True
    assert transformers_caps.supports_cpu_offload is True
    assert gguf_caps.supports_banned_phrase_enforcement is False
    assert gguf_caps.supports_cpu_offload is False


def test_session_facade_wraps_state_dict() -> None:
    state: dict[str, Any] = {
        "app_randomizer": __import__("random").Random(7),
        "transcript_history_events": [],
        "transcript_history_run_ids": [],
        "session_history_round_summaries": [],
        "resource_history": [],
    }
    session = SessionFacade(state)

    session.current_error_message = "oops"
    session.current_result = {"ok": True}
    session.transcript_history_events = [{"event_type": "round_started"}]

    assert session.current_error_message == "oops"
    assert session.current_result == {"ok": True}
    assert session.transcript_history_events[0]["event_type"] == "round_started"
