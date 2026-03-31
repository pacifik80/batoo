"""Background job helpers for Streamlit-triggered runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from time import time
from typing import Any, Literal

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.config import AppSettings
from taboo_arena.engine import (
    BenchmarkPlan,
    BenchmarkProgress,
    BenchmarkRunner,
    BenchmarkSummary,
    RoundEngine,
    RoundResult,
)
from taboo_arena.engine.benchmark import result_to_round_summary
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.models import ModelEntry, ModelManager


@dataclass(slots=True)
class ActiveJob:
    """One background run owned by the Streamlit session."""

    kind: Literal["single", "benchmark"]
    logger: RunLogger
    process: Any | None = None
    update_queue: Any | None = None
    started_at: float = field(default_factory=time)
    result: RoundResult | None = None
    benchmark_plan: BenchmarkPlan | None = None
    benchmark_progress: BenchmarkProgress | None = None
    benchmark_summary: BenchmarkSummary | None = None
    error_message: str | None = None
    completed: bool = False
    stop_requested: bool = False


def start_single_round_job(
    *,
    settings: AppSettings,
    model_manager: ModelManager,
    logger: RunLogger,
    card: Any,
    cluer_entry: ModelEntry,
    guesser_entry: ModelEntry,
    judge_entry: ModelEntry,
) -> ActiveJob:
    """Start one round in an isolated process."""
    del model_manager
    logger.set_run_metadata(
        {
            "run_mode": "single",
            "cluer_model_id": cluer_entry.id,
            "guesser_model_id": guesser_entry.id,
            "judge_model_id": judge_entry.id,
            "seed": settings.run.random_seed,
            "card_id": card.id,
        }
    )
    ctx = get_context("spawn")
    update_queue = ctx.Queue()
    process = ctx.Process(
        target=_single_round_process_target,
        daemon=True,
        name="taboo-arena-single-round",
        kwargs={
            "update_queue": update_queue,
            "settings_payload": settings.model_dump(mode="json"),
            "run_id": logger.run_id,
            "log_root": str(logger.log_root),
            "card_payload": card.model_dump(mode="json"),
            "cluer_payload": cluer_entry.model_dump(mode="json"),
            "guesser_payload": guesser_entry.model_dump(mode="json"),
            "judge_payload": judge_entry.model_dump(mode="json"),
        },
    )
    job = ActiveJob(
        kind="single",
        logger=logger,
        process=process,
        update_queue=update_queue,
    )
    process.start()
    return job


def start_benchmark_job(
    *,
    settings: AppSettings,
    model_manager: ModelManager,
    logger: RunLogger,
    plan: BenchmarkPlan,
    cards: list[CardRecord],
    cluer_entry: ModelEntry,
    guesser_entry: ModelEntry,
    judge_entry: ModelEntry,
) -> ActiveJob:
    """Start a benchmark in an isolated process."""
    del model_manager
    logger.set_run_metadata(
        {
            "run_mode": "benchmark",
            "benchmark_id": plan.benchmark_id,
            "cluer_model_id": cluer_entry.id,
            "guesser_model_id": guesser_entry.id,
            "judge_model_id": judge_entry.id,
            "seed": plan.seed,
            "card_selection_mode": plan.card_selection_mode,
            "requested_sample_size": plan.requested_sample_size,
            "eligible_card_count": plan.eligible_card_count,
            "actual_played_card_ids": list(plan.played_card_ids),
            "selected_categories_snapshot": list(plan.selected_categories),
        }
    )
    logger.write_json_artifact("benchmark_plan.json", plan.model_dump(mode="json"))
    ctx = get_context("spawn")
    update_queue = ctx.Queue()
    process = ctx.Process(
        target=_benchmark_process_target,
        daemon=True,
        name="taboo-arena-benchmark",
        kwargs={
            "update_queue": update_queue,
            "settings_payload": settings.model_dump(mode="json"),
            "run_id": logger.run_id,
            "log_root": str(logger.log_root),
            "plan_payload": plan.model_dump(mode="json"),
            "cards_payload": [card.model_dump(mode="json") for card in cards],
            "cluer_payload": cluer_entry.model_dump(mode="json"),
            "guesser_payload": guesser_entry.model_dump(mode="json"),
            "judge_payload": judge_entry.model_dump(mode="json"),
        },
    )
    job = ActiveJob(
        kind="benchmark",
        logger=logger,
        process=process,
        update_queue=update_queue,
        benchmark_plan=plan,
    )
    process.start()
    return job


def _single_round_process_target(
    *,
    update_queue: Any,
    settings_payload: dict[str, Any],
    run_id: str,
    log_root: str,
    card_payload: dict[str, Any],
    cluer_payload: dict[str, Any],
    guesser_payload: dict[str, Any],
    judge_payload: dict[str, Any],
) -> None:
    settings = AppSettings.model_validate(settings_payload)
    card = CardRecord.model_validate(card_payload)
    cluer_entry = ModelEntry.model_validate(cluer_payload)
    guesser_entry = ModelEntry.model_validate(guesser_payload)
    judge_entry = ModelEntry.model_validate(judge_payload)
    logger = RunLogger(
        run_id=run_id,
        log_root=Path(log_root),
        console_trace=settings.run.console_trace,
        event_callback=lambda event: update_queue.put({"type": "event", "event": event}),
    )
    logger.set_run_metadata(
        {
            "run_mode": "single",
            "cluer_model_id": cluer_entry.id,
            "guesser_model_id": guesser_entry.id,
            "judge_model_id": judge_entry.id,
            "seed": settings.run.random_seed,
            "card_id": card.id,
        }
    )
    model_manager = ModelManager(logger=logger)
    engine = RoundEngine(model_manager=model_manager, logger=logger, settings=settings.run)
    try:
        result = engine.play_round(
            card=card,
            cluer_entry=cluer_entry,
            guesser_entry=guesser_entry,
            judge_entry=judge_entry,
        )
        update_queue.put({"type": "result", "result": result})
        summaries = logger.snapshot_round_summaries()
        if summaries:
            update_queue.put({"type": "summary", "summary": summaries[-1].model_dump(mode="json")})
    except Exception as exc:
        logger.emit("error", error_message=str(exc), state="idle")
        update_queue.put({"type": "error", "error_message": str(exc)})
    finally:
        update_queue.put({"type": "completed"})


def _benchmark_process_target(
    *,
    update_queue: Any,
    settings_payload: dict[str, Any],
    run_id: str,
    log_root: str,
    plan_payload: dict[str, Any],
    cards_payload: list[dict[str, Any]],
    cluer_payload: dict[str, Any],
    guesser_payload: dict[str, Any],
    judge_payload: dict[str, Any],
) -> None:
    settings = AppSettings.model_validate(settings_payload)
    plan = BenchmarkPlan.model_validate(plan_payload)
    cards = [CardRecord.model_validate(payload) for payload in cards_payload]
    cluer_entry = ModelEntry.model_validate(cluer_payload)
    guesser_entry = ModelEntry.model_validate(guesser_payload)
    judge_entry = ModelEntry.model_validate(judge_payload)
    logger = RunLogger(
        run_id=run_id,
        log_root=Path(log_root),
        console_trace=settings.run.console_trace,
        event_callback=lambda event: update_queue.put({"type": "event", "event": event}),
    )
    logger.set_run_metadata(
        {
            "run_mode": "benchmark",
            "benchmark_id": plan.benchmark_id,
            "cluer_model_id": plan.cluer_model_id,
            "guesser_model_id": plan.guesser_model_id,
            "judge_model_id": plan.judge_model_id,
            "seed": plan.seed,
            "card_selection_mode": plan.card_selection_mode,
            "requested_sample_size": plan.requested_sample_size,
            "eligible_card_count": plan.eligible_card_count,
            "actual_played_card_ids": list(plan.played_card_ids),
            "selected_categories_snapshot": list(plan.selected_categories),
        }
    )
    logger.write_json_artifact("benchmark_plan.json", plan.model_dump(mode="json"))
    model_manager = ModelManager(logger=logger)
    engine = RoundEngine(model_manager=model_manager, logger=logger, settings=settings.run)
    runner = BenchmarkRunner(engine, logger)
    try:
        summary = runner.run_plan(
            plan=plan,
            cards_by_id={card.id: card for card in cards},
            cluer_entry=cluer_entry,
            guesser_entry=guesser_entry,
            judge_entry=judge_entry,
            progress_callback=lambda progress: update_queue.put(
                {"type": "benchmark_progress", "progress": progress.model_dump(mode="json")}
            ),
            round_completed_callback=lambda result: update_queue.put(
                {
                    "type": "summary",
                    "summary": result_to_round_summary(result).model_dump(mode="json"),
                }
            ),
        )
        update_queue.put({"type": "benchmark_summary", "summary": summary.model_dump(mode="json")})
    except Exception as exc:
        logger.emit("error", error_message=str(exc), state="idle")
        update_queue.put({"type": "error", "error_message": str(exc)})
    finally:
        update_queue.put({"type": "completed"})
