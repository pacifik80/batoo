"""Background job helpers for Streamlit-triggered runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from threading import Thread
from time import time
from typing import Any, Literal

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.config import AppSettings
from taboo_arena.engine import RoundEngine, RoundResult
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.models import ModelEntry, ModelManager, ModelRegistry
from taboo_arena.utils.ids import new_batch_id


@dataclass(slots=True)
class ActiveJob:
    """One background run owned by the Streamlit session."""

    kind: Literal["single", "batch"]
    logger: RunLogger
    thread: Thread | None = None
    process: Any | None = None
    update_queue: Any | None = None
    worker_kind: Literal["thread", "process"] = "thread"
    started_at: float = field(default_factory=time)
    result: RoundResult | None = None
    batch_results: list[RoundResult] = field(default_factory=list)
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
        worker_kind="process",
    )
    process.start()
    return job


def start_batch_job(
    *,
    settings: AppSettings,
    model_manager: ModelManager,
    logger: RunLogger,
    registry: ModelRegistry,
    tasks: list[dict[str, str]],
    cards_by_id: dict[str, Any],
) -> ActiveJob:
    """Start a batch in a background thread."""
    job = ActiveJob(
        kind="batch",
        logger=logger,
        thread=Thread(target=_batch_target, daemon=True, args=()),
        worker_kind="thread",
    )

    def target() -> None:
        _batch_target(
            job=job,
            settings=settings,
            model_manager=model_manager,
            logger=logger,
            registry=registry,
            tasks=tasks,
            cards_by_id=cards_by_id,
        )

    job.thread = Thread(target=target, daemon=True, name="taboo-arena-batch")
    job.thread.start()
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


def _single_round_target(
    *,
    job: ActiveJob,
    settings: AppSettings,
    model_manager: ModelManager,
    logger: RunLogger,
    card: Any,
    cluer_entry: ModelEntry,
    guesser_entry: ModelEntry,
    judge_entry: ModelEntry,
) -> None:
    engine = RoundEngine(model_manager=model_manager, logger=logger, settings=settings.run)
    model_manager.logger = logger
    try:
        job.result = engine.play_round(
            card=card,
            cluer_entry=cluer_entry,
            guesser_entry=guesser_entry,
            judge_entry=judge_entry,
        )
    except Exception as exc:
        logger.emit("error", error_message=str(exc), state="idle")
        job.error_message = str(exc)
    finally:
        job.completed = True


def _batch_target(
    *,
    job: ActiveJob,
    settings: AppSettings,
    model_manager: ModelManager,
    logger: RunLogger,
    registry: ModelRegistry,
    tasks: list[dict[str, str]],
    cards_by_id: dict[str, Any],
) -> None:
    engine = RoundEngine(model_manager=model_manager, logger=logger, settings=settings.run)
    model_manager.logger = logger
    batch_id = new_batch_id()

    try:
        logger.emit(
            "batch_started",
            batch_id=batch_id,
            state="batch_running",
            seed=settings.run.random_seed,
            card_count=len(tasks),
            combinations=len(tasks),
        )
        for task in tasks:
            if job.stop_requested:
                logger.emit("stopped", batch_id=batch_id, state="stopped")
                logger.emit("batch_finished", batch_id=batch_id, state="stopped")
                break
            result = engine.play_round(
                card=cards_by_id[task["card_id"]],
                cluer_entry=registry.get(task["cluer_model_id"]),
                guesser_entry=registry.get(task["guesser_model_id"]),
                judge_entry=registry.get(task["judge_model_id"]),
                batch_id=batch_id,
            )
            job.batch_results.append(result)
        else:
            logger.emit("batch_finished", batch_id=batch_id, state="idle")
    except Exception as exc:
        logger.emit("error", error_message=str(exc), state="idle")
        job.error_message = str(exc)
    finally:
        job.completed = True
