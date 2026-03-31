"""Benchmark planning and orchestration over the canonical one-round engine."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from taboo_arena.analytics.metrics import compute_benchmark_run_metrics
from taboo_arena.cards.schemas import CardRecord
from taboo_arena.engine.round_engine import RoundEngine, RoundResult
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.models.registry import ModelEntry
from taboo_arena.utils.ids import new_benchmark_id

BenchmarkSelectionMode = Literal["all_eligible", "random_sample"]


class BenchmarkPlan(BaseModel):
    """A deterministic benchmark execution plan for one selected model trio."""

    benchmark_id: str
    run_mode: Literal["benchmark"] = "benchmark"
    cluer_model_id: str
    guesser_model_id: str
    judge_model_id: str
    seed: int
    card_selection_mode: BenchmarkSelectionMode
    requested_sample_size: int | None = None
    eligible_card_count: int
    eligible_card_ids: list[str] = Field(default_factory=list)
    played_card_ids: list[str] = Field(default_factory=list)
    selected_categories: list[str] = Field(default_factory=list)


class BenchmarkProgress(BaseModel):
    """Live benchmark progress surfaced back to the Streamlit session."""

    benchmark_id: str
    completed_cards: int
    total_cards: int
    current_card_id: str | None = None
    current_target: str | None = None
    solved_so_far: int = 0
    card_solve_rate_so_far: float = 0.0
    elapsed_seconds: float = 0.0


class BenchmarkSummary(BaseModel):
    """Final benchmark KPI summary for one model trio."""

    benchmark_id: str
    run_mode: Literal["benchmark"] = "benchmark"
    cluer_model_id: str
    guesser_model_id: str
    judge_model_id: str
    seed: int
    card_selection_mode: BenchmarkSelectionMode
    requested_sample_size: int | None = None
    eligible_card_count: int
    played_card_ids: list[str] = Field(default_factory=list)
    selected_categories: list[str] = Field(default_factory=list)
    cards_played_total: int
    cards_solved_total: int
    card_solve_rate: float
    solved_on_attempt_1_count: int
    solved_on_attempt_2_count: int
    solved_on_attempt_3_count: int
    solved_on_attempt_3_label: str = "Attempt 3"
    max_solved_on_attempt: int = 0
    clue_not_repaired_count: int
    max_guess_attempts_reached_count: int
    average_repairs_per_card: float
    average_latency_ms: float
    failure_reason_breakdown: dict[str, int] = Field(default_factory=dict)
    elapsed_seconds: float = 0.0


def create_benchmark_plan(
    *,
    eligible_cards: list[CardRecord],
    selected_categories: list[str],
    cluer_model_id: str,
    guesser_model_id: str,
    judge_model_id: str,
    seed: int,
    card_selection_mode: BenchmarkSelectionMode,
    requested_sample_size: int | None = None,
) -> BenchmarkPlan:
    """Create a deterministic benchmark card order for the current model trio."""
    if not eligible_cards:
        raise ValueError("Enable at least one category before starting a benchmark.")

    rng = random.Random(seed)
    ordered_eligible_cards = list(eligible_cards)

    if card_selection_mode == "random_sample":
        sample_size = int(requested_sample_size or 0)
        if sample_size <= 0:
            raise ValueError("Benchmark sample size must be at least 1.")
        if sample_size > len(ordered_eligible_cards):
            raise ValueError(
                f"Requested sample size {sample_size} exceeds the eligible pool of {len(ordered_eligible_cards)} cards."
            )
        selected_cards = rng.sample(ordered_eligible_cards, sample_size)
    else:
        rng.shuffle(ordered_eligible_cards)
        selected_cards = ordered_eligible_cards

    return BenchmarkPlan(
        benchmark_id=new_benchmark_id(),
        cluer_model_id=cluer_model_id,
        guesser_model_id=guesser_model_id,
        judge_model_id=judge_model_id,
        seed=seed,
        card_selection_mode=card_selection_mode,
        requested_sample_size=(
            int(requested_sample_size) if card_selection_mode == "random_sample" and requested_sample_size is not None else None
        ),
        eligible_card_count=len(eligible_cards),
        eligible_card_ids=[card.id for card in eligible_cards],
        played_card_ids=[card.id for card in selected_cards],
        selected_categories=list(selected_categories),
    )


def compute_benchmark_summary(
    *,
    plan: BenchmarkPlan,
    results: list[RoundResult],
    elapsed_seconds: float,
) -> BenchmarkSummary:
    """Aggregate the final benchmark KPIs from finished round results."""
    metrics = compute_benchmark_run_metrics(
        [result_to_round_summary(result) for result in results]
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
        cards_played_total=int(metrics["cards_played_total"]),
        cards_solved_total=int(metrics["cards_solved_total"]),
        card_solve_rate=float(metrics["card_solve_rate"]),
        solved_on_attempt_1_count=int(metrics["solved_on_attempt_1_count"]),
        solved_on_attempt_2_count=int(metrics["solved_on_attempt_2_count"]),
        solved_on_attempt_3_count=int(metrics["solved_on_attempt_3_count"]),
        solved_on_attempt_3_label=str(metrics.get("solved_on_attempt_3_label", "Attempt 3")),
        max_solved_on_attempt=int(metrics.get("max_solved_on_attempt", 0)),
        clue_not_repaired_count=int(metrics["clue_not_repaired_count"]),
        max_guess_attempts_reached_count=int(metrics["max_guess_attempts_reached_count"]),
        average_repairs_per_card=float(metrics["average_repairs_per_card"]),
        average_latency_ms=float(metrics["average_latency_ms"]),
        failure_reason_breakdown=dict(metrics["failure_reason_breakdown"]),
        elapsed_seconds=round(elapsed_seconds, 2),
    )


class BenchmarkRunner:
    """Run a planned benchmark over the canonical one-round engine."""

    def __init__(self, engine: RoundEngine, logger: RunLogger) -> None:
        self.engine = engine
        self.logger = logger

    def run_plan(
        self,
        *,
        plan: BenchmarkPlan,
        cards_by_id: dict[str, CardRecord],
        cluer_entry: ModelEntry,
        guesser_entry: ModelEntry,
        judge_entry: ModelEntry,
        stop_requested: Callable[[], bool] | None = None,
        progress_callback: Callable[[BenchmarkProgress], None] | None = None,
        round_completed_callback: Callable[[RoundResult], None] | None = None,
    ) -> BenchmarkSummary:
        """Run every card in the benchmark plan once using the selected model trio."""
        started_at = time.monotonic()
        self.logger.set_run_metadata(
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
        self.logger.write_json_artifact("benchmark_plan.json", plan.model_dump(mode="json"))
        self.logger.emit(
            "benchmark_started",
            benchmark_id=plan.benchmark_id,
            state="benchmark_running",
            run_mode="benchmark",
            cluer_model_id=plan.cluer_model_id,
            guesser_model_id=plan.guesser_model_id,
            judge_model_id=plan.judge_model_id,
            seed=plan.seed,
            card_selection_mode=plan.card_selection_mode,
            requested_sample_size=plan.requested_sample_size,
            eligible_card_count=plan.eligible_card_count,
            total_cards=len(plan.played_card_ids),
            selected_categories=list(plan.selected_categories),
            played_card_ids=list(plan.played_card_ids),
        )

        results: list[RoundResult] = []
        solved_so_far = 0
        total_cards = len(plan.played_card_ids)
        if progress_callback is not None:
            first_card = cards_by_id.get(plan.played_card_ids[0]) if plan.played_card_ids else None
            progress_callback(
                BenchmarkProgress(
                    benchmark_id=plan.benchmark_id,
                    completed_cards=0,
                    total_cards=total_cards,
                    current_card_id=plan.played_card_ids[0] if plan.played_card_ids else None,
                    current_target=None if first_card is None else first_card.target,
                    solved_so_far=0,
                    card_solve_rate_so_far=0.0,
                    elapsed_seconds=0.0,
                )
            )

        for card_index, card_id in enumerate(plan.played_card_ids, start=1):
            if stop_requested is not None and stop_requested():
                self.logger.emit(
                    "benchmark_stopped",
                    benchmark_id=plan.benchmark_id,
                    state="stopped",
                    completed_cards=len(results),
                    total_cards=total_cards,
                    solved_so_far=solved_so_far,
                )
                break

            self.logger.emit(
                "benchmark_card_started",
                benchmark_id=plan.benchmark_id,
                state="benchmark_running",
                card_index=card_index,
                total_cards=total_cards,
                current_card_id=card_id,
                target=cards_by_id[card_id].target,
                solved_so_far=solved_so_far,
            )
            if progress_callback is not None:
                progress_callback(
                    BenchmarkProgress(
                        benchmark_id=plan.benchmark_id,
                        completed_cards=len(results),
                        total_cards=total_cards,
                        current_card_id=card_id,
                        current_target=cards_by_id[card_id].target,
                        solved_so_far=solved_so_far,
                        card_solve_rate_so_far=_ratio(solved_so_far, max(len(results), 1)) if results else 0.0,
                        elapsed_seconds=round(time.monotonic() - started_at, 2),
                    )
                )

            result = self.engine.play_round(
                card=cards_by_id[card_id],
                cluer_entry=cluer_entry,
                guesser_entry=guesser_entry,
                judge_entry=judge_entry,
                batch_id=plan.benchmark_id,
                flush_artifacts=False,
            )
            results.append(result)
            solved_so_far += int(result.solved)
            if round_completed_callback is not None:
                round_completed_callback(result)
            self.logger.flush()

            progress = BenchmarkProgress(
                benchmark_id=plan.benchmark_id,
                completed_cards=len(results),
                total_cards=total_cards,
                current_card_id=card_id,
                current_target=cards_by_id[card_id].target,
                solved_so_far=solved_so_far,
                card_solve_rate_so_far=_ratio(solved_so_far, len(results)),
                elapsed_seconds=round(time.monotonic() - started_at, 2),
            )
            self.logger.emit(
                "benchmark_progress",
                benchmark_id=plan.benchmark_id,
                state="benchmark_running",
                completed_cards=progress.completed_cards,
                total_cards=progress.total_cards,
                current_card_id=progress.current_card_id,
                target=progress.current_target,
                solved_so_far=progress.solved_so_far,
                card_solve_rate_so_far=progress.card_solve_rate_so_far,
                elapsed_seconds=progress.elapsed_seconds,
            )
            if progress_callback is not None:
                progress_callback(progress)

        summary = compute_benchmark_summary(
            plan=plan,
            results=results,
            elapsed_seconds=time.monotonic() - started_at,
        )
        self.logger.emit(
            "benchmark_finished",
            benchmark_id=plan.benchmark_id,
            state="idle",
            cards_played_total=summary.cards_played_total,
            cards_solved_total=summary.cards_solved_total,
            card_solve_rate=summary.card_solve_rate,
            elapsed_seconds=summary.elapsed_seconds,
        )
        self.logger.flush()
        return summary


def result_to_round_summary(result: RoundResult):
    """Convert a round result into a summary-like record for aggregation."""
    from taboo_arena.logging.schemas import RoundSummaryRecord

    return RoundSummaryRecord(
        run_id=result.run_id,
        round_id=result.round_id,
        card_id=result.card.id,
        target=result.card.target,
        solved=result.solved,
        solved_on_attempt=result.solved_on_attempt,
        total_guess_attempts_used=result.total_guess_attempts_used,
        total_visible_guesses_made=result.total_visible_guesses_made,
        total_clue_repairs=result.total_clue_repairs,
        first_clue_passed_without_repair=result.first_clue_passed_without_repair,
        clue_repaired_successfully=result.clue_repaired_successfully,
        clue_not_repaired=result.clue_not_repaired,
        terminal_reason=result.terminal_reason,
        cluer_model_id=result.cluer_model_id,
        guesser_model_id=result.guesser_model_id,
        judge_model_id=result.judge_model_id,
        total_latency_ms=result.total_latency_ms,
    )


def _ratio(numerator: int, denominator: int) -> float:
    """Return a rounded ratio while avoiding division by zero."""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)
