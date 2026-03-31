from __future__ import annotations

from taboo_arena.analytics.metrics import compute_benchmark_run_metrics, compute_summary_metrics
from taboo_arena.logging.schemas import RoundSummaryRecord


def _summary(*, solved_on_attempt: int | None) -> RoundSummaryRecord:
    return RoundSummaryRecord(
        run_id="run_test",
        round_id="round_test",
        card_id="card_test",
        target="Target",
        solved=solved_on_attempt is not None,
        solved_on_attempt=solved_on_attempt,
        total_guess_attempts_used=0 if solved_on_attempt is None else solved_on_attempt,
        total_visible_guesses_made=0 if solved_on_attempt is None else solved_on_attempt,
        total_clue_repairs=0,
        first_clue_passed_without_repair=True,
        clue_repaired_successfully=False,
        clue_not_repaired=solved_on_attempt is None,
        terminal_reason="solved" if solved_on_attempt is not None else "max_attempts_reached",
        cluer_model_id="cluer",
        guesser_model_id="guesser",
        judge_model_id="judge",
        total_latency_ms=10.0,
    )


def test_compute_summary_metrics_handles_fourth_attempt_without_key_errors() -> None:
    metrics = compute_summary_metrics([_summary(solved_on_attempt=4)], [])

    assert metrics["solve_on_attempt_1_rate"] == 0.0
    assert metrics["solve_on_attempt_2_rate"] == 0.0
    assert metrics["solve_on_attempt_3_rate"] == 1.0
    assert metrics["solve_on_attempt_3_label"] == "Solve on 3+"
    assert metrics["max_solved_on_attempt"] == 4


def test_compute_benchmark_run_metrics_buckets_late_solves_into_attempt_3_plus() -> None:
    metrics = compute_benchmark_run_metrics([
        _summary(solved_on_attempt=3),
        _summary(solved_on_attempt=4),
    ])

    assert metrics["solved_on_attempt_1_count"] == 0
    assert metrics["solved_on_attempt_2_count"] == 0
    assert metrics["solved_on_attempt_3_count"] == 2
    assert metrics["solved_on_attempt_3_label"] == "Attempt 3+"
    assert metrics["max_solved_on_attempt"] == 4
