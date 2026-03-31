"""Aggregate metrics for UI and exported summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from taboo_arena.logging.schemas import RoundSummaryRecord


def compute_summary_metrics(
    round_summaries: list[RoundSummaryRecord],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute headline benchmark metrics."""
    rounds_played = len(round_summaries)
    solved_rounds = [row for row in round_summaries if row.solved]
    attempt_metrics = _attempt_metrics(solved_rounds, prefix="Solve on")

    total_repairs = sum(row.total_clue_repairs for row in round_summaries)
    first_pass = sum(1 for row in round_summaries if row.first_clue_passed_without_repair)
    repaired_success = sum(1 for row in round_summaries if row.clue_repaired_successfully)
    clue_failures = sum(1 for row in round_summaries if row.clue_not_repaired)

    logical_validations = [event for event in events if event.get("event_type") == "logical_validation_completed"]
    llm_validations = [event for event in events if event.get("event_type") == "llm_validation_completed"]
    disagreements = [event for event in llm_validations if bool(event.get("judge_disagreement"))]

    latency_by_role: dict[str, list[float]] = defaultdict(list)
    for event in events:
        role = event.get("role")
        latency = event.get("latency_ms")
        if role and isinstance(latency, (int, float)):
            latency_by_role[str(role)].append(float(latency))

    wrong_guesses_before_success = [
        max(row.total_visible_guesses_made - 1, 0) for row in solved_rounds if row.total_visible_guesses_made > 0
    ]

    return {
        "rounds_played": rounds_played,
        "solve_rate_within_3": _ratio(len(solved_rounds), rounds_played),
        "solve_on_attempt_1_rate": _ratio(attempt_metrics["attempt_1_count"], rounds_played),
        "solve_on_attempt_2_rate": _ratio(attempt_metrics["attempt_2_count"], rounds_played),
        "solve_on_attempt_3_rate": _ratio(attempt_metrics["attempt_3_bucket_count"], rounds_played),
        "solve_on_attempt_3_label": attempt_metrics["attempt_3_bucket_label"],
        "max_solved_on_attempt": attempt_metrics["max_solved_on_attempt"],
        "average_wrong_guesses_before_success": _average(wrong_guesses_before_success),
        "first_draft_clue_pass_rate": _ratio(first_pass, rounds_played),
        "repaired_clue_success_rate": _ratio(repaired_success, rounds_played),
        "clue_total_failure_rate": _ratio(clue_failures, rounds_played),
        "average_repairs_per_round": _average([row.total_clue_repairs for row in round_summaries]),
        "logical_fail_rate": _ratio(
            sum(1 for event in logical_validations if event.get("logical_verdict") == "fail"),
            len(logical_validations),
        ),
        "llm_judge_fail_rate": _ratio(
            sum(1 for event in llm_validations if event.get("llm_judge_verdict") == "fail"),
            len(llm_validations),
        ),
        "judge_disagreement_rate": _ratio(len(disagreements), len(llm_validations)),
        "average_latency_per_role": {
            role: round(_average(values), 2) for role, values in latency_by_role.items()
        },
        "total_clue_repairs": total_repairs,
    }


def compute_benchmark_run_metrics(round_summaries: list[RoundSummaryRecord]) -> dict[str, Any]:
    """Compute benchmark-level KPIs where each selected card is played once."""
    rounds_played = len(round_summaries)
    cards_solved_total = sum(1 for row in round_summaries if row.solved)
    attempt_metrics = _attempt_metrics(round_summaries, prefix="Attempt")

    failure_reasons = defaultdict(int)
    for row in round_summaries:
        if not row.solved:
            failure_reasons[row.terminal_reason] += 1

    clue_not_repaired_count = int(failure_reasons.get("clue_not_repaired", 0))
    max_guess_attempts_reached_count = int(
        failure_reasons.get("max_attempts_reached", 0)
        + failure_reasons.get("max_guess_attempts_reached", 0)
    )
    average_latency_ms = _average([row.total_latency_ms for row in round_summaries])

    trio = round_summaries[0] if round_summaries else None
    return {
        "cards_played_total": rounds_played,
        "cards_solved_total": cards_solved_total,
        "card_solve_rate": _ratio(cards_solved_total, rounds_played),
        "solved_on_attempt_1_count": attempt_metrics["attempt_1_count"],
        "solved_on_attempt_2_count": attempt_metrics["attempt_2_count"],
        "solved_on_attempt_3_count": attempt_metrics["attempt_3_bucket_count"],
        "solved_on_attempt_3_label": attempt_metrics["attempt_3_bucket_label"],
        "max_solved_on_attempt": attempt_metrics["max_solved_on_attempt"],
        "clue_not_repaired_count": clue_not_repaired_count,
        "max_guess_attempts_reached_count": max_guess_attempts_reached_count,
        "average_repairs_per_card": _average([row.total_clue_repairs for row in round_summaries]),
        "average_latency_ms": average_latency_ms,
        "cluer_model_id": "" if trio is None else trio.cluer_model_id,
        "guesser_model_id": "" if trio is None else trio.guesser_model_id,
        "judge_model_id": "" if trio is None else trio.judge_model_id,
        "failure_reason_breakdown": dict(failure_reasons),
    }


def compute_role_analytics(
    round_summaries: list[RoundSummaryRecord],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute role-level analytics rows for display or export."""
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[RoundSummaryRecord]] = defaultdict(list)
    for row in round_summaries:
        grouped[(row.cluer_model_id, row.guesser_model_id, row.judge_model_id)].append(row)

    logical_fail_counts: dict[str, int] = defaultdict(int)
    llm_fail_counts: dict[str, int] = defaultdict(int)
    judge_rounds: dict[str, int] = defaultdict(int)
    for event in events:
        if event.get("event_type") == "logical_validation_completed":
            logical_fail_counts[str(event.get("cluer_model_id", ""))] += int(event.get("logical_verdict") == "fail")
        if event.get("event_type") == "llm_validation_completed":
            judge_model_id = str(event.get("judge_model_id", ""))
            judge_rounds[judge_model_id] += 1
            llm_fail_counts[judge_model_id] += int(event.get("llm_judge_verdict") == "fail")

    for (cluer_model_id, guesser_model_id, judge_model_id), group in grouped.items():
        rows.append(
            {
                "cluer_model_id": cluer_model_id,
                "guesser_model_id": guesser_model_id,
                "judge_model_id": judge_model_id,
                "rounds": len(group),
                "solve_rate_within_3": _ratio(sum(1 for row in group if row.solved), len(group)),
                "average_repairs_per_round": _average([row.total_clue_repairs for row in group]),
                "clue_failure_rate": _ratio(sum(1 for row in group if row.clue_not_repaired), len(group)),
                "judge_fail_rate_for_model": _ratio(llm_fail_counts[judge_model_id], judge_rounds[judge_model_id]),
                "logical_fail_count_for_cluer": logical_fail_counts[cluer_model_id],
            }
        )
    return rows


def _attempt_metrics(
    round_summaries: list[RoundSummaryRecord],
    *,
    prefix: str,
) -> dict[str, int | str]:
    """Bucket solve counts so UIs keep working when max attempts exceeds three."""
    attempt_counts: dict[int, int] = defaultdict(int)
    max_solved_on_attempt = 0
    for row in round_summaries:
        if row.solved_on_attempt is None:
            continue
        solved_on_attempt = int(row.solved_on_attempt)
        if solved_on_attempt <= 0:
            continue
        attempt_counts[solved_on_attempt] += 1
        max_solved_on_attempt = max(max_solved_on_attempt, solved_on_attempt)

    third_bucket_is_3_plus = max_solved_on_attempt > 3
    third_bucket_count = (
        sum(count for attempt_no, count in attempt_counts.items() if attempt_no >= 3)
        if third_bucket_is_3_plus
        else attempt_counts.get(3, 0)
    )
    third_bucket_label = f"{prefix} 3+" if third_bucket_is_3_plus else f"{prefix} 3"
    return {
        "attempt_1_count": attempt_counts.get(1, 0),
        "attempt_2_count": attempt_counts.get(2, 0),
        "attempt_3_bucket_count": third_bucket_count,
        "attempt_3_bucket_label": third_bucket_label,
        "max_solved_on_attempt": max_solved_on_attempt,
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _average(values: list[int] | list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
