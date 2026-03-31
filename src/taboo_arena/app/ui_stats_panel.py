"""Metrics panel rendering helpers."""

from __future__ import annotations

import html
from typing import Any, cast

import streamlit as st

from taboo_arena.analytics import compute_benchmark_run_metrics, compute_summary_metrics
from taboo_arena.app.session_facade import SessionFacade
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.models import ModelEntry, ModelManager


def render_round_pulse_inline(
    *,
    session: SessionFacade,
    logger: RunLogger | None,
    selected_models: dict[str, ModelEntry],
    model_manager: ModelManager,
    current_events: list[dict[str, Any]],
    allow_export_widget: bool = True,
) -> None:
    """Render round and session metrics while preserving existing semantics."""
    session_round_summaries, session_events = session_metric_inputs(
        session=session,
        current_logger=logger,
        current_events=current_events,
    )
    if logger is None and not session_round_summaries:
        st.markdown(
            "<div class='subtle-note'>Run metrics and export appear here after the first round.</div>",
            unsafe_allow_html=True,
        )
        return

    shared_models = len({entry.repo_id for entry in selected_models.values()})
    current_round_summaries = [] if logger is None else logger.snapshot_round_summaries()

    round_tab, session_tab = st.tabs(["Round metrics", "Session metrics"])
    with round_tab:
        if not current_round_summaries and current_events:
            render_live_round_metric_groups(
                events=current_events,
                shared_models=shared_models,
                loaded_model_count=len(model_manager.loaded_models),
            )
        elif not current_round_summaries:
            st.caption("Round metrics will populate after the current round finishes.")
        else:
            render_metric_groups(
                round_summaries=current_round_summaries,
                events=current_events,
                shared_models=shared_models,
                loaded_model_count=len(model_manager.loaded_models),
            )

    with session_tab:
        if not session_round_summaries:
            st.caption("Session metrics will accumulate here as you play rounds.")
        else:
            render_metric_groups(
                round_summaries=session_round_summaries,
                events=session_events,
                shared_models=shared_models,
                loaded_model_count=len(model_manager.loaded_models),
            )

    warnings = [] if logger is None else logger.latest_errors()
    if warnings:
        st.warning(warnings[-1])

    st.caption(
        f"{shared_models} unique model repo(s) selected. "
        f"Loaded now: {len(model_manager.loaded_models)}."
    )
    if logger is not None and allow_export_widget:
        st.download_button(
            "Export",
            data=logger.export_run_archive(),
            file_name=f"{logger.run_id}.zip",
            mime="application/zip",
            width="stretch",
        )


def render_live_round_pulse_inline(
    *,
    session: SessionFacade,
    logger: RunLogger | None,
    selected_models: dict[str, ModelEntry],
    model_manager: ModelManager,
    current_events: list[dict[str, Any]],
) -> None:
    """Render a smoother live-only metrics view without interactive widgets."""
    session_round_summaries, session_events = session_metric_inputs(
        session=session,
        current_logger=logger,
        current_events=current_events,
    )
    shared_models = len({entry.repo_id for entry in selected_models.values()})

    st.markdown("<div class='metrics-live-heading'>Round metrics</div>", unsafe_allow_html=True)
    if current_events:
        render_live_round_metric_groups(
            events=current_events,
            shared_models=shared_models,
            loaded_model_count=len(model_manager.loaded_models),
        )
    else:
        st.caption("Round metrics will populate after the current round starts.")

    if session_round_summaries:
        st.markdown("<div class='metrics-live-heading'>Session snapshot</div>", unsafe_allow_html=True)
        render_metric_groups(
            round_summaries=session_round_summaries,
            events=session_events,
            shared_models=shared_models,
            loaded_model_count=len(model_manager.loaded_models),
        )

    warnings = [] if logger is None else logger.latest_errors()
    if warnings:
        st.warning(warnings[-1])

    st.caption(
        f"{shared_models} unique model repo(s) selected. "
        f"Loaded now: {len(model_manager.loaded_models)}."
    )


def render_live_benchmark_inline(
    *,
    progress: dict[str, Any] | None,
    logger: RunLogger | None,
) -> None:
    """Render live benchmark progress in the metrics panel."""
    if not progress:
        st.caption("Benchmark progress will appear after the first card starts.")
        return
    completed_cards = int(progress.get("completed_cards", 0) or 0)
    total_cards = int(progress.get("total_cards", 0) or 0)
    solved_so_far = int(progress.get("solved_so_far", 0) or 0)
    solve_rate = float(progress.get("card_solve_rate_so_far", 0.0) or 0.0)
    elapsed_seconds = float(progress.get("elapsed_seconds", 0.0) or 0.0)
    current_card_id = str(progress.get("current_card_id", "n/a") or "n/a")
    current_target = str(progress.get("current_target", "") or "").strip()
    current_card_label = (
        f"{current_card_id} • {current_target}"
        if current_target and current_card_id != "n/a"
        else current_card_id
    )
    grouped_cards = [
        (
            "Benchmark",
            [
                ("Completed", f"{completed_cards} / {total_cards}"),
                ("Solved", str(solved_so_far)),
                ("Solve rate", format_ratio(solve_rate)),
                ("Elapsed", f"{elapsed_seconds:.1f}s"),
            ],
        ),
        (
            "Current card",
            [
                ("Card", current_card_label),
                ("Run mode", "benchmark"),
                ("Run id", "n/a" if logger is None else logger.run_id),
                ("Status", "running"),
            ],
        ),
    ]
    metrics_html = "".join(metric_group_html(title, items) for title, items in grouped_cards)
    st.markdown(metrics_html, unsafe_allow_html=True)
    warnings = [] if logger is None else logger.latest_errors()
    if warnings:
        st.warning(warnings[-1])


def render_benchmark_summary_inline(
    *,
    summary: dict[str, Any],
    logger: RunLogger | None,
) -> None:
    """Render the final benchmark KPI summary."""
    failure_breakdown = summary.get("failure_reason_breakdown", {})
    if not isinstance(failure_breakdown, dict):
        failure_breakdown = {}
    attempt_3_label = str(summary.get("solved_on_attempt_3_label", "Attempt 3"))
    grouped_cards = [
        (
            "Model trio",
            [
                ("Cluer", str(summary.get("cluer_model_id", "n/a"))),
                ("Guesser", str(summary.get("guesser_model_id", "n/a"))),
                ("Judge", str(summary.get("judge_model_id", "n/a"))),
                ("Run id", "n/a" if logger is None else logger.run_id),
            ],
        ),
        (
            "Primary KPI",
            [
                (
                    "Solved cards",
                    f"{int(summary.get('cards_solved_total', 0) or 0)} / {int(summary.get('cards_played_total', 0) or 0)}",
                ),
                ("Solve rate", format_ratio(summary.get("card_solve_rate", 0.0))),
                ("Attempt 1", str(int(summary.get("solved_on_attempt_1_count", 0) or 0))),
                ("Attempt 2", str(int(summary.get("solved_on_attempt_2_count", 0) or 0))),
            ],
        ),
        (
            "Failures",
            [
                (attempt_3_label, str(int(summary.get("solved_on_attempt_3_count", 0) or 0))),
                ("Clue not repaired", str(int(summary.get("clue_not_repaired_count", 0) or 0))),
                (
                    "Max attempts reached",
                    str(int(summary.get("max_guess_attempts_reached_count", 0) or 0)),
                ),
                ("Avg repairs", f"{float(summary.get('average_repairs_per_card', 0.0) or 0.0):.2f}"),
            ],
        ),
        (
            "Runtime",
            [
                ("Avg latency", format_ms(summary.get("average_latency_ms"))),
                ("Elapsed", f"{float(summary.get('elapsed_seconds', 0.0) or 0.0):.1f}s"),
                ("Failures seen", str(sum(int(value) for value in failure_breakdown.values()))),
                ("Logs", "Open run directory"),
            ],
        ),
    ]
    metrics_html = "".join(metric_group_html(title, items) for title, items in grouped_cards)
    st.markdown(metrics_html, unsafe_allow_html=True)
    if logger is not None:
        st.caption(f"Artifacts: {logger.run_dir}")
        st.download_button(
            "Export",
            data=logger.export_run_archive(),
            file_name=f"{logger.run_id}.zip",
            mime="application/zip",
            width="stretch",
        )


def render_live_round_metric_groups(
    *,
    events: list[dict[str, Any]],
    shared_models: int,
    loaded_model_count: int,
) -> None:
    """Render live round metrics derived directly from events."""
    if not events:
        st.caption("Round metrics will populate after the current round starts.")
        return

    last_event = events[-1]
    clue_drafts = [event for event in events if event.get("event_type") == "clue_draft_generated"]
    clue_repairs = [event for event in events if event.get("event_type") == "clue_internal_retry_requested"]
    repair_requests = [event for event in events if event.get("event_type") == "clue_repair_requested"]
    guess_hidden_retries = [
        event for event in events if event.get("event_type") == "guess_hidden_retry_requested"
    ]
    guesses = [
        event
        for event in events
        if event.get("event_type") == "guess_generated" and str(event.get("visible_guess_text", "")).strip()
    ]
    warnings = [
        event
        for event in events
        if event.get("event_type") == "llm_validation_completed"
        and event.get("final_judge_verdict") == "pass_with_warning"
    ]
    latencies = [
        float(cast(int | float, event.get("latency_ms")))
        for event in events
        if isinstance(event.get("latency_ms"), (int, float))
    ]
    grouped_cards = [
        (
            "Live overview",
            [
                ("State", str(last_event.get("state", "starting"))),
                ("Attempt", str(last_event.get("attempt_no", 0) or 0)),
                ("Cluer repairs", str(len(clue_repairs))),
                ("Loaded now", str(loaded_model_count)),
            ],
        ),
        (
            "Live flow",
            [
                ("Clues drafted", str(len(clue_drafts))),
                ("Judge rejects", str(len(repair_requests))),
                ("Guesses made", str(len(guesses))),
                ("Guesser retries", str(len(guess_hidden_retries))),
            ],
        ),
        (
            "Runtime",
            [
                ("Events", str(len(events))),
                ("Last event", str(last_event.get("event_type", "n/a"))),
                ("Latency seen", format_ms(sum(latencies)) if latencies else "n/a"),
                ("Judge warnings", str(len(warnings))),
            ],
        ),
    ]
    metrics_html = "".join(metric_group_html(title, items) for title, items in grouped_cards)
    st.markdown(metrics_html, unsafe_allow_html=True)


def render_metric_groups(
    *,
    round_summaries: list[Any],
    events: list[dict[str, Any]],
    shared_models: int,
    loaded_model_count: int,
) -> None:
    """Render summary metrics from round summaries and event history."""
    summary = compute_summary_metrics(round_summaries, events)
    latency_summary = latency_summary_rows(summary, round_summaries)
    attempt_3_label = str(summary.get("solve_on_attempt_3_label", "Solve on 3"))
    grouped_cards = [
        (
            "Overview",
            [
                ("Rounds", str(summary["rounds_played"])),
                ("Solve rate", format_ratio(summary["solve_rate_within_3"])),
                ("Loaded now", str(loaded_model_count)),
                ("Unique repos", str(shared_models)),
            ],
        ),
        (
            "Clue Flow",
            [
                ("First draft pass", format_ratio(summary["first_draft_clue_pass_rate"])),
                ("Repair success", format_ratio(summary["repaired_clue_success_rate"])),
                ("Clue fail rate", format_ratio(summary["clue_total_failure_rate"])),
                ("Avg repairs", f"{float(summary['average_repairs_per_round']):.2f}"),
            ],
        ),
        (
            "Attempts",
            [
                ("Solve on 1", format_ratio(summary["solve_on_attempt_1_rate"])),
                ("Solve on 2", format_ratio(summary["solve_on_attempt_2_rate"])),
                (attempt_3_label, format_ratio(summary["solve_on_attempt_3_rate"])),
                ("Wrong guesses", f"{float(summary['average_wrong_guesses_before_success']):.2f}"),
            ],
        ),
        (
            "Judging",
            [
                ("Logical fail", format_ratio(summary["logical_fail_rate"])),
                ("LLM fail", format_ratio(summary["llm_judge_fail_rate"])),
                ("Disagreement", format_ratio(summary["judge_disagreement_rate"])),
                ("Total repairs", str(int(summary["total_clue_repairs"]))),
            ],
        ),
        (
            "Latency",
            [
                ("Cluer avg", latency_summary["cluer"]),
                ("Guesser avg", latency_summary["guesser"]),
                ("Judge avg", latency_summary["judge"]),
                ("Round avg", latency_summary["round"]),
            ],
        ),
    ]
    metrics_html = "".join(metric_group_html(title, items) for title, items in grouped_cards)
    st.markdown(metrics_html, unsafe_allow_html=True)


def metric_group_html(title: str, items: list[tuple[str, str]]) -> str:
    """Render one metrics group to HTML."""
    cards_html = "".join(
        (
            "<div class='pulse-card'>"
            f"<div class='pulse-card-label'>{html.escape(label)}</div>"
            f"<div class='pulse-card-value'>{html.escape(value)}</div>"
            "</div>"
        )
        for label, value in items
    )
    return (
        "<div class='metrics-group'>"
        f"<div class='metrics-group-title'>{html.escape(title)}</div>"
        f"<div class='pulse-grid'>{cards_html}</div>"
        "</div>"
    )


def session_metric_inputs(
    *,
    session: SessionFacade,
    current_logger: RunLogger | None,
    current_events: list[dict[str, Any]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Return archived plus current metrics inputs."""
    archived_summaries = session.session_history_round_summaries
    archived_events = session.transcript_history_events
    if current_logger is None:
        return archived_summaries, archived_events
    return [
        *archived_summaries,
        *current_logger.snapshot_round_summaries(),
    ], [
        *archived_events,
        *current_events,
    ]


def latency_summary_rows(summary: dict[str, Any], round_summaries: list[Any]) -> dict[str, str]:
    """Render average latency rows from summary data."""
    role_latencies = summary.get("average_latency_per_role", {})
    round_latencies = [float(row.total_latency_ms) for row in round_summaries]
    round_average = sum(round_latencies) / len(round_latencies) if round_latencies else 0.0
    return {
        "cluer": format_ms(role_latencies.get("cluer")),
        "guesser": format_ms(role_latencies.get("guesser")),
        "judge": format_ms(role_latencies.get("judge")),
        "round": format_ms(round_average if round_latencies else None),
    }


def format_ratio(value: Any) -> str:
    """Render a ratio as a percentage."""
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def format_ms(value: Any) -> str:
    """Render a latency value in milliseconds."""
    try:
        return f"{float(value):.0f} ms"
    except (TypeError, ValueError):
        return "n/a"
