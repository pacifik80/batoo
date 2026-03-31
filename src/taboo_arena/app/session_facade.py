"""Typed access to Streamlit session state."""

from __future__ import annotations

import random
from typing import Any, cast

from taboo_arena.logging.run_logger import RunLogger


class SessionFacade:
    """Small typed facade over Streamlit's dict-like session state."""

    def __init__(self, state: Any) -> None:
        self._state = state

    @property
    def raw(self) -> Any:
        return self._state

    @property
    def active_job(self) -> Any | None:
        return self._state.get("active_job")

    @active_job.setter
    def active_job(self, value: Any | None) -> None:
        self._state["active_job"] = value

    @property
    def current_logger(self) -> RunLogger | None:
        return cast(RunLogger | None, self._state.get("current_logger"))

    @current_logger.setter
    def current_logger(self, value: RunLogger | None) -> None:
        self._state["current_logger"] = value

    @property
    def current_result(self) -> Any:
        return self._state.get("current_result")

    @current_result.setter
    def current_result(self, value: Any) -> None:
        self._state["current_result"] = value

    @property
    def current_error_message(self) -> str | None:
        value = self._state.get("current_error_message")
        return None if value is None else str(value)

    @current_error_message.setter
    def current_error_message(self, value: str | None) -> None:
        self._state["current_error_message"] = value

    @property
    def transcript_history_events(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self._state.get("transcript_history_events", []))

    @transcript_history_events.setter
    def transcript_history_events(self, value: list[dict[str, Any]]) -> None:
        self._state["transcript_history_events"] = value

    @property
    def transcript_history_run_ids(self) -> list[str]:
        return cast(list[str], self._state.get("transcript_history_run_ids", []))

    @transcript_history_run_ids.setter
    def transcript_history_run_ids(self, value: list[str]) -> None:
        self._state["transcript_history_run_ids"] = value

    @property
    def session_history_round_summaries(self) -> list[Any]:
        return list(self._state.get("session_history_round_summaries", []))

    @session_history_round_summaries.setter
    def session_history_round_summaries(self, value: list[Any]) -> None:
        self._state["session_history_round_summaries"] = value

    @property
    def resource_history(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self._state.get("resource_history", []))

    @resource_history.setter
    def resource_history(self, value: list[dict[str, Any]]) -> None:
        self._state["resource_history"] = value

    @property
    def resource_last_sample_at(self) -> float:
        return float(self._state.get("resource_last_sample_at", 0.0))

    @resource_last_sample_at.setter
    def resource_last_sample_at(self, value: float) -> None:
        self._state["resource_last_sample_at"] = float(value)

    @property
    def start_benchmark_clicked(self) -> bool:
        return bool(self._state.get("start_benchmark_clicked", False))

    @start_benchmark_clicked.setter
    def start_benchmark_clicked(self, value: bool) -> None:
        self._state["start_benchmark_clicked"] = bool(value)

    @property
    def benchmark_plan(self) -> Any:
        return self._state.get("benchmark_plan")

    @benchmark_plan.setter
    def benchmark_plan(self, value: Any) -> None:
        self._state["benchmark_plan"] = value

    @property
    def benchmark_progress(self) -> Any:
        return self._state.get("benchmark_progress")

    @benchmark_progress.setter
    def benchmark_progress(self, value: Any) -> None:
        self._state["benchmark_progress"] = value

    @property
    def benchmark_summary(self) -> Any:
        return self._state.get("benchmark_summary")

    @benchmark_summary.setter
    def benchmark_summary(self, value: Any) -> None:
        self._state["benchmark_summary"] = value

    @property
    def benchmark_error(self) -> str | None:
        value = self._state.get("benchmark_error")
        return None if value is None else str(value)

    @benchmark_error.setter
    def benchmark_error(self, value: str | None) -> None:
        self._state["benchmark_error"] = value

    @property
    def app_randomizer(self) -> random.Random:
        return cast(random.Random, self._state["app_randomizer"])
