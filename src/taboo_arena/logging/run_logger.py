"""Machine-friendly run logging."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd

from taboo_arena.analytics.metrics import compute_summary_metrics
from taboo_arena.logging.schemas import RoundSummaryRecord
from taboo_arena.utils.ids import new_run_id
from taboo_arena.utils.paths import get_log_dir


class RunLogger:
    """Persist events and round summaries for downstream analytics."""

    def __init__(
        self,
        run_id: str | None = None,
        log_root: Path | None = None,
        *,
        console_trace: bool = True,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.run_id = run_id or new_run_id()
        self.log_root = (log_root or get_log_dir()).resolve()
        self.console_trace = console_trace
        self.event_callback = event_callback
        self.run_dir = self.log_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.rounds_csv_path = self.run_dir / "rounds.csv"
        self.rounds_parquet_path = self.run_dir / "rounds.parquet"
        self.summary_csv_path = self.run_dir / "summary.csv"
        self._lock = RLock()
        self.events: list[dict[str, Any]] = []
        self.round_summaries: list[RoundSummaryRecord] = []
        self.latest_prompt_by_role: dict[str, dict[str, Any]] = {}
        self.latest_response_by_role: dict[str, dict[str, Any]] = {}

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        """Record one event and append it to the JSONL log."""
        event = {
            "event_type": event_type,
            "run_id": self.run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            **fields,
        }
        with self._lock:
            self.events.append(event)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, default=str))
                handle.write("\n")
        if self.console_trace:
            self._console_print_event(event)
        if self.event_callback is not None:
            try:
                self.event_callback(event)
            except Exception:
                pass
        return event

    def record_round_summary(self, summary: RoundSummaryRecord) -> None:
        """Append a round summary and refresh derived artifacts."""
        with self._lock:
            self.round_summaries.append(summary)
        self.flush()

    def flush(self) -> None:
        """Write round tables and summary metrics."""
        round_summaries = self.snapshot_round_summaries()
        events = self.snapshot_events()
        rounds_df = pd.DataFrame([item.model_dump(mode="json") for item in round_summaries])
        if rounds_df.empty:
            rounds_df = pd.DataFrame(
                columns=[
                    "run_id",
                    "round_id",
                    "card_id",
                    "target",
                    "solved",
                    "solved_on_attempt",
                    "total_guess_attempts_used",
                    "total_clue_repairs",
                    "first_clue_passed_without_repair",
                    "clue_repaired_successfully",
                    "clue_not_repaired",
                    "terminal_reason",
                    "cluer_model_id",
                    "guesser_model_id",
                    "judge_model_id",
                    "total_latency_ms",
                ]
            )
        rounds_df.to_csv(self.rounds_csv_path, index=False)
        rounds_df.to_parquet(self.rounds_parquet_path, index=False)

        summary_rows = compute_summary_metrics(round_summaries, events)
        pd.DataFrame([summary_rows]).to_csv(self.summary_csv_path, index=False)

    def latest_errors(self, limit: int = 10) -> list[str]:
        """Return the newest error or warning messages."""
        messages: list[str] = []
        for event in reversed(self.snapshot_events()):
            if event.get("event_type") == "error":
                messages.append(str(event.get("error_message", "Unknown error")))
            elif event.get("final_judge_verdict") == "pass_with_warning":
                messages.append(str(event.get("llm_judge_reasons", "Judge warning")))
            if len(messages) >= limit:
                break
        return list(reversed(messages))

    def snapshot_events(self) -> list[dict[str, Any]]:
        """Return a stable copy of the current event stream."""
        with self._lock:
            return list(self.events)

    def snapshot_round_summaries(self) -> list[RoundSummaryRecord]:
        """Return a stable copy of the current round summaries."""
        with self._lock:
            return list(self.round_summaries)

    def ingest_event(self, event: dict[str, Any]) -> None:
        """Append an externally produced event to in-memory state."""
        with self._lock:
            self.events.append(dict(event))

    def ingest_round_summary(self, summary: RoundSummaryRecord) -> None:
        """Append an externally produced round summary to in-memory state."""
        with self._lock:
            self.round_summaries.append(summary)

    def export_run_archive(self) -> bytes:
        """Create a zip archive of the current run artifacts."""
        self.flush()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in [
                self.events_path,
                self.rounds_csv_path,
                self.rounds_parquet_path,
                self.summary_csv_path,
            ]:
                if path.exists():
                    archive.write(path, arcname=path.name)
        return buffer.getvalue()

    def trace_prompt(
        self,
        *,
        role: str,
        model_id: str,
        prompt_template_id: str,
        prompt: str,
        generation_params: dict[str, Any],
    ) -> None:
        """Print a rendered prompt to the console."""
        with self._lock:
            self.latest_prompt_by_role[role] = {
                "role": role,
                "model_id": model_id,
                "prompt_template_id": prompt_template_id,
                "prompt": prompt,
                "generation_params": generation_params,
            }
        if not self.console_trace:
            return
        lines = [
            f"[taboo-arena:prompt] role={role} model={model_id} template={prompt_template_id}",
            f"[taboo-arena:prompt] generation={json.dumps(generation_params, default=str)}",
            prompt,
            "[taboo-arena:end-prompt]",
        ]
        print("\n".join(lines), flush=True)

    def trace_response(
        self,
        *,
        role: str,
        model_id: str,
        text: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Print a model response to the console."""
        with self._lock:
            self.latest_response_by_role[role] = {
                "role": role,
                "model_id": model_id,
                "text": text,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        if not self.console_trace:
            return
        lines = [
            (
                f"[taboo-arena:response] role={role} model={model_id} "
                f"latency_ms={latency_ms} prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}"
            ),
            text,
            "[taboo-arena:end-response]",
        ]
        print("\n".join(lines), flush=True)

    def _console_print_event(self, event: dict[str, Any]) -> None:
        """Print one structured event to stdout."""
        print(f"[taboo-arena:event] {json.dumps(event, default=str)}", flush=True)
