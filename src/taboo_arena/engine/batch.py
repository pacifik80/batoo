"""Batch benchmark runner."""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable
from dataclasses import dataclass

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.engine.round_engine import RoundEngine, RoundResult
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.models.registry import ModelEntry
from taboo_arena.utils.ids import new_batch_id


@dataclass(slots=True)
class BatchSpec:
    """Batch execution parameters."""

    cluer_entries: list[ModelEntry]
    guesser_entries: list[ModelEntry]
    judge_entries: list[ModelEntry]
    cards: list[CardRecord]
    repeats_per_card: int = 1
    seed: int = 7


class BatchRunner:
    """Run many rounds across role combinations on the same main engine."""

    def __init__(self, engine: RoundEngine, logger: RunLogger) -> None:
        self.engine = engine
        self.logger = logger

    def run(
        self,
        spec: BatchSpec,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> list[RoundResult]:
        """Run the provided batch spec."""
        batch_id = new_batch_id()
        rng = random.Random(spec.seed)
        cards = list(spec.cards)
        rng.shuffle(cards)
        combinations = list(itertools.product(spec.cluer_entries, spec.guesser_entries, spec.judge_entries))
        scheduled_cards = cards * max(spec.repeats_per_card, 1)

        self.logger.emit(
            "batch_started",
            batch_id=batch_id,
            state="batch_running",
            seed=spec.seed,
            card_count=len(cards),
            combinations=len(combinations),
        )

        results: list[RoundResult] = []
        for cluer_entry, guesser_entry, judge_entry in combinations:
            for card in scheduled_cards:
                if stop_requested is not None and stop_requested():
                    self.logger.emit("stopped", batch_id=batch_id, state="stopped")
                    self.logger.emit("batch_finished", batch_id=batch_id, state="stopped")
                    return results
                results.append(
                    self.engine.play_round(
                        card=card,
                        cluer_entry=cluer_entry,
                        guesser_entry=guesser_entry,
                        judge_entry=judge_entry,
                        batch_id=batch_id,
                    )
                )

        self.logger.emit("batch_finished", batch_id=batch_id, state="idle")
        return results
