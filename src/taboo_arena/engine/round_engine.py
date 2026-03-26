"""Thin synchronous wrapper over the canonical round stepper."""

from __future__ import annotations

from typing import Any

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.config import RunSettings
from taboo_arena.engine.round_session import RoundResult, RoundStepper
from taboo_arena.judge import LogicalValidator, NormalizedLLMJudge
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.models.registry import ModelEntry


class RoundEngine:
    """Run full single-round benchmark sessions via the canonical stepper."""

    def __init__(
        self,
        *,
        model_manager: Any,
        logger: RunLogger,
        settings: RunSettings,
        logical_validator: LogicalValidator | None = None,
        llm_judge: NormalizedLLMJudge | None = None,
    ) -> None:
        self.model_manager = model_manager
        self.logger = logger
        self.settings = settings
        self.logical_validator = logical_validator
        self.llm_judge = llm_judge

    def play_round(
        self,
        *,
        card: CardRecord,
        cluer_entry: ModelEntry,
        guesser_entry: ModelEntry,
        judge_entry: ModelEntry,
        batch_id: str | None = None,
        flush_artifacts: bool = True,
    ) -> RoundResult:
        """Run one round to completion without exposing stepwise phases."""
        stepper = RoundStepper(
            model_manager=self.model_manager,
            logger=self.logger,
            settings=self.settings,
            card=card,
            cluer_entry=cluer_entry,
            guesser_entry=guesser_entry,
            judge_entry=judge_entry,
            batch_id=batch_id,
            logical_validator=self.logical_validator,
            llm_judge=self.llm_judge,
        )
        return stepper.run_to_completion(flush_artifacts=flush_artifacts)
