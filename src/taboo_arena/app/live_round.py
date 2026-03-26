"""Thin Streamlit-facing adapter over the canonical round stepper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.config import RunSettings
from taboo_arena.engine.round_session import (
    RoundPhase,
    RoundResult,
    RoundSessionState,
    RoundStepper,
)
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.models.registry import ModelEntry

ControllerPhase = Literal[
    "clue_prepare",
    "clue_ready",
    "clue_generate",
    "judge_prepare",
    "judge_ready",
    "judge_generate",
    "guess_prepare",
    "guess_ready",
    "guess_generate",
    "finished",
]


@dataclass(slots=True)
class LiveRoundController:
    """Minimal controller facade that exposes canonical round state to the UI."""

    stepper: RoundStepper

    @property
    def state(self) -> RoundSessionState:
        return self.stepper.state

    @property
    def logger(self) -> RunLogger:
        return self.state.logger

    @property
    def settings(self) -> RunSettings:
        return self.state.settings

    @property
    def card(self) -> CardRecord:
        return self.state.card

    @property
    def cluer_entry(self) -> ModelEntry:
        return self.state.cluer_entry

    @property
    def guesser_entry(self) -> ModelEntry:
        return self.state.guesser_entry

    @property
    def judge_entry(self) -> ModelEntry:
        return self.state.judge_entry

    @property
    def round_id(self) -> str:
        return self.state.round_id

    @property
    def runtime_policy(self) -> str:
        return self.state.runtime_policy

    @property
    def phase(self) -> ControllerPhase:
        return self.state.phase.value

    @property
    def result(self) -> RoundResult | None:
        return self.state.result

    @property
    def error_message(self) -> str | None:
        return self.state.error_message


def start_live_round(
    *,
    settings: RunSettings,
    model_manager: Any,
    logger: RunLogger,
    card: CardRecord,
    cluer_entry: ModelEntry,
    guesser_entry: ModelEntry,
    judge_entry: ModelEntry,
) -> LiveRoundController:
    """Create a new UI controller backed by the canonical round stepper."""
    return LiveRoundController(
        stepper=RoundStepper(
            model_manager=model_manager,
            logger=logger,
            settings=settings.model_copy(deep=True),
            card=card,
            cluer_entry=cluer_entry,
            guesser_entry=guesser_entry,
            judge_entry=judge_entry,
        )
    )


def advance_live_round(controller: LiveRoundController, *, model_manager: Any) -> None:
    """Advance the live controller by one canonical phase."""
    controller.stepper.model_manager = model_manager
    if controller.stepper.is_finished:
        return
    controller.stepper.step()


def is_live_round_finished(controller: LiveRoundController) -> bool:
    """Return whether the canonical round session is finished."""
    return controller.stepper.is_finished or controller.state.phase is RoundPhase.FINISHED
