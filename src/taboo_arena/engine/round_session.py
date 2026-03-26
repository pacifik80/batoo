"""Canonical stepwise round execution shared by engine and UI adapters."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.config import RunSettings
from taboo_arena.engine.types import AttemptTrace, GuessTrace, RepairTrace
from taboo_arena.judge import (
    LogicalValidationResult,
    LogicalValidator,
    NormalizedLLMJudge,
    merge_judge_results,
)
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.logging.schemas import RoundSummaryRecord
from taboo_arena.models.registry import ModelEntry
from taboo_arena.prompts import build_cluer_messages, build_guesser_messages
from taboo_arena.utils.ids import new_round_id
from taboo_arena.utils.normalization import normalize_text, strip_punctuation


class RoundPhase(StrEnum):
    """One step in the canonical round state machine."""

    CLUE_PREPARE = "clue_prepare"
    CLUE_READY = "clue_ready"
    CLUE_GENERATE = "clue_generate"
    JUDGE_PREPARE = "judge_prepare"
    JUDGE_READY = "judge_ready"
    JUDGE_GENERATE = "judge_generate"
    GUESS_PREPARE = "guess_prepare"
    GUESS_READY = "guess_ready"
    GUESS_GENERATE = "guess_generate"
    FINISHED = "finished"


@dataclass(slots=True)
class RoundResult:
    """Result of a benchmark round.

    `total_clue_repairs` keeps the legacy exported meaning: total clue drafts attempted,
    including the first draft on each guess attempt.
    """

    run_id: str
    round_id: str
    card: CardRecord
    solved: bool
    solved_on_attempt: int | None
    total_guess_attempts_used: int
    total_clue_repairs: int
    first_clue_passed_without_repair: bool
    clue_repaired_successfully: bool
    clue_not_repaired: bool
    terminal_reason: str
    attempts: list[AttemptTrace]
    cluer_model_id: str
    guesser_model_id: str
    judge_model_id: str
    total_latency_ms: float


@dataclass(slots=True)
class RoundSessionState:
    """Mutable state for one canonical round session."""

    logger: RunLogger
    settings: RunSettings
    card: CardRecord
    cluer_entry: ModelEntry
    guesser_entry: ModelEntry
    judge_entry: ModelEntry
    round_id: str
    runtime_policy: str
    batch_id: str | None = None
    phase: RoundPhase = RoundPhase.CLUE_PREPARE
    attempt_no: int = 1
    repair_no: int = 1
    total_latency_ms: float = 0.0
    total_clue_drafts: int = 0
    total_repairs_after_first_failure: int = 0
    accepted_clues: list[str] = field(default_factory=list)
    rejected_clues: list[str] = field(default_factory=list)
    wrong_guesses: list[str] = field(default_factory=list)
    attempts: list[AttemptTrace] = field(default_factory=list)
    current_attempt_trace: AttemptTrace = field(default_factory=lambda: AttemptTrace(attempt_no=1))
    current_clue_text: str | None = None
    current_clue_normalized: str | None = None
    current_logical_result: LogicalValidationResult | None = None
    current_clue_latency_ms: float = 0.0
    current_clue_prompt_template_id: str = ""
    first_clue_passed_without_repair: bool = False
    clue_repaired_successfully: bool = False
    solved: bool = False
    solved_on_attempt: int | None = None
    terminal_reason: str = "max_guess_attempts_reached"
    last_rejection_feedback: str | None = None
    result: RoundResult | None = None
    error_message: str | None = None
    rng: random.Random = field(default_factory=random.Random)


class RoundStepper:
    """Drive one round phase by phase or until completion."""

    def __init__(
        self,
        *,
        model_manager: Any,
        logger: RunLogger,
        settings: RunSettings,
        card: CardRecord,
        cluer_entry: ModelEntry,
        guesser_entry: ModelEntry,
        judge_entry: ModelEntry,
        batch_id: str | None = None,
        logical_validator: LogicalValidator | None = None,
        llm_judge: NormalizedLLMJudge | None = None,
    ) -> None:
        self.model_manager = model_manager
        self.logger = logger
        self.settings = settings
        self.logical_validator = logical_validator or LogicalValidator(settings.logical_validator)
        self.llm_judge = llm_judge or NormalizedLLMJudge()
        runtime_policy = self.model_manager.resolve_runtime_policy(
            [cluer_entry, guesser_entry, judge_entry],
            requested_policy=settings.memory_policy,
            device_preference=settings.device_preference,
        )
        round_id = new_round_id()
        self.state = RoundSessionState(
            logger=logger,
            settings=settings,
            card=card,
            cluer_entry=cluer_entry,
            guesser_entry=guesser_entry,
            judge_entry=judge_entry,
            round_id=round_id,
            runtime_policy=runtime_policy,
            batch_id=batch_id,
            rng=random.Random(settings.random_seed),
        )
        self.logger.emit(
            "round_started",
            batch_id=batch_id,
            round_id=round_id,
            card_id=card.id,
            state="generating_clue",
            source_repo=card.source_repo,
            source_ref=card.source_ref,
            source_commit=card.source_commit,
            imported_language=card.lang,
            cluer_model_id=cluer_entry.id,
            guesser_model_id=guesser_entry.id,
            judge_model_id=judge_entry.id,
            cluer_backend=cluer_entry.backend,
            guesser_backend=guesser_entry.backend,
            judge_backend=judge_entry.backend,
            runtime_policy=runtime_policy,
            seed=settings.random_seed,
        )

    @property
    def is_finished(self) -> bool:
        """Return whether the session reached a terminal phase."""
        return self.state.phase is RoundPhase.FINISHED

    def step(self) -> RoundPhase:
        """Advance exactly one canonical phase."""
        if self.is_finished:
            return self.state.phase

        self.model_manager.logger = self.logger
        try:
            phase = self.state.phase
            if phase is RoundPhase.CLUE_PREPARE:
                self._emit_clue_started()
                self.state.phase = RoundPhase.CLUE_READY
            elif phase is RoundPhase.CLUE_READY:
                self.state.phase = RoundPhase.CLUE_GENERATE
            elif phase is RoundPhase.CLUE_GENERATE:
                self._generate_clue()
                self.state.phase = RoundPhase.JUDGE_PREPARE
            elif phase is RoundPhase.JUDGE_PREPARE:
                self._emit_judge_started()
                self.state.phase = RoundPhase.JUDGE_READY
            elif phase is RoundPhase.JUDGE_READY:
                self.state.phase = RoundPhase.JUDGE_GENERATE
            elif phase is RoundPhase.JUDGE_GENERATE:
                self._generate_judge_decision()
            elif phase is RoundPhase.GUESS_PREPARE:
                self._emit_guess_started()
                self.state.phase = RoundPhase.GUESS_READY
            elif phase is RoundPhase.GUESS_READY:
                self.state.phase = RoundPhase.GUESS_GENERATE
            elif phase is RoundPhase.GUESS_GENERATE:
                self._generate_guess()
        except Exception as exc:
            self.logger.emit("error", error_message=str(exc), state="idle")
            self.state.error_message = str(exc)
            self.state.phase = RoundPhase.FINISHED
        return self.state.phase

    def run_to_completion(self, *, flush_artifacts: bool = True) -> RoundResult:
        """Run the round until a result is produced."""
        while not self.is_finished:
            self.step()
        if self.state.result is None:
            error_message = self.state.error_message or "Round finished without a result."
            raise RuntimeError(error_message)
        if flush_artifacts:
            self.logger.flush()
        return self.state.result

    def build_result(self) -> RoundResult:
        """Return the finished round result."""
        if self.state.result is None:
            raise RuntimeError("Round result is not available yet.")
        return self.state.result

    def _emit_clue_started(self) -> None:
        state = self.state
        self.logger.emit(
            "clue_draft_started",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            role="cluer",
            state="generating_clue",
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
        )

    def _generate_clue(self) -> None:
        state = self.state
        clue_response = self.model_manager.generate(
            model_entry=state.cluer_entry,
            messages=build_cluer_messages(
                card=state.card,
                accepted_clues=state.accepted_clues,
                rejected_clues=state.rejected_clues,
                wrong_guesses=state.wrong_guesses,
                attempt_no=state.attempt_no,
                repair_no=state.repair_no,
                last_rejection_feedback=state.last_rejection_feedback,
            ),
            generation_params=state.settings.generation.cluer,
            runtime_policy=state.runtime_policy,
            device_preference=state.settings.device_preference,
            trace_role="cluer",
            banned_phrases=_cluer_banned_phrases(card=state.card, rejected_clues=state.rejected_clues),
        )
        clue_text_raw = _extract_single_line(clue_response.text)
        logical_result = self.logical_validator.validate(
            clue_text_raw,
            card=state.card,
            previous_accepted_clues=state.accepted_clues,
            previous_rejected_clues=state.rejected_clues,
        )
        state.total_latency_ms += clue_response.latency_ms
        state.total_clue_drafts += 1
        if state.repair_no > 1:
            state.total_repairs_after_first_failure += 1
        state.current_clue_text = clue_text_raw
        state.current_clue_normalized = logical_result.normalized_text
        state.current_logical_result = logical_result
        state.current_clue_latency_ms = clue_response.latency_ms
        state.current_clue_prompt_template_id = clue_response.prompt_template_id

        self.logger.emit(
            "clue_draft_generated",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            latency_ms=clue_response.latency_ms,
            prompt_tokens=clue_response.prompt_tokens,
            completion_tokens=clue_response.completion_tokens,
            clue_text_raw=clue_text_raw,
            clue_text_normalized=logical_result.normalized_text,
            prompt_template_id=clue_response.prompt_template_id,
            role="cluer",
            state="logical_validation",
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
            **_latest_prompt_fields(self.logger, "cluer"),
        )
        self.logger.emit(
            "logical_validation_completed",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            clue_text_raw=clue_text_raw,
            clue_text_normalized=logical_result.normalized_text,
            logical_verdict=logical_result.verdict,
            logical_violations=logical_result.violations,
            logical_matched_terms=logical_result.matched_terms,
            validator_version=logical_result.validator_version,
            role="cluer",
            state="llm_validation",
            cluer_model_id=state.cluer_entry.id,
        )

    def _emit_judge_started(self) -> None:
        state = self.state
        self.logger.emit(
            "clue_review_started",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            role="judge",
            state="llm_validation",
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
        )

    def _generate_judge_decision(self) -> None:
        state = self.state
        logical_result = state.current_logical_result
        if logical_result is None:
            raise RuntimeError("Judge step started without a logical validation result.")

        llm_result, judge_response = self.llm_judge.evaluate(
            model_manager=self.model_manager,
            model_entry=state.judge_entry,
            card=state.card,
            clue_draft=state.current_clue_text or "",
            accepted_clues=state.accepted_clues,
            wrong_guesses=state.wrong_guesses,
            attempt_no=state.attempt_no,
            generation_params=state.settings.generation.judge,
            runtime_policy=state.runtime_policy,
            device_preference=state.settings.device_preference,
        )
        merged = merge_judge_results(
            logical_result,
            llm_result,
            block_on_uncertain=state.settings.block_on_uncertain,
        )
        state.total_latency_ms += judge_response.latency_ms
        state.current_attempt_trace.repairs.append(
            RepairTrace(
                repair_no=state.repair_no,
                clue_text_raw=state.current_clue_text or "",
                clue_text_normalized=state.current_clue_normalized or "",
                logical_result=logical_result,
                llm_result=llm_result,
                merged_result=merged,
                latency_ms=state.current_clue_latency_ms + judge_response.latency_ms,
                prompt_template_id=state.current_clue_prompt_template_id,
            )
        )
        self.logger.emit(
            "llm_validation_completed",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            latency_ms=judge_response.latency_ms,
            prompt_tokens=judge_response.prompt_tokens,
            completion_tokens=judge_response.completion_tokens,
            llm_judge_verdict=llm_result.verdict,
            llm_judge_reasons=llm_result.reasons,
            llm_judge_suspicious_terms=llm_result.suspicious_terms,
            llm_judge_confidence=llm_result.confidence,
            final_judge_verdict=merged.final_verdict,
            judge_disagreement=merged.judge_disagreement,
            prompt_template_id=judge_response.prompt_template_id,
            role="judge",
            state="repairing_clue" if merged.final_verdict == "fail" else "generating_guess",
            judge_model_id=state.judge_entry.id,
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            **_latest_prompt_fields(self.logger, "judge"),
        )

        if merged.final_verdict != "fail":
            accepted_clue = state.current_clue_text or ""
            state.accepted_clues.append(accepted_clue)
            state.current_attempt_trace.accepted_clue = accepted_clue
            state.current_attempt_trace.accepted_repair_no = state.repair_no
            if state.attempt_no == 1 and state.repair_no == 1:
                state.first_clue_passed_without_repair = True
            if state.repair_no > 1:
                state.clue_repaired_successfully = True
            self.logger.emit(
                "clue_accepted",
                batch_id=state.batch_id,
                round_id=state.round_id,
                card_id=state.card.id,
                attempt_no=state.attempt_no,
                clue_repair_no=state.repair_no,
                clue_text_raw=accepted_clue,
                final_judge_verdict=merged.final_verdict,
                judge_disagreement=merged.judge_disagreement,
                state="generating_guess",
                cluer_model_id=state.cluer_entry.id,
                guesser_model_id=state.guesser_entry.id,
                judge_model_id=state.judge_entry.id,
            )
            state.last_rejection_feedback = None
            state.phase = RoundPhase.GUESS_PREPARE
            return

        self.logger.emit(
            "clue_repair_requested",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            final_judge_verdict=merged.final_verdict,
            logical_violations=logical_result.violations,
            logical_matched_terms=logical_result.matched_terms,
            llm_judge_verdict=llm_result.verdict,
            llm_judge_reasons=llm_result.reasons,
            judge_disagreement=merged.judge_disagreement,
            state="repairing_clue",
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
            **_latest_prompt_fields(self.logger, "judge"),
        )
        state.rejected_clues.append(state.current_clue_text or "")
        state.last_rejection_feedback = _format_rejection_feedback(logical_result, llm_result)

        if state.repair_no < state.settings.max_clue_repairs:
            state.repair_no += 1
            state.phase = RoundPhase.CLUE_PREPARE
            return

        state.attempts.append(state.current_attempt_trace)
        state.terminal_reason = "clue_not_repaired"
        self.logger.emit(
            "round_finished",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            state="round_finished",
            terminal_reason=state.terminal_reason,
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
        )
        self._finalize_round()

    def _emit_guess_started(self) -> None:
        state = self.state
        self.logger.emit(
            "guess_started",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            role="guesser",
            state="generating_guess",
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
        )

    def _generate_guess(self) -> None:
        state = self.state
        guess_response = self.model_manager.generate(
            model_entry=state.guesser_entry,
            messages=build_guesser_messages(
                card=state.card,
                current_clue=state.current_attempt_trace.accepted_clue or "",
                accepted_clues=state.accepted_clues,
                wrong_guesses=state.wrong_guesses,
                attempt_no=state.attempt_no,
            ),
            generation_params=state.settings.generation.guesser,
            runtime_policy=state.runtime_policy,
            device_preference=state.settings.device_preference,
            trace_role="guesser",
        )
        guess_text_raw = _extract_single_line(guess_response.text)
        guess_normalized = normalize_text(guess_text_raw)
        state.total_latency_ms += guess_response.latency_ms
        state.current_attempt_trace.guess = GuessTrace(
            attempt_no=state.attempt_no,
            guess_text_raw=guess_text_raw,
            guess_text_normalized=guess_normalized,
            latency_ms=guess_response.latency_ms,
            prompt_template_id=guess_response.prompt_template_id,
        )
        state.attempts.append(state.current_attempt_trace)

        solved = _guess_matches(guess_text_raw, state.card)
        self.logger.emit(
            "guess_generated",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            latency_ms=guess_response.latency_ms,
            prompt_tokens=guess_response.prompt_tokens,
            completion_tokens=guess_response.completion_tokens,
            guess_text_raw=guess_text_raw,
            guess_text_normalized=guess_normalized,
            prompt_template_id=guess_response.prompt_template_id,
            role="guesser",
            state="round_finished" if solved else "generating_clue",
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
            **_latest_prompt_fields(self.logger, "guesser"),
        )

        if solved:
            state.solved = True
            state.solved_on_attempt = state.attempt_no
            state.terminal_reason = "solved"
            self.logger.emit(
                "round_finished",
                batch_id=state.batch_id,
                round_id=state.round_id,
                card_id=state.card.id,
                state="round_finished",
                terminal_reason=state.terminal_reason,
                cluer_model_id=state.cluer_entry.id,
                guesser_model_id=state.guesser_entry.id,
                judge_model_id=state.judge_entry.id,
            )
            self._finalize_round()
            return

        state.wrong_guesses.append(guess_text_raw)
        if state.attempt_no >= state.settings.max_guess_attempts:
            state.terminal_reason = "max_guess_attempts_reached"
            self.logger.emit(
                "round_finished",
                batch_id=state.batch_id,
                round_id=state.round_id,
                card_id=state.card.id,
                state="round_finished",
                terminal_reason=state.terminal_reason,
                cluer_model_id=state.cluer_entry.id,
                guesser_model_id=state.guesser_entry.id,
                judge_model_id=state.judge_entry.id,
            )
            self._finalize_round()
            return

        state.attempt_no += 1
        state.repair_no = 1
        state.current_attempt_trace = AttemptTrace(attempt_no=state.attempt_no)
        state.phase = RoundPhase.CLUE_PREPARE

    def _finalize_round(self) -> None:
        if self.state.result is not None:
            self.state.phase = RoundPhase.FINISHED
            return
        self.state.result = RoundResult(
            run_id=self.logger.run_id,
            round_id=self.state.round_id,
            card=self.state.card,
            solved=self.state.solved,
            solved_on_attempt=self.state.solved_on_attempt,
            total_guess_attempts_used=len(self.state.attempts),
            total_clue_repairs=self.state.total_clue_drafts,
            first_clue_passed_without_repair=self.state.first_clue_passed_without_repair,
            clue_repaired_successfully=self.state.clue_repaired_successfully,
            clue_not_repaired=self.state.terminal_reason == "clue_not_repaired",
            terminal_reason=self.state.terminal_reason,
            attempts=self.state.attempts,
            cluer_model_id=self.state.cluer_entry.id,
            guesser_model_id=self.state.guesser_entry.id,
            judge_model_id=self.state.judge_entry.id,
            total_latency_ms=round(self.state.total_latency_ms, 2),
        )
        self.logger.record_round_summary(_build_round_summary(self.state.result), flush=False)
        self.state.phase = RoundPhase.FINISHED


def _build_round_summary(result: RoundResult) -> RoundSummaryRecord:
    return RoundSummaryRecord(
        run_id=result.run_id,
        round_id=result.round_id,
        card_id=result.card.id,
        target=result.card.target,
        solved=result.solved,
        solved_on_attempt=result.solved_on_attempt,
        total_guess_attempts_used=result.total_guess_attempts_used,
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


def _extract_single_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def _cluer_banned_phrases(*, card: CardRecord, rejected_clues: list[str]) -> list[str]:
    """Return exact phrases the cluer should be unable to emit verbatim."""
    phrases = [card.target, *card.aliases, *card.taboo_hard, *rejected_clues]
    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        normalized = phrase.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _format_rejection_feedback(logical_result: Any, llm_result: Any) -> str:
    parts: list[str] = []
    if logical_result.violations:
        parts.append(f"Rules failed: {', '.join(logical_result.violations)}.")
    if logical_result.matched_terms:
        parts.append(f"Matched terms: {', '.join(logical_result.matched_terms)}.")
    if llm_result.reasons:
        parts.append(f"Judge feedback: {'; '.join(llm_result.reasons)}.")
    return " ".join(parts) if parts else "The previous clue was rejected. Use a clearly different clue."


def _guess_matches(guess_text: str, card: CardRecord) -> bool:
    if normalize_text(guess_text) == normalize_text(card.target):
        return True
    if strip_punctuation(guess_text) == strip_punctuation(card.target):
        return True
    return any(strip_punctuation(guess_text) == strip_punctuation(alias) for alias in card.aliases)


def _latest_prompt_fields(logger: RunLogger, role: str) -> dict[str, str]:
    payload = logger.latest_prompt_by_role.get(role, {})
    prompt_text = str(payload.get("prompt", "")).strip()
    if not prompt_text:
        return {}

    fields = {"prompt_text": prompt_text}
    prompt_model_id = str(payload.get("model_id", "")).strip()
    if prompt_model_id:
        fields["prompt_model_id"] = prompt_model_id
    prompt_template_id = str(payload.get("prompt_template_id", "")).strip()
    if prompt_template_id:
        fields["prompt_trace_template_id"] = prompt_template_id
    return fields
