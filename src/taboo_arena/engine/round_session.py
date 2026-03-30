"""Canonical stepwise round execution shared by engine and UI adapters."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.config import RunSettings
from taboo_arena.engine.cluer_controller import (
    ClueAngle,
    ClueCandidateEvaluation,
    blocked_angles_from_evaluations,
    build_repair_feedback,
    evaluate_clue_candidates,
    parse_cluer_candidates,
    select_allowed_angles,
    select_best_candidate,
)
from taboo_arena.engine.types import AttemptTrace, GuessTrace, RepairTrace
from taboo_arena.judge import (
    GuessCanonicalizer,
    GuessJudgeResult,
    LogicalValidationResult,
    LogicalValidator,
    NormalizedLLMJudge,
    merge_judge_results,
)
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.logging.schemas import RoundSummaryRecord
from taboo_arena.models.registry import ModelEntry
from taboo_arena.prompts import build_cluer_messages, build_guesser_messages
from taboo_arena.prompts.schemas import CluerRepairFeedbackPayload, GuesserCandidatesPayload
from taboo_arena.utils.ids import new_round_id
from taboo_arena.utils.json_utils import extract_first_json_object
from taboo_arena.utils.normalization import dedupe_preserve_order, normalize_text
from taboo_arena.utils.structured_payloads import looks_like_structured_payload


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
    clue_internal_cycle_no: int = 1
    guess_internal_cycle_no: int = 0
    total_latency_ms: float = 0.0
    total_clue_drafts: int = 0
    total_repairs_after_first_failure: int = 0
    accepted_clues: list[str] = field(default_factory=list)
    rejected_clues: list[str] = field(default_factory=list)
    wrong_guesses: list[str] = field(default_factory=list)
    used_angles: list[str] = field(default_factory=list)
    blocked_angles: list[str] = field(default_factory=list)
    current_blocked_terms: list[str] = field(default_factory=list)
    judge_warning_flags: list[str] = field(default_factory=list)
    attempts: list[AttemptTrace] = field(default_factory=list)
    current_attempt_trace: AttemptTrace = field(default_factory=lambda: AttemptTrace(attempt_no=1))
    current_clue_text: str | None = None
    current_selected_clue_candidate: str | None = None
    current_clue_normalized: str | None = None
    current_logical_result: LogicalValidationResult | None = None
    current_clue_latency_ms: float = 0.0
    current_clue_prompt_template_id: str = ""
    current_allowed_angles: list[str] = field(default_factory=list)
    current_candidate_clues: list[str] = field(default_factory=list)
    current_selected_angle: str | None = None
    current_selected_guess_candidate: str | None = None
    current_guess_match_status: str | None = None
    first_clue_passed_without_repair: bool = False
    clue_repaired_successfully: bool = False
    solved: bool = False
    solved_on_attempt: int | None = None
    terminal_reason: str = "max_guess_attempts_reached"
    last_repair_feedback: CluerRepairFeedbackPayload | None = None
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
        self.guess_canonicalizer = GuessCanonicalizer()
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
                self.state.phase = self._generate_clue()
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
        state.clue_internal_cycle_no = state.repair_no
        self.logger.emit(
            "clue_draft_started",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            clue_internal_cycle_no=state.clue_internal_cycle_no,
            role="cluer",
            state="generating_clue",
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
        )

    def _generate_clue(self) -> RoundPhase:
        state = self.state
        state.current_clue_text = None
        state.current_selected_clue_candidate = None
        state.current_clue_normalized = None
        state.current_logical_result = None
        state.current_selected_angle = None
        state.current_candidate_clues = []
        state.current_blocked_terms = []
        state.clue_internal_cycle_no = state.repair_no
        allowed_angles = select_allowed_angles(
            attempt_no=state.attempt_no,
            used_angles=state.used_angles,
            blocked_angles=state.blocked_angles,
        )
        blocked_terms = _cluer_banned_phrases(card=state.card, rejected_clues=state.rejected_clues)
        state.current_blocked_terms = list(blocked_terms)
        blocked_prior_clues = dedupe_preserve_order([*state.accepted_clues, *state.rejected_clues])
        state.current_allowed_angles = [angle.value for angle in allowed_angles]
        self.logger.emit(
            "clue_candidate_cycle_started",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            clue_internal_cycle_no=state.clue_internal_cycle_no,
            allowed_angles=state.current_allowed_angles,
            blocked_angles=list(state.blocked_angles),
            blocked_terms=blocked_terms,
            blocked_prior_clues=blocked_prior_clues,
            role="cluer",
            state="planning_clue_candidates",
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
        )
        clue_response = self.model_manager.generate(
            model_entry=state.cluer_entry,
            messages=build_cluer_messages(
                card=state.card,
                accepted_clues=state.accepted_clues,
                rejected_clues=state.rejected_clues,
                wrong_guesses=state.wrong_guesses,
                attempt_no=state.attempt_no,
                repair_no=state.repair_no,
                allowed_angles=state.current_allowed_angles,
                blocked_terms=blocked_terms,
                blocked_prior_clues=blocked_prior_clues,
                blocked_angles=list(state.blocked_angles),
                repair_feedback_json=(
                    None
                    if state.last_repair_feedback is None
                    else json.dumps(state.last_repair_feedback.model_dump(mode="json"))
                ),
            ),
            generation_params=state.settings.generation.cluer,
            runtime_policy=state.runtime_policy,
            device_preference=state.settings.device_preference,
            trace_role="cluer",
            banned_phrases=blocked_terms,
        )
        parsed_candidates, parse_mode = parse_cluer_candidates(
            clue_response.text,
            allowed_angles=allowed_angles,
        )
        evaluations = evaluate_clue_candidates(
            candidates=parsed_candidates,
            validator=self.logical_validator,
            card=state.card,
            previous_accepted_clues=state.accepted_clues,
            previous_rejected_clues=state.rejected_clues,
            used_angles=state.used_angles,
        )
        state.total_latency_ms += clue_response.latency_ms
        state.total_clue_drafts += 1
        if state.repair_no > 1:
            state.total_repairs_after_first_failure += 1
        state.current_candidate_clues = [item.clue_text_raw for item in evaluations]

        self.logger.emit(
            "clue_candidates_generated",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            clue_internal_cycle_no=state.clue_internal_cycle_no,
            latency_ms=clue_response.latency_ms,
            prompt_tokens=clue_response.prompt_tokens,
            completion_tokens=clue_response.completion_tokens,
            clue_candidate_angles=[item.angle for item in parsed_candidates],
            clue_candidate_clues=[item.clue for item in parsed_candidates],
            clue_candidate_parse_mode=parse_mode,
            raw_model_output="" if parse_mode == "json" else clue_response.text,
            prompt_template_id=clue_response.prompt_template_id,
            role="cluer",
            state="hard_filtering_clue_candidates",
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
            **_latest_prompt_fields(self.logger, "cluer"),
        )
        self.logger.emit(
            "clue_candidate_validation_completed",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            clue_internal_cycle_no=state.clue_internal_cycle_no,
            candidate_results=[
                {
                    "angle": item.angle.value,
                    "clue_text_raw": item.clue_text_raw,
                    "clue_text_normalized": item.clue_text_normalized,
                    "logical_verdict": item.logical_result.verdict,
                    "logical_violations": item.logical_result.violations,
                    "logical_matched_terms": item.logical_result.matched_terms,
                    "score": item.score,
                }
                for item in evaluations
            ],
            validator_version=(
                evaluations[0].logical_result.validator_version if evaluations else "logical_v1"
            ),
            role="cluer",
            state="hard_filtering_clue_candidates",
            cluer_model_id=state.cluer_entry.id,
        )
        selected_candidate = select_best_candidate(evaluations)
        if selected_candidate is None:
            blocked_now = blocked_angles_from_evaluations(evaluations)
            state.blocked_angles = dedupe_preserve_order(
                [*state.blocked_angles, *[angle.value for angle in blocked_now]]
            )
            state.last_repair_feedback = build_repair_feedback(
                evaluations=evaluations,
                allowed_angles=allowed_angles,
                blocked_angles=blocked_now,
            )
            if parse_mode != "json":
                state.last_repair_feedback.reason_codes = dedupe_preserve_order(
                    [*state.last_repair_feedback.reason_codes, parse_mode]
                )
            self.logger.emit(
                "clue_internal_retry_requested",
                batch_id=state.batch_id,
                round_id=state.round_id,
                card_id=state.card.id,
                attempt_no=state.attempt_no,
                clue_repair_no=state.repair_no,
                clue_internal_cycle_no=state.clue_internal_cycle_no,
                reason_codes=list(state.last_repair_feedback.reason_codes),
                blocked_terms=list(state.last_repair_feedback.blocked_terms),
                blocked_angles=list(state.last_repair_feedback.blocked_angles),
                allowed_angles=list(state.last_repair_feedback.allowed_angles),
                role="cluer",
                state="repairing_clue",
                cluer_model_id=state.cluer_entry.id,
                guesser_model_id=state.guesser_entry.id,
                judge_model_id=state.judge_entry.id,
            )
            if state.repair_no < state.settings.max_clue_repairs:
                state.repair_no += 1
                state.clue_internal_cycle_no = state.repair_no
                return RoundPhase.CLUE_PREPARE
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
            return RoundPhase.FINISHED

        logical_result = selected_candidate.logical_result
        state.current_clue_text = selected_candidate.clue_text_raw
        state.current_selected_clue_candidate = selected_candidate.clue_text_raw
        state.current_clue_normalized = selected_candidate.clue_text_normalized
        state.current_logical_result = logical_result
        state.current_clue_latency_ms = clue_response.latency_ms
        state.current_clue_prompt_template_id = clue_response.prompt_template_id
        state.current_selected_angle = selected_candidate.angle.value
        self.logger.emit(
            "clue_candidate_selected",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            clue_internal_cycle_no=state.clue_internal_cycle_no,
            selected_angle=selected_candidate.angle.value,
            visible_clue_text=selected_candidate.clue_text_raw,
            clue_text_raw=selected_candidate.clue_text_raw,
            clue_text_normalized=selected_candidate.clue_text_normalized,
            candidate_score=selected_candidate.score,
            clue_candidate_clues=state.current_candidate_clues,
            role="cluer",
            state="selected_clue_candidate",
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
        )
        self.logger.emit(
            "clue_draft_generated",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            clue_internal_cycle_no=state.clue_internal_cycle_no,
            latency_ms=clue_response.latency_ms,
            prompt_tokens=clue_response.prompt_tokens,
            completion_tokens=clue_response.completion_tokens,
            visible_clue_text=selected_candidate.clue_text_raw,
            clue_text_raw=selected_candidate.clue_text_raw,
            clue_text_normalized=selected_candidate.clue_text_normalized,
            selected_angle=selected_candidate.angle.value,
            clue_candidate_clues=state.current_candidate_clues,
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
            clue_internal_cycle_no=state.clue_internal_cycle_no,
            visible_clue_text=selected_candidate.clue_text_raw,
            clue_text_raw=selected_candidate.clue_text_raw,
            clue_text_normalized=selected_candidate.clue_text_normalized,
            logical_verdict=logical_result.verdict,
            logical_violations=logical_result.violations,
            logical_matched_terms=logical_result.matched_terms,
            selected_angle=selected_candidate.angle.value,
            validator_version=logical_result.validator_version,
            role="cluer",
            state="llm_validation",
            cluer_model_id=state.cluer_entry.id,
        )
        return RoundPhase.JUDGE_PREPARE

    def _emit_judge_started(self) -> None:
        state = self.state
        self.logger.emit(
            "clue_review_started",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            clue_internal_cycle_no=state.clue_internal_cycle_no,
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

        llm_result, judge_response = self.llm_judge.evaluate_clue(
            model_manager=self.model_manager,
            model_entry=state.judge_entry,
            card=state.card,
            clue_draft=state.current_clue_text or "",
            accepted_clues=state.accepted_clues,
            rejected_clues=state.rejected_clues,
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
                selected_angle=state.current_selected_angle or "",
                candidate_clues=list(state.current_candidate_clues),
            )
        )
        self.logger.emit(
            "llm_validation_completed",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            clue_internal_cycle_no=state.clue_internal_cycle_no,
            latency_ms=judge_response.latency_ms,
            prompt_tokens=judge_response.prompt_tokens,
            completion_tokens=judge_response.completion_tokens,
            llm_judge_verdict=llm_result.verdict,
            llm_judge_reasons=llm_result.reasons,
            llm_judge_warnings=llm_result.warnings,
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
            if state.current_selected_angle:
                state.used_angles = dedupe_preserve_order([*state.used_angles, state.current_selected_angle])
            state.current_attempt_trace.accepted_clue = accepted_clue
            state.current_attempt_trace.accepted_repair_no = state.repair_no
            if state.attempt_no == 1 and state.repair_no == 1:
                state.first_clue_passed_without_repair = True
            if state.repair_no > 1:
                state.clue_repaired_successfully = True
            state.judge_warning_flags = (
                list(merged.llm_judge_warnings or llm_result.warnings or llm_result.reasons)
                if merged.final_verdict == "pass_with_warning"
                else []
            )
            self.logger.emit(
                "clue_accepted",
                batch_id=state.batch_id,
                round_id=state.round_id,
                card_id=state.card.id,
                attempt_no=state.attempt_no,
                clue_repair_no=state.repair_no,
                clue_internal_cycle_no=state.clue_internal_cycle_no,
                visible_clue_text=accepted_clue,
                clue_text_raw=accepted_clue,
                selected_angle=state.current_selected_angle,
                judge_warning_flags=list(state.judge_warning_flags),
                final_judge_verdict=merged.final_verdict,
                judge_disagreement=merged.judge_disagreement,
                state="generating_guess",
                cluer_model_id=state.cluer_entry.id,
                guesser_model_id=state.guesser_entry.id,
                judge_model_id=state.judge_entry.id,
            )
            state.last_repair_feedback = None
            state.phase = RoundPhase.GUESS_PREPARE
            return

        blocked_for_retry = [
            ClueAngle(state.current_selected_angle)
            for _ in [0]
            if state.current_selected_angle
        ]
        if state.current_selected_angle:
            state.blocked_angles = dedupe_preserve_order([*state.blocked_angles, state.current_selected_angle])
        self.logger.emit(
            "clue_repair_requested",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            clue_repair_no=state.repair_no,
            clue_internal_cycle_no=state.clue_internal_cycle_no,
            selected_angle=state.current_selected_angle,
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
        state.last_repair_feedback = build_repair_feedback(
            evaluations=[
                ClueCandidateEvaluation(
                    angle=ClueAngle(state.current_selected_angle or state.current_allowed_angles[0]),
                    clue_text_raw=state.current_clue_text or "",
                    clue_text_normalized=state.current_clue_normalized or "",
                    logical_result=logical_result,
                    score=0,
                )
            ],
            allowed_angles=[ClueAngle(value) for value in state.current_allowed_angles],
            blocked_angles=blocked_for_retry,
            llm_result=llm_result,
        )

        if state.repair_no < state.settings.max_clue_repairs:
            state.repair_no += 1
            state.clue_internal_cycle_no = state.repair_no
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
        state.phase = RoundPhase.FINISHED

    def _emit_guess_started(self) -> None:
        state = self.state
        state.guess_internal_cycle_no = 1
        state.current_selected_guess_candidate = None
        state.current_guess_match_status = None
        self.logger.emit(
            "guess_started",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            guess_internal_cycle_no=state.guess_internal_cycle_no,
            role="guesser",
            state="generating_guess",
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
        )

    def _generate_guess(self) -> None:
        state = self.state
        guess_shortlist: list[str] = []
        selected_guess_text_raw: str | None = None
        selected_match_result: Any | None = None
        hidden_retry_count = 0
        total_guess_latency_ms = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        final_prompt_template_id = ""

        for hidden_retry_no in range(state.settings.guesser_hidden_retry_budget + 1):
            state.guess_internal_cycle_no = hidden_retry_no + 1
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
                banned_phrases=list(state.wrong_guesses),
            )
            state.total_latency_ms += guess_response.latency_ms
            total_guess_latency_ms += guess_response.latency_ms
            total_prompt_tokens += int(guess_response.prompt_tokens)
            total_completion_tokens += int(guess_response.completion_tokens)
            final_prompt_template_id = guess_response.prompt_template_id

            parsed_guesses, parse_mode = _parse_guesser_candidates(guess_response.text)
            evaluations = self.guess_canonicalizer.evaluate_shortlist(
                parsed_guesses,
                target=state.card.target,
                previous_wrong_guesses=state.wrong_guesses,
            )
            guess_shortlist = parsed_guesses
            self.logger.emit(
                "guess_shortlist_generated",
                batch_id=state.batch_id,
                round_id=state.round_id,
                card_id=state.card.id,
                attempt_no=state.attempt_no,
                guess_hidden_retry_no=hidden_retry_no,
                guess_internal_cycle_no=state.guess_internal_cycle_no,
                guess_shortlist_candidates=parsed_guesses,
                guess_shortlist_candidate_keys=[item.analysis.candidate_keys for item in evaluations],
                guess_shortlist_invalid_reasons=[
                    item.invalid_reason or ""
                    for item in evaluations
                ],
                guess_shortlist_repeated_against=[
                    item.repeated_against
                    for item in evaluations
                ],
                guess_parse_mode=parse_mode,
                raw_model_output="" if parse_mode == "json" else guess_response.text,
                latency_ms=guess_response.latency_ms,
                prompt_tokens=guess_response.prompt_tokens,
                completion_tokens=guess_response.completion_tokens,
                prompt_template_id=guess_response.prompt_template_id,
                role="guesser",
                state="guess_filtering",
                cluer_model_id=state.cluer_entry.id,
                guesser_model_id=state.guesser_entry.id,
                judge_model_id=state.judge_entry.id,
                **_latest_prompt_fields(self.logger, "guesser"),
            )
            selected_evaluation = next(
                (item for item in evaluations if item.is_valid_visible_candidate),
                None,
            )
            if selected_evaluation is not None:
                selected_guess_text_raw = selected_evaluation.guess_text_raw
                selected_match_result = selected_evaluation.match_result
                hidden_retry_count = hidden_retry_no
                break
            if hidden_retry_no >= state.settings.guesser_hidden_retry_budget:
                hidden_retry_count = hidden_retry_no
                break
            hidden_retry_reasons = dedupe_preserve_order(
                [item.invalid_reason or "no_valid_guess" for item in evaluations]
            )
            if parse_mode != "json":
                hidden_retry_reasons = dedupe_preserve_order([*hidden_retry_reasons, parse_mode])
            self.logger.emit(
                "guess_hidden_retry_requested",
                batch_id=state.batch_id,
                round_id=state.round_id,
                card_id=state.card.id,
                attempt_no=state.attempt_no,
                guess_hidden_retry_no=hidden_retry_no + 1,
                guess_internal_cycle_no=state.guess_internal_cycle_no,
                reason_codes=hidden_retry_reasons,
                state="generating_guess",
                cluer_model_id=state.cluer_entry.id,
                guesser_model_id=state.guesser_entry.id,
                judge_model_id=state.judge_entry.id,
            )

        if selected_guess_text_raw is None or selected_match_result is None:
            selected_guess_text_raw = "(no valid guess)"
            selected_match_result = self.guess_canonicalizer.match("", state.card.target)
            selected_match_result.reason = "no_valid_new_guess"

        self.logger.emit(
            "guess_review_started",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            guess_internal_cycle_no=state.guess_internal_cycle_no,
            role="judge",
            state="verifying_guess",
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
        )
        if selected_guess_text_raw == "(no valid guess)":
            guess_judge_result = GuessJudgeResult(
                correct=False,
                reason_codes=[selected_match_result.reason],
                warnings=[],
                matched_surface_forms=[],
                judge_version="guess_judge_v1",
            )
            guess_judge_latency_ms = 0.0
            guess_judge_prompt_tokens = 0
            guess_judge_completion_tokens = 0
            guess_judge_prompt_template_id = ""
            guess_judge_prompt_fields: dict[str, str] = {}
        else:
            guess_judge_result, guess_judge_response = self.llm_judge.evaluate_guess(
                model_manager=self.model_manager,
                model_entry=state.judge_entry,
                card=state.card,
                guess_text=selected_guess_text_raw,
                attempt_no=state.attempt_no,
                match_status=str(selected_match_result.status),
                match_reason=selected_match_result.reason,
                candidate_spans=list(selected_match_result.analysis.candidate_spans),
                warnings=list(selected_match_result.warnings),
                generation_params=state.settings.generation.judge,
                runtime_policy=state.runtime_policy,
                device_preference=state.settings.device_preference,
            )
            guess_judge_latency_ms = guess_judge_response.latency_ms
            guess_judge_prompt_tokens = int(guess_judge_response.prompt_tokens)
            guess_judge_completion_tokens = int(guess_judge_response.completion_tokens)
            guess_judge_prompt_template_id = guess_judge_response.prompt_template_id
            guess_judge_prompt_fields = _latest_prompt_fields(self.logger, "judge")
            state.total_latency_ms += guess_judge_latency_ms
            total_guess_latency_ms += guess_judge_latency_ms
            total_prompt_tokens += guess_judge_prompt_tokens
            total_completion_tokens += guess_judge_completion_tokens

        final_guess_correct = bool(selected_match_result.is_correct)
        guess_judge_disagreement = bool(guess_judge_result.correct) != final_guess_correct
        state.current_selected_guess_candidate = selected_guess_text_raw
        state.current_guess_match_status = str(selected_match_result.status)
        visible_guess_text = "" if selected_guess_text_raw == "(no valid guess)" else selected_guess_text_raw
        self.logger.emit(
            "guess_validation_completed",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            guess_internal_cycle_no=state.guess_internal_cycle_no,
            latency_ms=guess_judge_latency_ms,
            prompt_tokens=guess_judge_prompt_tokens,
            completion_tokens=guess_judge_completion_tokens,
            visible_guess_text=visible_guess_text,
            guess_text_raw=selected_guess_text_raw,
            guess_match_status=str(selected_match_result.status),
            guess_match_reason=selected_match_result.reason,
            guess_judge_correct=guess_judge_result.correct,
            guess_judge_reason_codes=list(guess_judge_result.reason_codes),
            guess_judge_warnings=list(guess_judge_result.warnings),
            guess_judge_matched_surface_forms=list(guess_judge_result.matched_surface_forms),
            judge_disagreement=guess_judge_disagreement,
            final_guess_correct=final_guess_correct,
            prompt_template_id=guess_judge_prompt_template_id,
            role="judge",
            state="round_finished" if final_guess_correct else "generating_clue",
            cluer_model_id=state.cluer_entry.id,
            guesser_model_id=state.guesser_entry.id,
            judge_model_id=state.judge_entry.id,
            **guess_judge_prompt_fields,
        )

        guess_normalized = normalize_text(selected_guess_text_raw)
        state.current_attempt_trace.guess = GuessTrace(
            attempt_no=state.attempt_no,
            guess_text_raw=selected_guess_text_raw,
            guess_text_normalized=guess_normalized,
            latency_ms=total_guess_latency_ms,
            prompt_template_id=final_prompt_template_id,
            match_status=str(selected_match_result.status),
            match_reason=selected_match_result.reason,
            warnings=list(selected_match_result.warnings),
            hidden_retry_count=hidden_retry_count,
            shortlist_candidates=list(guess_shortlist),
            guess_judge_result=guess_judge_result,
            judge_disagreement=guess_judge_disagreement,
        )
        state.attempts.append(state.current_attempt_trace)

        solved = final_guess_correct
        self.logger.emit(
            "guess_generated",
            batch_id=state.batch_id,
            round_id=state.round_id,
            card_id=state.card.id,
            attempt_no=state.attempt_no,
            guess_internal_cycle_no=state.guess_internal_cycle_no,
            latency_ms=total_guess_latency_ms,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            visible_guess_text=visible_guess_text,
            guess_text_raw=selected_guess_text_raw,
            guess_text_normalized=guess_normalized,
            guess_match_status=str(selected_match_result.status),
            guess_match_reason=selected_match_result.reason,
            guess_match_warnings=list(selected_match_result.warnings),
            guess_match_candidates=selected_match_result.analysis.candidate_spans,
            guess_hidden_retry_count=hidden_retry_count,
            guess_shortlist_candidates=guess_shortlist,
            guess_judge_correct=guess_judge_result.correct,
            guess_judge_reason_codes=list(guess_judge_result.reason_codes),
            guess_judge_warnings=list(guess_judge_result.warnings),
            guess_judge_matched_surface_forms=list(guess_judge_result.matched_surface_forms),
            guess_judge_disagreement=guess_judge_disagreement,
            final_guess_correct=final_guess_correct,
            prompt_template_id=final_prompt_template_id,
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

        if selected_guess_text_raw != "(no valid guess)":
            state.wrong_guesses.append(selected_guess_text_raw)
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
        state.clue_internal_cycle_no = 1
        state.guess_internal_cycle_no = 0
        state.current_attempt_trace = AttemptTrace(attempt_no=state.attempt_no)
        state.current_selected_angle = None
        state.current_selected_clue_candidate = None
        state.current_candidate_clues = []
        state.current_allowed_angles = []
        state.current_blocked_terms = []
        state.current_selected_guess_candidate = None
        state.current_guess_match_status = None
        state.last_repair_feedback = None
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
    phrases = [card.target, *card.taboo_hard, *rejected_clues]
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


def _parse_guesser_candidates(text: str) -> tuple[list[str], str]:
    """Parse strict JSON shortlist output without promoting raw fallback text."""
    try:
        payload = extract_first_json_object(text)
        structured = GuesserCandidatesPayload.model_validate(payload)
        guesses = [item.strip() for item in structured.guesses if isinstance(item, str) and item.strip()]
        if guesses:
            return guesses, "json"
    except (ValueError, ValidationError):
        return [], "parse_failure"
    return [], "structured_payload_rejected" if looks_like_structured_payload(text) else "unstructured_output"


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
