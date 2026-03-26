"""Stepwise single-round execution for the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.config import RunSettings
from taboo_arena.engine.round_engine import RoundResult
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
    """One resumable single round for the Streamlit UI."""

    logger: RunLogger
    settings: RunSettings
    card: CardRecord
    cluer_entry: ModelEntry
    guesser_entry: ModelEntry
    judge_entry: ModelEntry
    round_id: str
    runtime_policy: str
    phase: ControllerPhase = "clue_prepare"
    attempt_no: int = 1
    repair_no: int = 1
    total_latency_ms: float = 0.0
    total_clue_repairs: int = 0
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
    current_guess_text: str | None = None
    current_guess_normalized: str | None = None
    first_clue_passed_without_repair: bool = False
    clue_repaired_successfully: bool = False
    solved: bool = False
    solved_on_attempt: int | None = None
    terminal_reason: str = "max_guess_attempts_reached"
    last_rejection_feedback: str | None = None
    result: RoundResult | None = None
    error_message: str | None = None


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
    """Create a new resumable round and emit the opening event."""
    runtime_policy = model_manager.resolve_runtime_policy(
        [cluer_entry, guesser_entry, judge_entry],
        requested_policy=settings.memory_policy,
        device_preference=settings.device_preference,
    )
    round_id = new_round_id()
    logger.emit(
        "round_started",
        batch_id=None,
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
    return LiveRoundController(
        logger=logger,
        settings=settings.model_copy(deep=True),
        card=card,
        cluer_entry=cluer_entry,
        guesser_entry=guesser_entry,
        judge_entry=judge_entry,
        round_id=round_id,
        runtime_policy=runtime_policy,
    )


def advance_live_round(controller: LiveRoundController, *, model_manager: Any) -> None:
    """Advance the controller by exactly one UI-safe phase."""
    if controller.phase == "finished":
        return

    model_manager.logger = controller.logger
    try:
        if controller.phase == "clue_prepare":
            _emit_clue_started(controller)
            controller.phase = "clue_ready"
            return
        if controller.phase == "clue_ready":
            controller.phase = "clue_generate"
            return
        if controller.phase == "clue_generate":
            _generate_clue(controller, model_manager=model_manager)
            controller.phase = "judge_prepare"
            return
        if controller.phase == "judge_prepare":
            _emit_judge_started(controller)
            controller.phase = "judge_ready"
            return
        if controller.phase == "judge_ready":
            controller.phase = "judge_generate"
            return
        if controller.phase == "judge_generate":
            _generate_judge_decision(controller, model_manager=model_manager)
            return
        if controller.phase == "guess_prepare":
            _emit_guess_started(controller)
            controller.phase = "guess_ready"
            return
        if controller.phase == "guess_ready":
            controller.phase = "guess_generate"
            return
        if controller.phase == "guess_generate":
            _generate_guess(controller, model_manager=model_manager)
            return
    except Exception as exc:
        controller.logger.emit("error", error_message=str(exc), state="idle")
        controller.error_message = str(exc)
        controller.phase = "finished"


def _emit_clue_started(controller: LiveRoundController) -> None:
    controller.logger.emit(
        "clue_draft_started",
        batch_id=None,
        round_id=controller.round_id,
        card_id=controller.card.id,
        attempt_no=controller.attempt_no,
        clue_repair_no=controller.repair_no,
        role="cluer",
        state="generating_clue",
        cluer_model_id=controller.cluer_entry.id,
        guesser_model_id=controller.guesser_entry.id,
        judge_model_id=controller.judge_entry.id,
    )


def _generate_clue(controller: LiveRoundController, *, model_manager: Any) -> None:
    clue_response = model_manager.generate(
        model_entry=controller.cluer_entry,
        messages=build_cluer_messages(
            card=controller.card,
            accepted_clues=controller.accepted_clues,
            rejected_clues=controller.rejected_clues,
            wrong_guesses=controller.wrong_guesses,
            attempt_no=controller.attempt_no,
            repair_no=controller.repair_no,
            last_rejection_feedback=controller.last_rejection_feedback,
        ),
        generation_params=controller.settings.generation.cluer,
        runtime_policy=controller.runtime_policy,
        device_preference=controller.settings.device_preference,
        trace_role="cluer",
        banned_phrases=_cluer_banned_phrases(
            card=controller.card,
            rejected_clues=controller.rejected_clues,
        ),
    )
    clue_text_raw = _extract_single_line(clue_response.text)
    logical_result = LogicalValidator(controller.settings.logical_validator).validate(
        clue_text_raw,
        card=controller.card,
        previous_accepted_clues=controller.accepted_clues,
        previous_rejected_clues=controller.rejected_clues,
    )
    controller.total_latency_ms += clue_response.latency_ms
    controller.total_clue_repairs += 1
    controller.current_clue_text = clue_text_raw
    controller.current_clue_normalized = logical_result.normalized_text
    controller.current_logical_result = logical_result
    controller.current_clue_latency_ms = clue_response.latency_ms
    controller.current_clue_prompt_template_id = clue_response.prompt_template_id

    controller.logger.emit(
        "clue_draft_generated",
        batch_id=None,
        round_id=controller.round_id,
        card_id=controller.card.id,
        attempt_no=controller.attempt_no,
        clue_repair_no=controller.repair_no,
        latency_ms=clue_response.latency_ms,
        prompt_tokens=clue_response.prompt_tokens,
        completion_tokens=clue_response.completion_tokens,
        clue_text_raw=clue_text_raw,
        clue_text_normalized=logical_result.normalized_text,
        prompt_template_id=clue_response.prompt_template_id,
        role="cluer",
        state="logical_validation",
        cluer_model_id=controller.cluer_entry.id,
        guesser_model_id=controller.guesser_entry.id,
        judge_model_id=controller.judge_entry.id,
        **_latest_prompt_fields(controller.logger, "cluer"),
    )
    controller.logger.emit(
        "logical_validation_completed",
        batch_id=None,
        round_id=controller.round_id,
        card_id=controller.card.id,
        attempt_no=controller.attempt_no,
        clue_repair_no=controller.repair_no,
        clue_text_raw=clue_text_raw,
        clue_text_normalized=logical_result.normalized_text,
        logical_verdict=logical_result.verdict,
        logical_violations=logical_result.violations,
        logical_matched_terms=logical_result.matched_terms,
        validator_version=logical_result.validator_version,
        role="cluer",
        state="llm_validation",
        cluer_model_id=controller.cluer_entry.id,
    )


def _emit_judge_started(controller: LiveRoundController) -> None:
    controller.logger.emit(
        "clue_review_started",
        batch_id=None,
        round_id=controller.round_id,
        card_id=controller.card.id,
        attempt_no=controller.attempt_no,
        clue_repair_no=controller.repair_no,
        role="judge",
        state="llm_validation",
        cluer_model_id=controller.cluer_entry.id,
        guesser_model_id=controller.guesser_entry.id,
        judge_model_id=controller.judge_entry.id,
    )


def _generate_judge_decision(controller: LiveRoundController, *, model_manager: Any) -> None:
    logical_result = controller.current_logical_result
    if logical_result is None:
        raise RuntimeError("Judge step started without a logical validation result.")

    llm_result, judge_response = NormalizedLLMJudge().evaluate(
        model_manager=model_manager,
        model_entry=controller.judge_entry,
        card=controller.card,
        clue_draft=controller.current_clue_text or "",
        accepted_clues=controller.accepted_clues,
        wrong_guesses=controller.wrong_guesses,
        attempt_no=controller.attempt_no,
        generation_params=controller.settings.generation.judge,
        runtime_policy=controller.runtime_policy,
        device_preference=controller.settings.device_preference,
    )
    merged = merge_judge_results(
        logical_result,
        llm_result,
        block_on_uncertain=controller.settings.block_on_uncertain,
    )
    controller.total_latency_ms += judge_response.latency_ms
    controller.current_attempt_trace.repairs.append(
        RepairTrace(
            repair_no=controller.repair_no,
            clue_text_raw=controller.current_clue_text or "",
            clue_text_normalized=controller.current_clue_normalized or "",
            logical_result=logical_result,
            llm_result=llm_result,
            merged_result=merged,
            latency_ms=controller.current_clue_latency_ms + judge_response.latency_ms,
            prompt_template_id=controller.current_clue_prompt_template_id,
        )
    )

    controller.logger.emit(
        "llm_validation_completed",
        batch_id=None,
        round_id=controller.round_id,
        card_id=controller.card.id,
        attempt_no=controller.attempt_no,
        clue_repair_no=controller.repair_no,
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
        judge_model_id=controller.judge_entry.id,
        cluer_model_id=controller.cluer_entry.id,
        guesser_model_id=controller.guesser_entry.id,
        **_latest_prompt_fields(controller.logger, "judge"),
    )

    if merged.final_verdict != "fail":
        accepted_clue = controller.current_clue_text or ""
        controller.accepted_clues.append(accepted_clue)
        controller.current_attempt_trace.accepted_clue = accepted_clue
        controller.current_attempt_trace.accepted_repair_no = controller.repair_no
        if controller.attempt_no == 1 and controller.repair_no == 1:
            controller.first_clue_passed_without_repair = True
        if controller.repair_no > 1:
            controller.clue_repaired_successfully = True
        controller.logger.emit(
            "clue_accepted",
            batch_id=None,
            round_id=controller.round_id,
            card_id=controller.card.id,
            attempt_no=controller.attempt_no,
            clue_repair_no=controller.repair_no,
            clue_text_raw=accepted_clue,
            final_judge_verdict=merged.final_verdict,
            judge_disagreement=merged.judge_disagreement,
            state="generating_guess",
            cluer_model_id=controller.cluer_entry.id,
            guesser_model_id=controller.guesser_entry.id,
            judge_model_id=controller.judge_entry.id,
        )
        controller.last_rejection_feedback = None
        controller.phase = "guess_prepare"
        return

    controller.logger.emit(
        "clue_repair_requested",
        batch_id=None,
        round_id=controller.round_id,
        card_id=controller.card.id,
        attempt_no=controller.attempt_no,
        clue_repair_no=controller.repair_no,
        final_judge_verdict=merged.final_verdict,
        logical_violations=logical_result.violations,
        logical_matched_terms=logical_result.matched_terms,
        llm_judge_verdict=llm_result.verdict,
        llm_judge_reasons=llm_result.reasons,
        judge_disagreement=merged.judge_disagreement,
        state="repairing_clue",
        cluer_model_id=controller.cluer_entry.id,
        guesser_model_id=controller.guesser_entry.id,
        judge_model_id=controller.judge_entry.id,
        **_latest_prompt_fields(controller.logger, "judge"),
    )
    controller.rejected_clues.append(controller.current_clue_text or "")
    controller.last_rejection_feedback = _format_rejection_feedback(
        logical_result,
        llm_result,
    )

    if controller.repair_no < controller.settings.max_clue_repairs:
        controller.repair_no += 1
        controller.phase = "clue_prepare"
        return

    controller.attempts.append(controller.current_attempt_trace)
    controller.terminal_reason = "clue_not_repaired"
    controller.logger.emit(
        "round_finished",
        batch_id=None,
        round_id=controller.round_id,
        card_id=controller.card.id,
        state="round_finished",
        terminal_reason=controller.terminal_reason,
        cluer_model_id=controller.cluer_entry.id,
        guesser_model_id=controller.guesser_entry.id,
        judge_model_id=controller.judge_entry.id,
    )
    _finalize_live_round(controller)


def _emit_guess_started(controller: LiveRoundController) -> None:
    controller.logger.emit(
        "guess_started",
        batch_id=None,
        round_id=controller.round_id,
        card_id=controller.card.id,
        attempt_no=controller.attempt_no,
        role="guesser",
        state="generating_guess",
        cluer_model_id=controller.cluer_entry.id,
        guesser_model_id=controller.guesser_entry.id,
        judge_model_id=controller.judge_entry.id,
    )


def _generate_guess(controller: LiveRoundController, *, model_manager: Any) -> None:
    guess_response = model_manager.generate(
        model_entry=controller.guesser_entry,
        messages=build_guesser_messages(
            card=controller.card,
            current_clue=controller.current_attempt_trace.accepted_clue or "",
            accepted_clues=controller.accepted_clues,
            wrong_guesses=controller.wrong_guesses,
            attempt_no=controller.attempt_no,
        ),
        generation_params=controller.settings.generation.guesser,
        runtime_policy=controller.runtime_policy,
        device_preference=controller.settings.device_preference,
        trace_role="guesser",
    )
    guess_text_raw = _extract_single_line(guess_response.text)
    guess_normalized = normalize_text(guess_text_raw)
    controller.total_latency_ms += guess_response.latency_ms
    controller.current_attempt_trace.guess = GuessTrace(
        attempt_no=controller.attempt_no,
        guess_text_raw=guess_text_raw,
        guess_text_normalized=guess_normalized,
        latency_ms=guess_response.latency_ms,
        prompt_template_id=guess_response.prompt_template_id,
    )
    controller.attempts.append(controller.current_attempt_trace)

    solved = _guess_matches(guess_text_raw, controller.card)
    controller.logger.emit(
        "guess_generated",
        batch_id=None,
        round_id=controller.round_id,
        card_id=controller.card.id,
        attempt_no=controller.attempt_no,
        latency_ms=guess_response.latency_ms,
        prompt_tokens=guess_response.prompt_tokens,
        completion_tokens=guess_response.completion_tokens,
        guess_text_raw=guess_text_raw,
        guess_text_normalized=guess_normalized,
        prompt_template_id=guess_response.prompt_template_id,
        role="guesser",
        state="round_finished" if solved else "generating_clue",
        cluer_model_id=controller.cluer_entry.id,
        guesser_model_id=controller.guesser_entry.id,
        judge_model_id=controller.judge_entry.id,
        **_latest_prompt_fields(controller.logger, "guesser"),
    )

    if solved:
        controller.solved = True
        controller.solved_on_attempt = controller.attempt_no
        controller.terminal_reason = "solved"
        controller.logger.emit(
            "round_finished",
            batch_id=None,
            round_id=controller.round_id,
            card_id=controller.card.id,
            state="round_finished",
            terminal_reason=controller.terminal_reason,
            cluer_model_id=controller.cluer_entry.id,
            guesser_model_id=controller.guesser_entry.id,
            judge_model_id=controller.judge_entry.id,
        )
        _finalize_live_round(controller)
        return

    controller.wrong_guesses.append(guess_text_raw)
    if controller.attempt_no >= controller.settings.max_guess_attempts:
        controller.terminal_reason = "max_guess_attempts_reached"
        controller.logger.emit(
            "round_finished",
            batch_id=None,
            round_id=controller.round_id,
            card_id=controller.card.id,
            state="round_finished",
            terminal_reason=controller.terminal_reason,
            cluer_model_id=controller.cluer_entry.id,
            guesser_model_id=controller.guesser_entry.id,
            judge_model_id=controller.judge_entry.id,
        )
        _finalize_live_round(controller)
        return

    controller.attempt_no += 1
    controller.repair_no = 1
    controller.current_attempt_trace = AttemptTrace(attempt_no=controller.attempt_no)
    controller.phase = "clue_prepare"


def _finalize_live_round(controller: LiveRoundController) -> None:
    controller.result = RoundResult(
        run_id=controller.logger.run_id,
        round_id=controller.round_id,
        card=controller.card,
        solved=controller.solved,
        solved_on_attempt=controller.solved_on_attempt,
        total_guess_attempts_used=len(controller.attempts),
        total_clue_repairs=controller.total_clue_repairs,
        first_clue_passed_without_repair=controller.first_clue_passed_without_repair,
        clue_repaired_successfully=controller.clue_repaired_successfully,
        clue_not_repaired=controller.terminal_reason == "clue_not_repaired",
        terminal_reason=controller.terminal_reason,
        attempts=controller.attempts,
        cluer_model_id=controller.cluer_entry.id,
        guesser_model_id=controller.guesser_entry.id,
        judge_model_id=controller.judge_entry.id,
        total_latency_ms=round(controller.total_latency_ms, 2),
    )
    controller.logger.record_round_summary(
        RoundSummaryRecord(
            run_id=controller.result.run_id,
            round_id=controller.result.round_id,
            card_id=controller.result.card.id,
            target=controller.result.card.target,
            solved=controller.result.solved,
            solved_on_attempt=controller.result.solved_on_attempt,
            total_guess_attempts_used=controller.result.total_guess_attempts_used,
            total_clue_repairs=controller.result.total_clue_repairs,
            first_clue_passed_without_repair=controller.result.first_clue_passed_without_repair,
            clue_repaired_successfully=controller.result.clue_repaired_successfully,
            clue_not_repaired=controller.result.clue_not_repaired,
            terminal_reason=controller.result.terminal_reason,
            cluer_model_id=controller.result.cluer_model_id,
            guesser_model_id=controller.result.guesser_model_id,
            judge_model_id=controller.result.judge_model_id,
            total_latency_ms=controller.result.total_latency_ms,
        )
    )
    controller.phase = "finished"


def _extract_single_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def _cluer_banned_phrases(*, card: CardRecord, rejected_clues: list[str]) -> list[str]:
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
