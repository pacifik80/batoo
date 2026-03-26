"""Single-round benchmark engine."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.config import RunSettings
from taboo_arena.engine.types import AttemptTrace, GuessTrace, RepairTrace
from taboo_arena.judge import LogicalValidator, NormalizedLLMJudge, merge_judge_results
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.logging.schemas import RoundSummaryRecord
from taboo_arena.models.registry import ModelEntry
from taboo_arena.prompts import build_cluer_messages, build_guesser_messages
from taboo_arena.utils.ids import new_round_id
from taboo_arena.utils.normalization import normalize_text, strip_punctuation


@dataclass(slots=True)
class RoundResult:
    """Result of a benchmark round."""

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


class RoundEngine:
    """Run single-round benchmark sessions with hidden clue repair."""

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
        self.logical_validator = logical_validator or LogicalValidator(settings.logical_validator)
        self.llm_judge = llm_judge or NormalizedLLMJudge()
        random.seed(settings.random_seed)

    def play_round(
        self,
        *,
        card: CardRecord,
        cluer_entry: ModelEntry,
        guesser_entry: ModelEntry,
        judge_entry: ModelEntry,
        batch_id: str | None = None,
    ) -> RoundResult:
        """Run one round with exactly three guess attempts."""
        round_id = new_round_id()
        runtime_policy = self.model_manager.resolve_runtime_policy(
            [cluer_entry, guesser_entry, judge_entry],
            requested_policy=self.settings.memory_policy,
            device_preference=self.settings.device_preference,
        )
        total_latency_ms = 0.0
        accepted_clues: list[str] = []
        rejected_clues: list[str] = []
        wrong_guesses: list[str] = []
        attempts: list[AttemptTrace] = []
        solved = False
        solved_on_attempt: int | None = None
        total_clue_repairs = 0
        first_clue_passed_without_repair = False
        clue_repaired_successfully = False
        terminal_reason = "max_guess_attempts_reached"
        last_rejection_feedback: str | None = None

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
            seed=self.settings.random_seed,
        )

        for attempt_no in range(1, self.settings.max_guess_attempts + 1):
            attempt_trace = AttemptTrace(attempt_no=attempt_no)
            accepted_clue: str | None = None

            for repair_no in range(1, self.settings.max_clue_repairs + 1):
                self.logger.emit(
                    "clue_draft_started",
                    batch_id=batch_id,
                    round_id=round_id,
                    card_id=card.id,
                    attempt_no=attempt_no,
                    clue_repair_no=repair_no,
                    role="cluer",
                    state="generating_clue",
                    cluer_model_id=cluer_entry.id,
                    guesser_model_id=guesser_entry.id,
                    judge_model_id=judge_entry.id,
                )
                clue_response = self.model_manager.generate(
                    model_entry=cluer_entry,
                    messages=build_cluer_messages(
                        card=card,
                        accepted_clues=accepted_clues,
                        rejected_clues=rejected_clues,
                        wrong_guesses=wrong_guesses,
                        attempt_no=attempt_no,
                        repair_no=repair_no,
                        last_rejection_feedback=last_rejection_feedback,
                    ),
                    generation_params=self.settings.generation.cluer,
                    runtime_policy=runtime_policy,
                    device_preference=self.settings.device_preference,
                    trace_role="cluer",
                    banned_phrases=_cluer_banned_phrases(card=card, rejected_clues=rejected_clues),
                )
                clue_text_raw = _extract_single_line(clue_response.text)
                total_latency_ms += clue_response.latency_ms
                total_clue_repairs += 1

                self.logger.emit(
                    "clue_draft_generated",
                    batch_id=batch_id,
                    round_id=round_id,
                    card_id=card.id,
                    attempt_no=attempt_no,
                    clue_repair_no=repair_no,
                    latency_ms=clue_response.latency_ms,
                    prompt_tokens=clue_response.prompt_tokens,
                    completion_tokens=clue_response.completion_tokens,
                    clue_text_raw=clue_text_raw,
                    clue_text_normalized=normalize_text(clue_text_raw),
                    prompt_template_id=clue_response.prompt_template_id,
                    role="cluer",
                    state="logical_validation",
                    cluer_model_id=cluer_entry.id,
                    guesser_model_id=guesser_entry.id,
                    judge_model_id=judge_entry.id,
                    **_latest_prompt_fields(self.logger, "cluer"),
                )

                logical_result = self.logical_validator.validate(
                    clue_text_raw,
                    card=card,
                    previous_accepted_clues=accepted_clues,
                    previous_rejected_clues=rejected_clues,
                )
                self.logger.emit(
                    "logical_validation_completed",
                    batch_id=batch_id,
                    round_id=round_id,
                    card_id=card.id,
                    attempt_no=attempt_no,
                    clue_repair_no=repair_no,
                    clue_text_raw=clue_text_raw,
                    clue_text_normalized=logical_result.normalized_text,
                    logical_verdict=logical_result.verdict,
                    logical_violations=logical_result.violations,
                    logical_matched_terms=logical_result.matched_terms,
                    validator_version=logical_result.validator_version,
                    role="cluer",
                    state="llm_validation",
                    cluer_model_id=cluer_entry.id,
                )

                self.logger.emit(
                    "clue_review_started",
                    batch_id=batch_id,
                    round_id=round_id,
                    card_id=card.id,
                    attempt_no=attempt_no,
                    clue_repair_no=repair_no,
                    role="judge",
                    state="llm_validation",
                    cluer_model_id=cluer_entry.id,
                    guesser_model_id=guesser_entry.id,
                    judge_model_id=judge_entry.id,
                )
                llm_result, judge_response = self.llm_judge.evaluate(
                    model_manager=self.model_manager,
                    model_entry=judge_entry,
                    card=card,
                    clue_draft=clue_text_raw,
                    accepted_clues=accepted_clues,
                    wrong_guesses=wrong_guesses,
                    attempt_no=attempt_no,
                    generation_params=self.settings.generation.judge,
                    runtime_policy=runtime_policy,
                    device_preference=self.settings.device_preference,
                )
                total_latency_ms += judge_response.latency_ms
                merged = merge_judge_results(
                    logical_result,
                    llm_result,
                    block_on_uncertain=self.settings.block_on_uncertain,
                )

                self.logger.emit(
                    "llm_validation_completed",
                    batch_id=batch_id,
                    round_id=round_id,
                    card_id=card.id,
                    attempt_no=attempt_no,
                    clue_repair_no=repair_no,
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
                    judge_model_id=judge_entry.id,
                    cluer_model_id=cluer_entry.id,
                    guesser_model_id=guesser_entry.id,
                    **_latest_prompt_fields(self.logger, "judge"),
                )

                attempt_trace.repairs.append(
                    RepairTrace(
                        repair_no=repair_no,
                        clue_text_raw=clue_text_raw,
                        clue_text_normalized=logical_result.normalized_text,
                        logical_result=logical_result,
                        llm_result=llm_result,
                        merged_result=merged,
                        latency_ms=clue_response.latency_ms + judge_response.latency_ms,
                        prompt_template_id=clue_response.prompt_template_id,
                    )
                )

                if merged.final_verdict != "fail":
                    accepted_clue = clue_text_raw
                    accepted_clues.append(clue_text_raw)
                    attempt_trace.accepted_clue = clue_text_raw
                    attempt_trace.accepted_repair_no = repair_no
                    if attempt_no == 1 and repair_no == 1:
                        first_clue_passed_without_repair = True
                    if repair_no > 1:
                        clue_repaired_successfully = True
                    self.logger.emit(
                        "clue_accepted",
                        batch_id=batch_id,
                        round_id=round_id,
                        card_id=card.id,
                        attempt_no=attempt_no,
                        clue_repair_no=repair_no,
                        clue_text_raw=clue_text_raw,
                        final_judge_verdict=merged.final_verdict,
                        judge_disagreement=merged.judge_disagreement,
                        state="generating_guess",
                        cluer_model_id=cluer_entry.id,
                        guesser_model_id=guesser_entry.id,
                        judge_model_id=judge_entry.id,
                    )
                    last_rejection_feedback = None
                    break

                self.logger.emit(
                    "clue_repair_requested",
                    batch_id=batch_id,
                    round_id=round_id,
                    card_id=card.id,
                    attempt_no=attempt_no,
                    clue_repair_no=repair_no,
                    final_judge_verdict=merged.final_verdict,
                    logical_violations=logical_result.violations,
                    logical_matched_terms=logical_result.matched_terms,
                    llm_judge_verdict=llm_result.verdict,
                    llm_judge_reasons=llm_result.reasons,
                    judge_disagreement=merged.judge_disagreement,
                    state="repairing_clue",
                    cluer_model_id=cluer_entry.id,
                    guesser_model_id=guesser_entry.id,
                    judge_model_id=judge_entry.id,
                    **_latest_prompt_fields(self.logger, "judge"),
                )
                rejected_clues.append(clue_text_raw)
                last_rejection_feedback = _format_rejection_feedback(logical_result, llm_result)

            if accepted_clue is None:
                attempts.append(attempt_trace)
                terminal_reason = "clue_not_repaired"
                self.logger.emit(
                    "round_finished",
                    batch_id=batch_id,
                    round_id=round_id,
                    card_id=card.id,
                    state="round_finished",
                    terminal_reason=terminal_reason,
                    cluer_model_id=cluer_entry.id,
                    guesser_model_id=guesser_entry.id,
                    judge_model_id=judge_entry.id,
                )
                break

            self.logger.emit(
                "guess_started",
                batch_id=batch_id,
                round_id=round_id,
                card_id=card.id,
                attempt_no=attempt_no,
                role="guesser",
                state="generating_guess",
                cluer_model_id=cluer_entry.id,
                guesser_model_id=guesser_entry.id,
                judge_model_id=judge_entry.id,
            )
            guess_response = self.model_manager.generate(
                model_entry=guesser_entry,
                messages=build_guesser_messages(
                    card=card,
                    current_clue=accepted_clue,
                    accepted_clues=accepted_clues,
                    wrong_guesses=wrong_guesses,
                    attempt_no=attempt_no,
                ),
                generation_params=self.settings.generation.guesser,
                runtime_policy=runtime_policy,
                device_preference=self.settings.device_preference,
                trace_role="guesser",
            )
            guess_text_raw = _extract_single_line(guess_response.text)
            guess_normalized = normalize_text(guess_text_raw)
            total_latency_ms += guess_response.latency_ms
            attempt_trace.guess = GuessTrace(
                attempt_no=attempt_no,
                guess_text_raw=guess_text_raw,
                guess_text_normalized=guess_normalized,
                latency_ms=guess_response.latency_ms,
                prompt_template_id=guess_response.prompt_template_id,
            )
            attempts.append(attempt_trace)

            self.logger.emit(
                "guess_generated",
                batch_id=batch_id,
                round_id=round_id,
                card_id=card.id,
                attempt_no=attempt_no,
                latency_ms=guess_response.latency_ms,
                prompt_tokens=guess_response.prompt_tokens,
                completion_tokens=guess_response.completion_tokens,
                guess_text_raw=guess_text_raw,
                guess_text_normalized=guess_normalized,
                prompt_template_id=guess_response.prompt_template_id,
                role="guesser",
                state="round_finished" if _guess_matches(guess_text_raw, card) else "generating_clue",
                cluer_model_id=cluer_entry.id,
                guesser_model_id=guesser_entry.id,
                judge_model_id=judge_entry.id,
                **_latest_prompt_fields(self.logger, "guesser"),
            )

            if _guess_matches(guess_text_raw, card):
                solved = True
                solved_on_attempt = attempt_no
                terminal_reason = "solved"
                self.logger.emit(
                    "round_finished",
                    batch_id=batch_id,
                    round_id=round_id,
                    card_id=card.id,
                    state="round_finished",
                    terminal_reason=terminal_reason,
                    cluer_model_id=cluer_entry.id,
                    guesser_model_id=guesser_entry.id,
                    judge_model_id=judge_entry.id,
                )
                break

            wrong_guesses.append(guess_text_raw)

        result = RoundResult(
            run_id=self.logger.run_id,
            round_id=round_id,
            card=card,
            solved=solved,
            solved_on_attempt=solved_on_attempt,
            total_guess_attempts_used=len(attempts),
            total_clue_repairs=total_clue_repairs,
            first_clue_passed_without_repair=first_clue_passed_without_repair,
            clue_repaired_successfully=clue_repaired_successfully,
            clue_not_repaired=terminal_reason == "clue_not_repaired",
            terminal_reason=terminal_reason,
            attempts=attempts,
            cluer_model_id=cluer_entry.id,
            guesser_model_id=guesser_entry.id,
            judge_model_id=judge_entry.id,
            total_latency_ms=round(total_latency_ms, 2),
        )
        self.logger.record_round_summary(
            RoundSummaryRecord(
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
        )
        return result


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
