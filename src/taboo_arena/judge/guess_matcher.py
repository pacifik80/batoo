"""Deterministic guess canonicalization and repeat filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from taboo_arena.utils.normalization import dedupe_preserve_order, normalize_text, strip_punctuation
from taboo_arena.utils.structured_payloads import looks_like_structured_payload

_SPLIT_PATTERN = re.compile(r"\s*(?:\bor\b|/|,|;|\|)\s*")


class GuessMatchStatus(StrEnum):
    """Outcome of matching one visible guess against the target."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    CORRECT_WITH_EXTRA_TEXT = "correct_with_extra_text"
    CORRECT_MULTI_CANDIDATE = "correct_multi_candidate"
    AMBIGUOUS_GUESS = "ambiguous_guess"


@dataclass(slots=True)
class GuessAnalysis:
    """Normalized view of one raw guess string."""

    raw_text: str
    normalized_text: str
    candidate_spans: list[str]
    candidate_keys: list[str]
    wrapper_flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GuessMatchResult:
    """Match result for one raw guess string against a target."""

    status: GuessMatchStatus
    reason: str
    analysis: GuessAnalysis
    matched_candidate: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_correct(self) -> bool:
        return self.status in {
            GuessMatchStatus.CORRECT,
            GuessMatchStatus.CORRECT_WITH_EXTRA_TEXT,
            GuessMatchStatus.CORRECT_MULTI_CANDIDATE,
        }


@dataclass(slots=True)
class GuessCandidateEvaluation:
    """Evaluation of one shortlist candidate before it becomes visible."""

    guess_text_raw: str
    analysis: GuessAnalysis
    match_result: GuessMatchResult
    valid_new_keys: list[str]
    invalid_reason: str | None = None
    repeated_against: list[str] = field(default_factory=list)

    @property
    def is_valid_visible_candidate(self) -> bool:
        return self.invalid_reason is None


class GuessCanonicalizer:
    """Canonicalize guesses and decide whether they count as correct."""

    _wrappers: tuple[tuple[str, str], ...] = (
        ("i think it is ", "i_think_it_is"),
        ("i think it's ", "i_think_its"),
        ("i think its ", "i_think_its"),
        ("my guess is ", "my_guess_is"),
        ("the answer is ", "the_answer_is"),
        ("could it be ", "could_it_be"),
        ("it is ", "it_is"),
        ("it's ", "its"),
        ("its ", "its"),
        ("maybe ", "maybe"),
        ("perhaps ", "perhaps"),
        ("the ", "article"),
        ("a ", "article"),
        ("an ", "article"),
    )

    def analyze(self, guess_text: str) -> GuessAnalysis:
        """Normalize a raw guess into candidate answer spans."""
        normalized = normalize_text(guess_text)
        if not normalized:
            return GuessAnalysis(
                raw_text=guess_text,
                normalized_text="",
                candidate_spans=[],
                candidate_keys=[],
            )

        wrapped = normalized
        wrapper_flags: list[str] = []
        while True:
            matched_wrapper = False
            for prefix, flag in self._wrappers:
                if wrapped.startswith(prefix):
                    wrapped = wrapped[len(prefix) :].strip()
                    wrapper_flags.append(flag)
                    matched_wrapper = True
                    break
            if not matched_wrapper:
                break

        if not wrapped:
            return GuessAnalysis(
                raw_text=guess_text,
                normalized_text=normalized,
                candidate_spans=[],
                candidate_keys=[],
                wrapper_flags=dedupe_preserve_order(wrapper_flags),
            )

        raw_candidates = [segment.strip() for segment in _SPLIT_PATTERN.split(wrapped) if segment.strip()]
        if not raw_candidates:
            raw_candidates = [wrapped]

        candidate_spans: list[str] = []
        candidate_keys: list[str] = []
        for segment in raw_candidates:
            candidate = segment
            while True:
                article_removed = False
                for article in ("the ", "a ", "an "):
                    if candidate.startswith(article):
                        candidate = candidate[len(article) :].strip()
                        wrapper_flags.append("article")
                        article_removed = True
                        break
                if not article_removed:
                    break
            candidate_key = strip_punctuation(candidate)
            if not candidate_key:
                continue
            candidate_spans.append(candidate)
            candidate_keys.append(candidate_key)

        return GuessAnalysis(
            raw_text=guess_text,
            normalized_text=normalized,
            candidate_spans=dedupe_preserve_order(candidate_spans),
            candidate_keys=dedupe_preserve_order(candidate_keys),
            wrapper_flags=dedupe_preserve_order(wrapper_flags),
        )

    def match(self, guess_text: str, target: str) -> GuessMatchResult:
        """Return whether the guess counts as correct for the target."""
        analysis = self.analyze(guess_text)
        target_key = strip_punctuation(target)
        if not analysis.candidate_keys or not target_key:
            return GuessMatchResult(
                status=GuessMatchStatus.AMBIGUOUS_GUESS,
                reason="no_candidate_spans",
                analysis=analysis,
            )

        matched_candidate: str | None = None
        matched_key: str | None = None
        for candidate_span, candidate_key in zip(analysis.candidate_spans, analysis.candidate_keys, strict=False):
            if candidate_key == target_key or self._contains_target_phrase(candidate_key, target_key):
                matched_candidate = candidate_span
                matched_key = candidate_key
                break

        if matched_candidate is None:
            return GuessMatchResult(
                status=GuessMatchStatus.INCORRECT,
                reason="target_not_present",
                analysis=analysis,
            )

        warnings: list[str] = []
        if len(analysis.candidate_keys) > 1:
            warnings.append("multi_candidate_guess")
            return GuessMatchResult(
                status=GuessMatchStatus.CORRECT_MULTI_CANDIDATE,
                reason="multi_candidate_contains_target",
                analysis=analysis,
                matched_candidate=matched_candidate,
                warnings=warnings,
            )
        if matched_key != target_key:
            warnings.append("extra_text_around_target")
            return GuessMatchResult(
                status=GuessMatchStatus.CORRECT_WITH_EXTRA_TEXT,
                reason="contained_target_match",
                analysis=analysis,
                matched_candidate=matched_candidate,
                warnings=warnings,
            )
        if analysis.wrapper_flags:
            warnings.append("normalized_wrapper_match")
            reason = "normalized_wrapper_match"
        else:
            reason = "exact_match"
        return GuessMatchResult(
            status=GuessMatchStatus.CORRECT,
            reason=reason,
            analysis=analysis,
            matched_candidate=matched_candidate,
            warnings=warnings,
        )

    def evaluate_shortlist(
        self,
        guesses: list[str],
        *,
        target: str,
        previous_wrong_guesses: list[str],
    ) -> list[GuessCandidateEvaluation]:
        """Score a shortlist and mark which candidates are valid new visible guesses."""
        previous_keys = self._previous_wrong_guess_keys(previous_wrong_guesses)
        seen_shortlist_keys: set[str] = set()
        evaluations: list[GuessCandidateEvaluation] = []
        for guess_text in guesses:
            if looks_like_structured_payload(guess_text):
                analysis = self.analyze(guess_text)
                match_result = self.match("", target)
                match_result.reason = "structured_payload_detected"
                evaluations.append(
                    GuessCandidateEvaluation(
                        guess_text_raw=guess_text,
                        analysis=analysis,
                        match_result=match_result,
                        valid_new_keys=[],
                        invalid_reason="structured_payload_guess",
                    )
                )
                continue
            analysis = self.analyze(guess_text)
            match_result = self.match(guess_text, target)
            if not analysis.candidate_keys:
                evaluations.append(
                    GuessCandidateEvaluation(
                        guess_text_raw=guess_text,
                        analysis=analysis,
                        match_result=match_result,
                        valid_new_keys=[],
                        invalid_reason="empty_or_non_answer",
                    )
                )
                continue

            repeated_against = sorted(
                {
                    previous_keys[key]
                    for key in analysis.candidate_keys
                    if key in previous_keys
                }
            )
            valid_new_keys = [
                key
                for key in analysis.candidate_keys
                if key not in previous_keys and key not in seen_shortlist_keys
            ]
            invalid_reason: str | None = None
            if not valid_new_keys:
                invalid_reason = (
                    "repeat_previous_wrong_guess"
                    if repeated_against
                    else "duplicate_shortlist_guess"
                )
            seen_shortlist_keys.update(analysis.candidate_keys)
            evaluations.append(
                GuessCandidateEvaluation(
                    guess_text_raw=guess_text,
                    analysis=analysis,
                    match_result=match_result,
                    valid_new_keys=valid_new_keys,
                    invalid_reason=invalid_reason,
                    repeated_against=repeated_against,
                )
            )
        return evaluations

    @staticmethod
    def _contains_target_phrase(candidate_key: str, target_key: str) -> bool:
        return re.search(rf"\b{re.escape(target_key)}\b", candidate_key) is not None

    def _previous_wrong_guess_keys(self, guesses: list[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for guess in guesses:
            analysis = self.analyze(guess)
            for key in analysis.candidate_keys:
                mapping.setdefault(key, guess)
        return mapping
