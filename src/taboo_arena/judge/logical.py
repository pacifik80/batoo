"""Deterministic clue validation."""

from __future__ import annotations

import re

from nltk.stem import PorterStemmer
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.config import LogicalValidatorSettings
from taboo_arena.utils.normalization import (
    dedupe_preserve_order,
    normalize_text,
    strip_punctuation,
    tokenize,
)

VALIDATOR_VERSION = "logical_v1"


class LogicalValidationResult(BaseModel):
    """Structured validator result."""

    verdict: str
    normalized_text: str
    violations: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    validator_version: str = VALIDATOR_VERSION


class LogicalValidator:
    """Check a clue against the benchmark's deterministic rules."""

    def __init__(self, settings: LogicalValidatorSettings | None = None) -> None:
        self.settings = settings or LogicalValidatorSettings()
        self.stemmer = PorterStemmer()

    def validate(
        self,
        clue_text: str,
        *,
        card: CardRecord,
        previous_accepted_clues: list[str],
        previous_rejected_clues: list[str] | None = None,
    ) -> LogicalValidationResult:
        """Validate a clue draft."""
        normalized = normalize_text(clue_text)
        stripped = strip_punctuation(clue_text)
        clue_tokens = tokenize(clue_text)
        clue_stems = {self.stemmer.stem(token) for token in clue_tokens}

        violations: list[str] = []
        matched_terms: list[str] = []

        if not normalized:
            violations.append("empty_clue")
        if len(clue_tokens) < self.settings.min_token_count:
            violations.append("clue_too_short")

        previous_normalized = {normalize_text(item) for item in previous_accepted_clues}
        previous_rejected_normalized = {
            normalize_text(item) for item in (previous_rejected_clues or [])
        }
        if normalized and normalized in previous_normalized:
            violations.append("repeated_clue")
        if normalized and normalized in previous_rejected_normalized:
            violations.append("repeated_rejected_clue")

        comparison_terms = [card.target, *card.taboo_hard, *card.aliases]
        for term in comparison_terms:
            matched = self._match_term(
                term=term,
                normalized_clue=normalized,
                stripped_clue=stripped,
                clue_tokens=clue_tokens,
                clue_stems=clue_stems,
            )
            if matched is None:
                continue
            violations.append(matched)
            matched_terms.append(term)

        if self.settings.check_similarity_to_previous:
            similar_previous = self._find_similar_previous_clue(normalized, previous_accepted_clues)
            if similar_previous is not None:
                violations.append("too_similar_to_previous_clue")
                matched_terms.append(similar_previous)

        return LogicalValidationResult(
            verdict="fail" if violations else "pass",
            normalized_text=normalized,
            violations=dedupe_preserve_order(violations),
            matched_terms=dedupe_preserve_order(matched_terms),
        )

    def _match_term(
        self,
        *,
        term: str,
        normalized_clue: str,
        stripped_clue: str,
        clue_tokens: list[str],
        clue_stems: set[str],
    ) -> str | None:
        term_normalized = normalize_text(term)
        term_stripped = strip_punctuation(term)
        term_tokens = tokenize(term)
        term_stems = {self.stemmer.stem(token) for token in term_tokens}
        clue_token_set = set(clue_tokens)

        if normalized_clue == term_normalized:
            return "exact_term_match"
        if stripped_clue and stripped_clue == term_stripped:
            return "punctuation_stripped_match"
        if self.settings.check_whole_word and term_normalized and self._contains_whole_phrase(normalized_clue, term_normalized):
            return "whole_word_match"
        if self.settings.check_token_match and self._has_component_match(
            term_components=set(term_tokens),
            clue_components=clue_token_set,
        ):
            return "token_match"
        if self.settings.check_substring and self._contains_substring(stripped_clue, term_stripped):
            return "substring_match"
        if self.settings.check_stemming and self._has_component_match(
            term_components=term_stems,
            clue_components=clue_stems,
        ):
            return "stem_match"
        return None

    @staticmethod
    def _has_component_match(
        *,
        term_components: set[str],
        clue_components: set[str],
    ) -> bool:
        """Match multi-word taboo phrases more carefully than single-word terms.

        Single-word terms still fail on any overlap. Multi-word phrases require at
        least two overlapping components so generic one-word overlaps like
        "star" vs "Star Wars" do not produce hard false positives.
        """
        overlap = term_components.intersection(clue_components)
        if not overlap:
            return False
        if len(term_components) == 1:
            return True
        return len(overlap) >= 2

    @staticmethod
    def _contains_whole_phrase(text: str, phrase: str) -> bool:
        return re.search(rf"\b{re.escape(phrase)}\b", text) is not None

    @staticmethod
    def _contains_substring(left: str, right: str) -> bool:
        if not left or not right:
            return False
        if len(right) < 4 and len(left) < 4:
            return False
        return right in left or left in right

    def _find_similar_previous_clue(self, clue: str, previous_accepted_clues: list[str]) -> str | None:
        for previous in previous_accepted_clues:
            score = fuzz.token_sort_ratio(clue, normalize_text(previous))
            if score >= self.settings.similarity_threshold:
                return previous
        return None
