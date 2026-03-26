from __future__ import annotations

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.judge.llm import LLMJudgeResult
from taboo_arena.judge.logical import LogicalValidator
from taboo_arena.judge.merge import merge_judge_results
from taboo_arena.prompts.tasks import (
    build_cluer_messages,
    build_guesser_messages,
    build_judge_messages,
)


def test_logical_validator_catches_target_and_repeat(sample_card) -> None:
    validator = LogicalValidator()
    result = validator.validate(
        "Bear cave",
        card=sample_card,
        previous_accepted_clues=["forest mammal"],
    )
    assert result.verdict == "fail"
    assert "whole_word_match" in result.violations or "token_match" in result.violations


def test_logical_validator_catches_stem_match(sample_card) -> None:
    validator = LogicalValidator()
    result = validator.validate(
        "grizzlies roam",
        card=sample_card,
        previous_accepted_clues=[],
    )
    assert result.verdict == "fail"
    assert "stem_match" in result.violations or "substring_match" in result.violations


def test_merge_logic_defaults_to_warning_on_uncertain(sample_card) -> None:
    validator = LogicalValidator()
    logical = validator.validate("forest giant", card=sample_card, previous_accepted_clues=[])
    llm = LLMJudgeResult(verdict="uncertain", reasons=["close"], confidence=0.4)
    merged = merge_judge_results(logical, llm, block_on_uncertain=False)
    assert merged.final_verdict == "pass_with_warning"
    blocked = merge_judge_results(logical, llm, block_on_uncertain=True)
    assert blocked.final_verdict == "fail"


def test_logical_validator_catches_repeated_rejected_clue(sample_card) -> None:
    validator = LogicalValidator()
    result = validator.validate(
        "forest giant",
        card=sample_card,
        previous_accepted_clues=[],
        previous_rejected_clues=["forest giant"],
    )
    assert result.verdict == "fail"
    assert "repeated_rejected_clue" in result.violations


def test_logical_validator_does_not_fail_on_single_token_overlap_from_multiword_taboo() -> None:
    validator = LogicalValidator()
    card = CardRecord(
        id="people:harrison-ford:0097",
        target="Harrison Ford",
        taboo_hard=["Han Solo", "Star Wars", "Indiana Jones", "actor", "Hollywood"],
        aliases=[],
        source_category="people",
        source_repo="Kovah/Taboo-Data",
        source_ref="main",
        category_label="People",
    )
    result = validator.validate(
        "\"American film star known for iconic roles in 'Raiders of the Lost Ark' and 'Blade Runner'.\"",
        card=card,
        previous_accepted_clues=[],
    )

    assert result.verdict == "pass"
    assert "Star Wars" not in result.matched_terms


def test_cluer_prompt_includes_rejection_feedback_and_rejected_clues(sample_card) -> None:
    messages = build_cluer_messages(
        card=sample_card,
        accepted_clues=["mammal"],
        rejected_clues=["forest giant"],
        wrong_guesses=["wolf"],
        attempt_no=2,
        repair_no=3,
        last_rejection_feedback="Rules failed: repeated_rejected_clue.",
    )

    assert len(messages) == 1
    content = messages[0].content
    assert "Previous rejected clues" in content
    assert "forest giant" in content
    assert "Latest rejection feedback" in content
    assert "repeated_rejected_clue" in content


def test_cluer_prompt_pushes_for_concrete_type_anchored_clues(sample_card) -> None:
    messages = build_cluer_messages(
        card=sample_card,
        accepted_clues=[],
        rejected_clues=[],
        wrong_guesses=[],
        attempt_no=1,
        repair_no=1,
    )

    assert len(messages) == 1
    content = messages[0].content
    assert "usually 6 to 18 words" in content
    assert "person, place, animal, object, event, profession" in content
    assert "more informative than a one-word abstract label" in content
    assert "Output contract: return only the final clue text." in content


def test_guesser_prompt_prefers_concrete_target_over_broad_concept(sample_card) -> None:
    messages = build_guesser_messages(
        card=sample_card,
        current_clue="large forest mammal known for hibernation",
        accepted_clues=["large forest mammal known for hibernation"],
        wrong_guesses=["animal"],
        attempt_no=2,
    )

    assert len(messages) == 1
    content = messages[0].content
    assert "Prefer a concrete target over a broad concept" in content
    assert "Do not repeat a previous wrong guess" in content
    assert "Current accepted clue: large forest mammal known for hibernation." in content


def test_judge_prompt_discourages_overstrict_descriptive_failures(sample_card) -> None:
    messages = build_judge_messages(
        card=sample_card,
        clue_draft="forest giant",
        accepted_clues=[],
        wrong_guesses=[],
        attempt_no=1,
    )

    assert len(messages) == 1
    content = messages[0].content
    assert "Do not fail a clue merely because it is informative" in content
    assert "return 'uncertain' instead of 'fail'" in content
    assert "Output contract: return strict JSON only" in content
