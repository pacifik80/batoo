from __future__ import annotations

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.judge.llm import LLMJudgeResult, _normalize_clue_judge_codes
from taboo_arena.judge.logical import LogicalValidator
from taboo_arena.judge.merge import merge_judge_results
from taboo_arena.models.registry import ModelEntry
from taboo_arena.prompts.store import load_prompt_fragment
from taboo_arena.prompts.tasks import (
    build_clue_judge_messages,
    build_cluer_messages,
    build_guess_judge_messages,
    build_guesser_messages,
)


def _entry(profile_id: str) -> ModelEntry:
    return ModelEntry(
        id=f"test-{profile_id}",
        display_name=f"test-{profile_id}",
        backend="transformers_safetensors",
        repo_id="test/test",
        architecture_family="test",
        chat_template_id="generic_completion",
        prompt_profile_id=profile_id,
        supports_system_prompt=True,
        roles_supported=["cluer", "guesser", "judge"],
        languages=["en"],
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


def test_merge_logic_preserves_warning_only_clue_judge_signal(sample_card) -> None:
    validator = LogicalValidator()
    logical = validator.validate("forest giant", card=sample_card, previous_accepted_clues=[])
    llm = LLMJudgeResult(verdict="pass", warnings=["close_but_allowed"], confidence=1.0)

    merged = merge_judge_results(logical, llm, block_on_uncertain=False)

    assert merged.final_verdict == "pass_with_warning"
    assert merged.llm_judge_warnings == ["close_but_allowed"]


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
        allowed_angles=["type", "context", "use"],
        blocked_terms=["bear", "grizzly"],
        blocked_prior_clues=["forest giant"],
        blocked_angles=["effect"],
        repair_feedback_json='{"reason_codes":["repeated_rejected_clue"]}',
        model_entry=_entry("standard"),
    )

    assert len(messages) == 1
    content = messages[0].content
    assert "Current state" in content
    assert '- allowed_angles: ["type", "context", "use"]' in content
    assert "forest giant" in content
    assert "revision_notes" in content
    assert "repeated_rejected_clue" in content
    assert "Internal repair cycle" not in content
    assert "Structured repair feedback" not in content


def test_cluer_prompt_pushes_for_concrete_type_anchored_clues(sample_card) -> None:
    messages = build_cluer_messages(
        card=sample_card,
        accepted_clues=[],
        rejected_clues=[],
        wrong_guesses=[],
        attempt_no=1,
        repair_no=1,
        allowed_angles=["type", "use", "context"],
        blocked_terms=["bear", "grizzly"],
        blocked_prior_clues=[],
        blocked_angles=[],
        model_entry=_entry("compact_small"),
    )

    assert len(messages) == 1
    content = messages[0].content
    assert "TASK" in content
    assert "CONTROL_STATE" in content
    assert "Return exactly one short clue candidate for each allowed angle" in content
    assert "Use only the provided angle labels" in content
    assert 'allowed_angles=["type", "use", "context"]' in content
    assert "Return strict JSON only." in content
    assert "Pivot away from blocked terms and blocked angles" not in content


def test_guesser_prompt_prefers_concrete_target_over_broad_concept(sample_card) -> None:
    messages = build_guesser_messages(
        card=sample_card,
        current_clue="large forest mammal known for hibernation",
        accepted_clues=["large forest mammal known for hibernation"],
        wrong_guesses=["animal"],
        attempt_no=2,
        model_entry=_entry("compact_small"),
    )

    assert len(messages) == 1
    content = messages[0].content
    assert "Prefer a concrete target over a broad concept" in content
    assert "Do not repeat a previous wrong guess" in content
    assert "current_clue=large forest mammal known for hibernation" in content
    assert "type_hint=Animals" in content
    assert "Return strict JSON only." in content


def test_clue_judge_prompt_discourages_overstrict_descriptive_failures(sample_card) -> None:
    messages = build_clue_judge_messages(
        card=sample_card,
        clue_draft="forest giant",
        accepted_clues=[],
        rejected_clues=[],
        attempt_no=1,
        model_entry=_entry("strict_judge"),
    )

    assert len(messages) == 1
    content = messages[0].content
    assert "Block only for actual game-rule violations" in content
    assert "Do not act as a broad semantic critic" in content
    assert "Do not block on near-explicit paraphrase" in content
    assert '"allow":true' in content
    assert "Return strict JSON only." in content
    assert "|".join(load_prompt_fragment("judge_reason_codes")["clue_block"]) in content


def test_guess_judge_prompt_accepts_target_inside_visible_guess(sample_card) -> None:
    messages = build_guess_judge_messages(
        card=sample_card,
        guess_text="maybe bear or wolf",
        attempt_no=2,
        match_status="correct_multi_candidate",
        match_reason="multi_candidate_contains_target",
        candidate_spans=["bear", "wolf"],
        warnings=["multi_candidate_guess"],
        model_entry=_entry("strict_judge"),
    )

    assert len(messages) == 1
    content = messages[0].content
    assert "If the target answer appears anywhere in the visible guess" in content
    assert "Do not require perfectly clean formatting" in content
    assert '"correct":false' in content
    assert "Return strict JSON only." in content
    assert "|".join(load_prompt_fragment("judge_reason_codes")["guess"]) in content


def test_unknown_clue_judge_reason_codes_are_demoted_to_warning() -> None:
    normalized_blocks, normalized_warnings = _normalize_clue_judge_codes(["invented_code"], [])

    assert normalized_blocks == []
    assert normalized_warnings == ["unknown_reason_code"]
