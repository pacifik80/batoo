from __future__ import annotations

import json

from taboo_arena.models.registry import ModelEntry
from taboo_arena.prompts.store import (
    load_prompt_fragment,
    render_prompt_override_messages,
)
from taboo_arena.prompts.tasks import (
    build_clue_judge_messages,
    build_cluer_messages,
)


def _entry(profile_id: str, *, override_id: str | None = None) -> ModelEntry:
    return ModelEntry(
        id=f"test-{profile_id}",
        display_name=f"test-{profile_id}",
        backend="transformers_safetensors",
        repo_id="test/test",
        architecture_family="test",
        chat_template_id="generic_completion",
        prompt_profile_id=profile_id,
        prompt_override_id=override_id,
        supports_system_prompt=True,
        roles_supported=["cluer", "guesser", "judge"],
        languages=["en"],
    )


def test_shared_rule_wording_propagates_across_profiles(sample_card) -> None:
    shared_rule = load_prompt_fragment("wording")["cluer_hide_target_and_blocked_terms"]

    compact = build_cluer_messages(
        card=sample_card,
        accepted_clues=[],
        rejected_clues=[],
        wrong_guesses=[],
        attempt_no=1,
        repair_no=1,
        allowed_angles=["type", "use", "context"],
        blocked_terms=["bear"],
        blocked_prior_clues=[],
        blocked_angles=[],
        model_entry=_entry("compact_small"),
    )[0].content
    standard = build_cluer_messages(
        card=sample_card,
        accepted_clues=[],
        rejected_clues=[],
        wrong_guesses=[],
        attempt_no=1,
        repair_no=1,
        allowed_angles=["type", "use", "context"],
        blocked_terms=["bear"],
        blocked_prior_clues=[],
        blocked_angles=[],
        model_entry=_entry("standard"),
    )[0].content

    assert shared_rule in compact
    assert shared_rule in standard


def test_registry_selected_prompt_profile_changes_packaging_not_core_semantics(sample_card) -> None:
    compact = build_cluer_messages(
        card=sample_card,
        accepted_clues=[],
        rejected_clues=[],
        wrong_guesses=[],
        attempt_no=1,
        repair_no=1,
        allowed_angles=["type", "use", "context"],
        blocked_terms=["bear"],
        blocked_prior_clues=[],
        blocked_angles=[],
        model_entry=_entry("compact_small"),
    )[0].content
    standard = build_cluer_messages(
        card=sample_card,
        accepted_clues=[],
        rejected_clues=[],
        wrong_guesses=[],
        attempt_no=1,
        repair_no=1,
        allowed_angles=["type", "use", "context"],
        blocked_terms=["bear"],
        blocked_prior_clues=[],
        blocked_angles=[],
        model_entry=_entry("standard"),
    )[0].content

    assert "You are the Cluer in a local Taboo benchmark." in compact
    assert "You are the Cluer in a local Taboo benchmark." in standard
    assert "Return exactly one short clue candidate for each allowed angle." in compact
    assert "Return exactly one short clue candidate for each allowed angle." in standard
    assert compact != standard
    assert "CONTROL_STATE" in compact
    assert 'taboo_words=["grizzly", "honey", "pooh"]' in compact
    assert "Current state" in standard
    assert '- forbidden_words: ["grizzly", "honey", "pooh"]' in standard


def test_compact_small_cluer_prompt_excludes_toxic_controller_phrases(sample_card) -> None:
    content = build_cluer_messages(
        card=sample_card,
        accepted_clues=[],
        rejected_clues=[],
        wrong_guesses=[],
        attempt_no=2,
        repair_no=3,
        allowed_angles=["type", "use", "context"],
        blocked_terms=["bear"],
        blocked_prior_clues=["forest giant"],
        blocked_angles=["effect"],
        repair_feedback_json=json.dumps({"reason_codes": ["repeated_rejected_clue"]}),
        model_entry=_entry("compact_small"),
    )[0].content

    assert "Internal repair cycle" not in content
    assert "Structured repair feedback" not in content
    assert "Pivot away from blocked terms and blocked angles" not in content
    assert "CONTROL_STATE" in content
    assert "revision_notes=" in content


def test_judge_reason_code_enum_is_centralized_and_consistent(sample_card) -> None:
    reason_codes = load_prompt_fragment("judge_reason_codes")
    clue_content = build_clue_judge_messages(
        card=sample_card,
        clue_draft="forest giant",
        accepted_clues=[],
        rejected_clues=[],
        attempt_no=1,
        model_entry=_entry("strict_judge"),
    )[0].content

    assert "|".join(reason_codes["clue_block"]) in clue_content
    assert "|".join(reason_codes["clue_warning"]) in clue_content


def test_prompt_override_is_optional_escape_hatch(monkeypatch, tmp_path) -> None:
    override_root = tmp_path / "prompts"
    override_dir = override_root / "overrides" / "test_override"
    override_dir.mkdir(parents=True)
    (override_dir / "cluer.json").write_text(
        json.dumps(
            {
                "id": "override-cluer",
                "messages": [
                    {
                        "role": "user",
                        "template": "OVERRIDE role={{ profile_id }} target={{ target }}",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    import taboo_arena.prompts.store as prompt_store

    monkeypatch.setattr(prompt_store, "get_prompt_root", lambda: override_root)

    rendered = render_prompt_override_messages(
        "test_override",
        "cluer",
        {"profile_id": "compact_small", "target": "Bear"},
    )

    assert rendered is not None
    assert rendered[0].content == "OVERRIDE role=compact_small target=Bear"
