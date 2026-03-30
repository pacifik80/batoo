from __future__ import annotations

from taboo_arena.prompts.store import (
    get_fragment_path,
    get_profile_path,
    get_prompt_override_path,
    get_role_spec_path,
    load_prompt_fragment,
    load_prompt_profile,
    load_role_task_spec,
    render_prompt_override_messages,
)


def test_layered_prompt_files_exist_and_validate() -> None:
    for spec_id in ["cluer_base", "guesser_base", "judge_clue_base", "judge_guess_base"]:
        path = get_role_spec_path(spec_id)
        spec = load_role_task_spec(spec_id)

        assert path.exists()
        assert spec.id == spec_id

    for profile_id in ["compact_small", "standard", "strict_judge"]:
        path = get_profile_path(profile_id)
        profile = load_prompt_profile(profile_id)

        assert path.exists()
        assert profile.id == profile_id

    for fragment_id in ["angle_enum", "judge_reason_codes", "output_contracts", "wording"]:
        path = get_fragment_path(fragment_id)
        payload = load_prompt_fragment(fragment_id)

        assert path.exists()
        assert payload


def test_prompt_override_path_and_missing_override_render_are_safe() -> None:
    path = get_prompt_override_path("does-not-exist", "cluer")

    assert path.name == "cluer.json"
    assert render_prompt_override_messages("does-not-exist", "cluer", {"target": "Bear"}) is None
