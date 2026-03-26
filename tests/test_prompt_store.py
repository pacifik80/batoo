from __future__ import annotations

from taboo_arena.prompts.store import (
    get_prompt_path,
    load_prompt_definition,
    render_prompt_messages,
)


def test_prompt_files_exist_and_validate() -> None:
    for prompt_id in [
        "cluer",
        "cluer_candidates",
        "cluer_repair",
        "guesser",
        "guesser_candidates",
        "judge",
        "judge_clue",
        "judge_guess",
    ]:
        path = get_prompt_path(prompt_id)
        definition = load_prompt_definition(prompt_id)

        assert path.exists()
        assert definition.id == prompt_id
        assert len(definition.messages) >= 1


def test_render_prompt_messages_uses_json_files() -> None:
    messages = render_prompt_messages(
        "cluer_candidates",
        {
            "target": "Bear",
            "forbidden_words_json": "[\"grizzly\", \"honey\", \"pooh\"]",
            "allowed_angles_json": "[\"type\", \"use\", \"context\"]",
            "blocked_terms_json": "[\"bear\"]",
            "blocked_prior_clues_json": "[\"forest mammal\"]",
            "blocked_angles_json": "[\"effect\"]",
            "previous_accepted_clues_json": "[\"forest mammal\"]",
            "previous_rejected_clues_json": "[\"cave dweller\"]",
            "previous_wrong_guesses_json": "[\"wolf\"]",
            "attempt_no": 2,
            "repair_no": 1,
            "repair_feedback_json": "{}",
            "output_schema_json": "{\"candidates\":[{\"angle\":\"type\",\"clue\":\"string\"}]}",
        },
    )

    assert len(messages) == 1
    assert messages[0].content.startswith("You are the Cluer")
    assert "Target: Bear." in messages[0].content
    assert "Allowed angles" in messages[0].content
