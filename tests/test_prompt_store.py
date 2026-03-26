from __future__ import annotations

from taboo_arena.prompts.store import (
    get_prompt_path,
    load_prompt_definition,
    render_prompt_messages,
)


def test_prompt_files_exist_and_validate() -> None:
    for prompt_id in ["cluer", "guesser", "judge"]:
        path = get_prompt_path(prompt_id)
        definition = load_prompt_definition(prompt_id)

        assert path.exists()
        assert definition.id == prompt_id
        assert len(definition.messages) >= 1


def test_render_prompt_messages_uses_json_files() -> None:
    messages = render_prompt_messages(
        "cluer",
        {
            "target": "Bear",
            "forbidden_words_csv": "grizzly, honey, pooh",
            "aliases_csv": "Grizzly Bear",
            "previous_accepted_clues_json": "[\"forest mammal\"]",
            "previous_rejected_clues_json": "[\"cave dweller\"]",
            "previous_wrong_guesses_json": "[\"wolf\"]",
            "attempt_no": 2,
            "repair_no": 1,
            "last_rejection_feedback": "Rules failed: token_match.",
        },
    )

    assert len(messages) == 1
    assert messages[0].content.startswith("You are the Cluer")
    assert "Target: Bear." in messages[0].content
    assert "Latest rejection feedback: Rules failed: token_match." in messages[0].content
